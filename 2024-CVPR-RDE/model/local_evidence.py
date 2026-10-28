"""Training-only, rival-conditioned local evidence; no independent parameters."""

import math

import torch
import torch.nn.functional as F


DIFFERENCE_EPS = 1e-3  # Avoid amplifying near-identical caption differences in fp16 encoders.


def validate_evidence_options(candidates=4, weight=0.1, temperature=0.07, margin=0.1):
    if not isinstance(candidates, int) or isinstance(candidates, bool) or candidates < 1:
        raise ValueError("evidence_candidates must be a positive integer")
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("evidence_weight must be finite and positive")
    if not math.isfinite(temperature) or temperature < 1e-4:
        raise ValueError("evidence_temperature must be finite and at least 1e-4")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("evidence_margin must be finite and nonnegative")


def _finite(name, value):
    if not torch.isfinite(value).all().item():
        raise FloatingPointError("mechanism A: nonfinite " + name)


def local_evidence_loss(patches, text_global, image_global, pids, query_mask=None,
                        candidates=4, weight=0.1, temperature=0.07, margin=0.1,
                        return_attention=False):
    """Build sparse (query, rival, image) relations from one encoder forward.

    patches: [B, L, D], excluding CLS; text_global/image_global: [B, D].
    Every same-PID batch image is positive. Hard rivals have a different PID.
    Only discrete selection/masks stop gradients. Scores retain gradients to
    both the patches and the query/rival text encodings. Uncomputed scores are
    zero placeholders and MUST be read with evidence_valid_mask.
    """
    validate_evidence_options(candidates, weight, temperature, margin)
    if patches.ndim != 3 or text_global.ndim != 2 or image_global.ndim != 2:
        raise ValueError("expected patches [B,L,D], text/image globals [B,D]")
    b, length, dim = patches.shape
    if b < 1 or length < 1 or dim < 1 or text_global.shape != (b, dim) or image_global.shape != (b, dim):
        raise ValueError("mechanism A feature shapes are incompatible")
    if pids.ndim != 1 or len(pids) != b or pids.is_floating_point():
        raise ValueError("expected integer PID vector [B]")
    if text_global.device != patches.device or image_global.device != patches.device:
        raise ValueError("mechanism A features must share a device")
    if query_mask is not None and query_mask.shape != (b,):
        raise ValueError("expected CCD query mask [B]")

    with torch.cuda.amp.autocast(enabled=False):
        patches, text_global, image_global = patches.float(), text_global.float(), image_global.float()
        for name, value in (("patches", patches), ("text", text_global), ("image global", image_global)):
            _finite(name, value)
        if query_mask is not None:
            _finite("CCD query mask", query_mask)
        device = patches.device
        pids = pids.to(device)
        positive = pids[:, None].eq(pids[None, :])
        allowed_query = (torch.ones(b, device=device, dtype=torch.bool) if query_mask is None
                         else query_mask.to(device) > 0)
        k = min(candidates, b - 1)
        text = F.normalize(text_global, dim=-1, eps=1e-6)
        # Only hard candidate selection is detached, not the text difference.
        with torch.no_grad():
            coarse = text.detach() @ F.normalize(image_global, dim=-1, eps=1e-6).t()
            values, indices = coarse.masked_fill(positive, float("-inf")).topk(k, dim=1)
            selected = torch.isfinite(values)
            negative_indices = indices.masked_fill(~selected, -1)
        rivals = text[negative_indices.clamp_min(0)]
        delta = text[:, None, :] - rivals
        text_valid = text_global.norm(dim=-1) > 1e-6
        valid = (selected & allowed_query[:, None] & text_valid[:, None] &
                 text_valid[negative_indices.clamp_min(0)] & (delta.norm(dim=-1) > DIFFERENCE_EPS))
        difference = F.normalize(delta, dim=-1, eps=DIFFERENCE_EPS)
        negative_pairs = torch.zeros((b, k, b), device=device, dtype=torch.bool)
        if k:
            negative_pairs.scatter_(2, negative_indices.clamp_min(0).unsqueeze(-1), selected.unsqueeze(-1))
        candidate_mask = (positive[:, None, :] | negative_pairs) & valid[:, :, None]
        # Avoid an overflowing full-tensor sum even when every input element
        # is finite and no relation is eligible for supervision.
        zero = patches.reshape(-1)[0] * 0 + text_global.reshape(-1)[0] * 0
        scores = patches.new_zeros((b, k, b))
        evidence_valid = torch.zeros_like(candidate_mask)
        pair_indices, attentions, score_values, score_offsets, pair_valids = [], [], [], [], []

        if valid.any().item():
            patch = F.normalize(patches, dim=-1, eps=1e-6)
            # Group by image: reuse [L,D] patch storage for every selected
            # relation, never materialize [B,K,B,L,D] or re-encode images.
            for image_index in range(b):
                relations = candidate_mask[:, :, image_index].nonzero(as_tuple=False)
                if relations.shape[0] == 0:
                    continue
                queries = difference[relations[:, 0], relations[:, 1]]
                attention = F.softmax((queries @ patch[image_index].t()) / temperature, dim=-1)
                pooled = attention @ patch[image_index]
                local = F.normalize(pooled, dim=-1, eps=1e-6)
                score_values.append((queries * local).sum(-1))
                score_offsets.append((relations[:, 0] * k + relations[:, 1]) * b + image_index)
                pair_valids.append(pooled.norm(dim=-1) > 1e-6)
                if return_attention:
                    pair_indices.append(torch.cat((relations, relations.new_full((len(relations), 1), image_index)), dim=1))
                    attentions.append(attention)
            offsets = torch.cat(score_offsets)
            scores = scores.reshape(-1).index_add(0, offsets, torch.cat(score_values)).view(b, k, b)
            evidence_valid.view(-1)[offsets] = torch.cat(pair_valids)
            _finite("local scores", scores)

        positive_pairs = evidence_valid & positive[:, None, :]
        negative_valid = (evidence_valid & negative_pairs).any(dim=-1)
        valid_relations = valid & negative_valid & positive_pairs.any(dim=-1)
        positive_pairs = positive_pairs & valid_relations[:, :, None]
        valid_queries = valid_relations.any(dim=-1)
        valid_query_count = valid_queries.sum()
        if valid_relations.any().item():
            negative_score = scores.gather(2, negative_indices.clamp_min(0).unsqueeze(-1))
            penalties = temperature * F.softplus((margin + negative_score - scores) / temperature)
            per_rival = (penalties * positive_pairs).sum(-1) / positive_pairs.sum(-1).clamp_min(1)
            per_query = (per_rival * valid_relations).sum(-1) / valid_relations.sum(-1).clamp_min(1)
            unweighted = per_query.sum() / valid_query_count.clamp_min(1)
        else:
            per_query = patches.new_zeros(b) + zero
            unweighted = zero
        loss = weight * unweighted
        _finite("A loss", loss)
        result = {"loss": loss, "unweighted_loss": unweighted,
                  "per_query_loss": per_query, "valid_query_count": valid_query_count,
                  "pids": pids,
                  "local_scores": scores, "positive_mask": positive,
                  "negative_indices": negative_indices, "selected_negative_mask": selected,
                  "candidate_mask": candidate_mask, "evidence_valid_mask": evidence_valid,
                  "valid_relation_mask": valid_relations,
                  "valid_query_mask": valid_queries,
                  "difference_queries": difference}
        if return_attention:
            result["pair_indices"] = (torch.cat(pair_indices) if pair_indices
                                      else pids.new_empty((0, 3)))
            result["attention_weights"] = (torch.cat(attentions) if attentions
                                           else patches.new_empty((0, length)))
        return result
