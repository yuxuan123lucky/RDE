# Mechanism A: training-only difference-conditioned local evidence

Base: `83a538e4128e55201a4a18b1cbb445c44b7eff24` on
`exp-tbpr-evidence-distill`. This implements only A. There is no distillation
B, adapter, gate, extra annotation, image re-encoding, or inference reranking.

## Interface audit and design

`VisionTransformer.forward` already returns all projected tokens with their
autograd graph. At 384x128, ViT-B/16 with stride 16 returns `[B,193,512]`:
one CLS and 192 patch tokens. `CLIP.encode_text` returns `[B,77,512]`; RDE
selects EOT by the maximum token ID. Neither output is detached in training.
The visual path is `transformer -> ln_post -> visual.proj` for every token,
not an unprojected 768-dimensional patch path. CLIP pretraining aligns its
global CLS/EOT outputs; applying that existing projection to patches does
**not** establish local patch/text semantic alignment. This is an empirical
assumption of A to check on Val, not a consequence of matching dimensions.
`ImageTextDataset` returns training PID, image IDs, caption IDs and batch
indices. At explicit `noisy_rate=0.0`, the original caption pairing is retained.

RDE's `compute_rbs` applies TAL separately to BGE and TSE, weighted by CCD's
`label_hat`. `processor` sums returned keys containing `loss`. A adds one
`evidence_a_loss` to that path. Logs/TensorBoard include A loss and valid-query
counts even when zero. CCD's `compute_per_loss`, GMM, and mask logic
are unchanged. A uses `label_hat > 0` to select supervised query anchors;
candidate identity labels still come only from the training batch's true PID.
Other same-PID images remain positive even if their own caption anchor was
rejected by CCD. Rival rows are mined from all different-PID batch rows,
including CCD-rejected rows, as original batch-relative losses also use the
whole batch as candidates. This is an explicit choice, not a reliability claim.

## Exact formula

Let `p[j,l]` be projected image patches, `v[j]` the original BGE image global,
`t[i]` the original EOT text global, and `y[i]` its training identity. A hat
denotes L2 normalization in float32, with epsilon `1e-6`.
Text differences use epsilon/validity threshold `1e-3` to avoid amplifying
nearly identical caption differences through fp16 encoders.

```
P_i = { j : y[j] == y[i] }                 # ALL batch positives, not just i
H_i = TopK_{n: y[n] != y[i]} stopgrad(t_hat[i]^T v_hat[n])
delta[i,n] = t_hat[i] - t_hat[n]
d[i,n] = normalize(delta[i,n])
a[i,n,j,l] = softmax_l(d[i,n]^T p_hat[j,l] / tau)
e[i,n,j] = normalize(sum_l a[i,n,j,l] * p_hat[j,l])
s[i,n,j] = d[i,n]^T e[i,n,j]

L_A = lambda / N_valid * sum_{i: valid} mean_{n in H_i, j in P_i}
        tau * softplus((margin + s[i,n,n] - s[i,n,j]) / tau)
L_total = L_RDE_BGE + L_RDE_TSE + L_A
```

Means are taken over valid positive evidence and valid rivals separately
per query, then over valid queries (`N_valid`). This corrects the first
draft's query sum: varying CCD acceptance no longer linearly multiplies
the A loss. Original TAL retains its original batch-sum reduction. A's
relative weight is therefore smaller than in that draft; lambda is not
increased to compensate. Changes in which queries are valid can still
change the mean loss. Queries rejected by CCD, queries without a different-PID rival,
zero text differences, and relations without nonzero positive/negative
pooled evidence do not contribute. Invalid relations have masks; they never
fall back to ordinary text-only pooling. An entirely invalid batch returns
a differentiable zero. NaN/Inf inputs, scores and loss raise an error.

Defaults: `K=4`, `lambda=0.1`, `tau=0.07`, `margin=0.1`. Tau is used both
for attention and the smooth ranking penalty to keep the pilot small.
It must be at least `1e-4`. These are proposed pilot hyperparameters,
not author RDE hyperparameters or values tuned on Test. If positive/negative
scores coincide, the weighted A loss is approximately 0.0115 regardless
of the nonzero valid-query count. Weight is applied once inside the helper;
the processor adds the returned weighted loss once. Division by positive,
rival and valid-query counts prevents those counts from simply multiplying
loss scale. An empty valid-query set returns differentiable zero.

The difference against EACH rival conditions both attention and scoring.
Changing the rival can change attention for the same query/image. This is
the mechanism's distinction from plain query-conditioned local alignment;
it is a structural hypothesis, not proof of novelty or retrieval improvement.

## Gradient and inference behavior

- Only hard Top-K selection and PID/CCD validity masks are nondifferentiable.
- A's gradients flow through difference queries to the original query/rival
  text encoder, and through pooled evidence to image projection, patch tokens,
  transformer and convolutional patch embedding. No independent head can
  absorb the auxiliary objective: A adds **zero trainable parameters**.
- TSE-specific MLP parameters do not receive a direct A gradient. The shared
  CLIP encoder changes, so both final RDE branches can change. CCD decisions
  in later epochs can also change indirectly through the updated encoder.
- All A operations run in float32 with autocast disabled; original CLIP/TSE
  precision and optimizer groups remain unchanged.
- `--evidence_mode none` (default, including old YAML) or its `off` alias makes no A calls,
  adds no parameters, consumes no new RNG draws, and retains Baseline state
  keys. `a_only` requires explicit noise 0.0 and TAL.
- A runs only in training `RDE.forward`. It is absent from CCD loss extraction
  and all four inference encoders. `test.py`, `utils/metrics.py`, PID retrieval
  matching and the original fixed `(BGE+TSE)/2` fusion are unchanged.

## Contrast with related methods

- [FLAIR](https://github.com/ExplainableML/flair) learns language-conditioned
  image pooling using sub-captions and supports conditioned retrieval.
  A uses PID-separated hard rivals and a rival-specific text difference,
  with a local identity-relative ranking objective only during training.
- [GOAL](https://openaccess.thecvf.com/content/CVPR2025/html/Choi_GOAL_Global-local_Object_Alignment_Learning_CVPR_2025_paper.html)
  uses Local Image-Sentence Matching and Token Similarity-based Learning for
  lengthy descriptions. A builds batch PID-based rival relations directly
  on existing tokens, without image segmentation or sentence-region matching.
- Original RDE pools attention-selected tokens independently in each modality
  and supervises two retrieval branches with TAL/CCD. A adds contrast-specific
  cross-modal evidence during training and keeps the RDE inference structure.

These differences do not establish publication novelty. Hard-negative mining,
attention pooling, and ranking supervision themselves are established ideas.

## Sparse computation and estimated budget

For each selected `(query i, rival n)`, calculate evidence only for `P_i` and
image `n`. Group relations by image to reuse one `[L,D]` patch tensor; never
gather/repeat a `[B,K,B,L,D]` tensor. The encoder runs exactly once per batch.

Let `E = sum_i |H_i|*(|P_i|+1)` before validity filtering. Extra matching is
approximately `2*E*L*D` multiply-accumulates (attention logits and pooling),
plus small normalization/ranking operations. With B=64, K=4, L=192, D=512,
and one positive per query, E=512: about 0.101 billion MACs / 0.201 GFLOPs
in the forward matching. A conservative E<=B*K*(B+1)=16640 bound is about
6.54 GFLOPs. No-negative batches skip local matching.

Patch normalization alone has 6.29 million float32 elements (~24 MiB per
tensor); sparse attention has E*L elements. Reused patch storage avoids
hundreds of MiB of repeated per-pair patch tensors. Expected extra live
storage is tens to hundreds of MiB, with a conservative sub-GiB planning
budget for this batch size, not a measured CUDA peak. Autograd temporaries
and the PyTorch allocator can increase the peak. The original encoder still
dominates arithmetic; up to 64 groups of small GPU operations and validity
checks can add launch/synchronization overhead. No credible seconds/epoch
or percentage estimate is available without server measurement. Record
`torch.cuda.max_memory_allocated()` and elapsed time on the same RSTPReid
batch/environment before committing to 60 epochs. A100 40 GB feasibility is
supported by bounded incremental tensors, not yet established experimentally.

## Future B interface (not implemented)

`model(batch, return_evidence=True)` in A training returns `evidence_relations`:

| Field | Shape | Meaning |
| --- | --- | --- |
| `local_scores` | B,K,B | Rival-conditioned local cosine scores; uncomputed entries are zero placeholders |
| `positive_mask` | B,B | All true same-PID pairs |
| `negative_indices` | B,K | Batch rival row indices, -1 for missing rivals |
| `selected_negative_mask` | B,K | Valid hard-negative selection |
| `candidate_mask` | B,K,B | Relations selected for local computation |
| `evidence_valid_mask` | B,K,B | Computed nonzero pooled evidence |
| `valid_relation_mask` | B,K | Relations eligible for ranking supervision |
| `valid_query_mask` | B | Queries with at least one valid relation |
| `valid_query_count` | scalar | Effective denominator of the query mean |
| `per_query_loss` | B | Unweighted within-query mean, zero for invalid queries |
| `difference_queries` | B,K,D | Normalized rival-specific text differences |
| `pids` | B | Training-batch identities |

The standalone helper optionally returns sparse `pair_indices [E,3]` and
`attention_weights [E,L]` for checks. No tensors are cached between steps or
written to disk. K in the interface is `min(evidence_candidates, B-1)`;
missing rivals use -1 and false masks. Scores retain autograd; B must explicitly decide teacher
detachment and reliability rules when its design is approved. Validity,
PID supervision and CCD acceptance do not prove that an attention region
is semantically correct. B's planned entry is alongside these relations
after the original encoder forward, before the training loss is summed.
There is no B/student head, teacher update, KL objective or distillation here.

## Configuration, checkpoint and output protection

CLI: `--evidence_mode {none,off,a_only}`, `--evidence_candidates`,
`--evidence_weight`, `--evidence_temperature`, `--evidence_margin`.
Original CLI defaults and original optimizer groups are retained.
`--skip_final_test` explicitly skips post-training Best/Last Test evaluation;
it is false by default. With `--val_dataset val`, use this switch for smoke
and mechanism tuning so no final Test metric is consulted.
`configs.yaml` saves all options. A checkpoints also record `evidence_config`
in metadata; their model tensors use exactly the Baseline state schema.
Old weights can therefore be loaded strictly. State tensors alone cannot
identify whether A was used in training; retain the matching YAML/metadata.
For continuation use the same architecture/options and original `--resume`
interface; A-only resume rejects missing or different `evidence_config`
metadata. This pilot does not introduce weight-only warm-start semantics.

A-only supports one GPU and requires outputs inside repository `outputs/`.
Default `logs` is redirected to `outputs` only for A. A run creates
`outputs/<dataset>/<timestamp>_<name>_evidence_a_<loss>/` atomically before
opening its logs/configs. Existing final run directories are rejected. This
does not change off-mode output behavior. Nothing deletes or renames old
checkpoints/results. `/outputs/` is ignored by Git.

## Tests and server 1-epoch smoke run

Local data-free checks:

```bash
python -B -m unittest discover -s tests -p 'test_local_evidence.py' -v
```

Tensor tests cover multi-positive PID masks, hard-negative index placement,
singleton/no negatives, CCD rejection, zero positive/negative evidence,
zero text differences, valid-query mean and single weight application,
the explicit loss formula, changed-rival attention, minimum-temperature
softmax stability, finite-input zero-loss overflow, fp16 input casts, nonfinite inputs, strict
checkpoint round trip and off-mode parity against commit 83a538e. A tiny
instance of the actual CLIP checks that A ALONE produces finite nonzero
visual conv/projection and text embedding/projection gradients and changes
the original global retrieval encoding after an optimizer update. RDE
integration tests stub only TSE pooling and original TAL for CPU execution;
they verify no extra encoder forward and no inference evidence call. These
tests do not claim full fp16 CUDA training coverage. Without PyTorch, tensor
tests are explicitly skipped, not passed. Do not install dependencies locally
just to hide skips; run them in the existing server RDE environment.
An additional CUDA test applies the original `convert_weights` to a tiny
actual CLIP and checks finite nonzero A gradients through fp16 encoder weights.
It is separately skipped without CUDA and does not replace real-data testing.
A further CUDA parity test uses the actual original TAL and TSE with a tiny
CLIP and identical weights to compare off-mode training, CCD loss extraction
and inference against the Baseline. No pretrained download is required.
Static tests also verify that the optional final-Test guard preserves the
original Test block exactly.

After approved synchronization to the server, in the existing RDE conda env:

```bash
cd /root/siton-data-gaominData/ZYX/RDE/2024-CVPR-RDE
python -B -m unittest discover -s tests -p 'test_local_evidence.py' -v
# Require no skips and no failures here before the following smoke run.
read -r -p 'Existing ViT-B-16.pt absolute path: ' PRETRAIN
test -f "$PRETRAIN" || exit 1
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset_name RSTPReid \
  --root_dir /root/siton-data-gaominData/ZYX/datasets \
  --pretrain_choice "$PRETRAIN" \
  --output_dir /root/siton-data-gaominData/ZYX/RDE/outputs \
  --name "evidence_a_smoke_$(date +%Y%m%d_%H%M%S)" \
  --noisy_rate 0.0 --noisy_file ./noiseindex/RSTPReid_0.0.npy \
  --loss_names TAL+sr0.3_tau0.015_margin0.1_n0.0 \
  --tau 0.015 --margin 0.1 --select_ratio 0.3 \
  --batch_size 64 --num_epoch 1 --img_aug --txt_aug \
  --val_dataset val --skip_final_test \
  --evidence_mode a_only --evidence_candidates 4 \
  --evidence_weight 0.1 --evidence_temperature 0.07 --evidence_margin 0.1
```

The smoke command explicitly uses independent Val for training-time evaluation
and skips post-training Best/Last Test. Without `--skip_final_test`, that
original Test block still runs regardless of `val_dataset`, with no A
computations. The original selection default is still Test. This smoke is an engineering check,
not a comparable final retrieval experiment. Check finite A/total loss,
`evidence_valid_queries` in the logs (average valid query count per batch), memory/time,
and new best/last checkpoint metadata. Full real-data CUDA behavior and
retrieval benefit remain unverified until these server runs are executed.
