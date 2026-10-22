"""Inspect text-to-image retrieval errors without changing the RDE baseline."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import textwrap
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BRANCHES = ("BGE", "TSE", "BGE+TSE")
DATASET_CLASSES = {
    "CUHK-PEDES": ("cuhkpedes", "CUHKPEDES", "file_path"),
    "ICFG-PEDES": ("icfgpedes", "ICFGPEDES", "file_path"),
    "RSTPReid": ("rstpreid", "RSTPReid", "img_path"),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", required=True, choices=DATASET_CLASSES)
    parser.add_argument("--root_dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--config_file", type=Path,
                        help="defaults to configs.yaml beside the checkpoint")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--max_queries", type=int,
                        help="analyze the first N test captions; gallery stays complete")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score_chunk_size", type=int, default=64)
    parser.add_argument("--num_visualizations", type=int, default=24)
    args = parser.parse_args(argv)

    args.root_dir = args.root_dir.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.config_file = (args.config_file or args.checkpoint.parent / "configs.yaml").expanduser().resolve()
    for label, path, kind in (("dataset root", args.root_dir, "dir"),
                              ("checkpoint", args.checkpoint, "file"),
                              ("configuration", args.config_file, "file")):
        if not (path.is_dir() if kind == "dir" else path.is_file()):
            parser.error(f"{label} does not exist or is not a {kind}: {path}")
    if args.output_dir.exists():
        parser.error(f"output directory already exists; refusing to overwrite: {args.output_dir}")
    if args.output_dir == args.checkpoint.parent or args.checkpoint.parent in args.output_dir.parents:
        parser.error("output directory must be outside the checkpoint run directory")
    if args.output_dir == args.root_dir or args.root_dir in args.output_dir.parents:
        parser.error("output directory must be outside the dataset root")
    if args.topk < 1 or args.score_chunk_size < 1:
        parser.error("--topk and --score_chunk_size must be positive")
    if args.max_queries is not None and args.max_queries < 1:
        parser.error("--max_queries must be positive")
    if args.num_visualizations < 0:
        parser.error("--num_visualizations cannot be negative")
    return args


def build_query_metadata(test_annos, img_dataset, txt_dataset, img_dir, image_key):
    """Recover the original caption/image link and verify the loader's order."""
    queries = []
    image_index = 0
    for anno in test_annos:
        pid = int(anno["id"])
        image_path = os.path.normpath(os.path.join(img_dir, anno[image_key]))
        if image_index >= len(img_dataset.img_paths):
            raise ValueError("annotation has more test images than the image loader")
        if (pid != int(img_dataset.image_pids[image_index]) or
                image_path != os.path.normpath(img_dataset.img_paths[image_index])):
            raise ValueError(f"test image order/PID differs at index {image_index}")
        for caption in anno["captions"]:
            query_index = len(queries)
            if query_index >= len(txt_dataset.captions):
                raise ValueError("annotation has more test captions than the text loader")
            if (caption != txt_dataset.captions[query_index] or
                    pid != int(txt_dataset.caption_pids[query_index])):
                raise ValueError(f"test caption order/PID differs at index {query_index}")
            queries.append({"query_index": query_index, "text": caption,
                            "pid": pid, "source_image_index": image_index,
                            "source_image_path": image_path})
        image_index += 1
    if image_index != len(img_dataset.img_paths) or len(queries) != len(txt_dataset.captions):
        raise ValueError("annotation and test loader lengths differ")
    return queries


def classify_query(bge_correct, tse_correct, fused_correct):
    if fused_correct and not (bge_correct and tse_correct):
        return "fusion_corrects_both" if not (bge_correct or tse_correct) else "fusion_preserves_single_success"
    if not fused_correct and (bge_correct or tse_correct):
        return "fusion_introduces_error"
    if bge_correct and tse_correct:
        return "all_correct"
    return "all_wrong"


def score_query(scores, sorted_indices, query_pid, gallery_pids, gallery_paths, topk):
    """Describe one query using the exact full-gallery order from metrics.rank."""
    ordered_pids = gallery_pids[sorted_indices]
    positives = (ordered_pids == query_pid).nonzero(as_tuple=True)[0]
    if positives.numel() == 0:
        raise ValueError(f"query PID {query_pid} has no positive gallery image")
    first_positive_rank = int(positives[0]) + 1
    first_positive_index = int(sorted_indices[positives[0]])
    top1_index = int(sorted_indices[0])
    top1_score = float(scores[top1_index])
    best_positive_score = float(scores[first_positive_index])
    ranked = []
    for position, index_tensor in enumerate(sorted_indices[:topk], 1):
        index = int(index_tensor)
        pid = int(gallery_pids[index])
        ranked.append({"rank": position, "gallery_index": index,
                       "image_path": gallery_paths[index], "pid": pid,
                       "score": float(scores[index]), "is_positive": pid == query_pid})
    return {
        "top1_correct": int(gallery_pids[top1_index]) == query_pid,
        "top1_gallery_index": top1_index,
        "top1_path": gallery_paths[top1_index],
        "top1_pid": int(gallery_pids[top1_index]),
        "top1_score": top1_score,
        "best_positive_rank": first_positive_rank,
        "best_positive_gallery_index": first_positive_index,
        "best_positive_path": gallery_paths[first_positive_index],
        "best_positive_score": best_positive_score,
        "top1_minus_best_positive_score": top1_score - best_positive_score,
        "topk": ranked,
    }


def compute_diagnostics(q_bge, g_bge, q_tse, g_tse, qids, gids,
                        gallery_paths, queries, topk, chunk_size):
    """Use RDE's rank() on full-gallery scores, one query block at a time."""
    import torch
    import torch.nn.functional as F
    from utils.metrics import rank

    if not (len(qids) == len(queries) == len(q_bge) == len(q_tse)):
        raise ValueError("query features, PIDs, and annotation counts differ")
    if not (len(gids) == len(gallery_paths) == len(g_bge) == len(g_tse)):
        raise ValueError("gallery features, PIDs, and paths differ")
    if len(gids) < max(10, topk):
        raise ValueError("gallery must contain at least max(10, --topk) images")
    gallery_pid_set = set(int(pid) for pid in gids)
    if any(int(pid) not in gallery_pid_set for pid in qids):
        raise ValueError("one or more test queries have no positive gallery image")

    q_bge = F.normalize(q_bge, p=2, dim=1)
    g_bge = F.normalize(g_bge, p=2, dim=1)
    q_tse = F.normalize(q_tse, p=2, dim=1)
    g_tse = F.normalize(g_tse, p=2, dim=1)
    g_bge_t, g_tse_t = g_bge.t(), g_tse.t()
    metric_sums = {branch: torch.zeros(5, dtype=torch.float64) for branch in BRANCHES}
    query_rows, comparison_rows = [], []
    gallery_paths = [str(path) for path in gallery_paths]

    for start in range(0, len(qids), chunk_size):
        end = min(start + chunk_size, len(qids))
        sims_bge = q_bge[start:end] @ g_bge_t
        sims_tse = q_tse[start:end] @ g_tse_t
        similarities = {"BGE": sims_bge, "TSE": sims_tse,
                        "BGE+TSE": (sims_bge + sims_tse) / 2}
        branch_indices = {}
        for branch, scores in similarities.items():
            cmc, m_ap, m_inp, indices = rank(scores, qids[start:end], gids,
                                             max_rank=10, get_mAP=True)
            metric_sums[branch] += torch.stack(
                (cmc[0], cmc[4], cmc[9], m_ap, m_inp)
            ).to(torch.float64) * (end - start)
            branch_indices[branch] = indices

        for local_index in range(end - start):
            query_index = start + local_index
            meta = queries[query_index]
            if meta["query_index"] != query_index or meta["pid"] != int(qids[query_index]):
                raise ValueError(f"query order/PID differs at index {query_index}")
            row = dict(meta)
            row["num_positive_gallery"] = int((gids == qids[query_index]).sum())
            details = {}
            for branch in BRANCHES:
                details[branch] = score_query(
                    similarities[branch][local_index], branch_indices[branch][local_index],
                    int(qids[query_index]), gids, gallery_paths, topk)
                prefix = {"BGE": "bge", "TSE": "tse", "BGE+TSE": "fused"}[branch]
                for key, value in details[branch].items():
                    row[f"{prefix}_{key}"] = json.dumps(value, ensure_ascii=False) if key == "topk" else value
            bge_ok, tse_ok, fused_ok = (details[b]["top1_correct"] for b in BRANCHES)
            category = classify_query(bge_ok, tse_ok, fused_ok)
            row["category"] = category
            query_rows.append(row)
            comparison_rows.append({
                "query_index": query_index, "pid": meta["pid"], "text": meta["text"],
                "bge_correct": bge_ok, "tse_correct": tse_ok, "fused_correct": fused_ok,
                "bge_correct_tse_wrong": bge_ok and not tse_ok,
                "tse_correct_bge_wrong": tse_ok and not bge_ok,
                "fusion_corrects_both": fused_ok and not (bge_ok or tse_ok),
                "fusion_corrects_at_least_one": fused_ok and not (bge_ok and tse_ok),
                "fusion_introduces_error": not fused_ok and (bge_ok or tse_ok),
                "category": category,
                "bge_best_positive_rank": details["BGE"]["best_positive_rank"],
                "tse_best_positive_rank": details["TSE"]["best_positive_rank"],
                "fused_best_positive_rank": details["BGE+TSE"]["best_positive_rank"],
            })

    n = len(qids)
    metrics = {branch: dict(zip(("R1", "R5", "R10", "mAP", "mINP"),
                                (metric_sums[branch] / n).tolist())) for branch in BRANCHES}
    category_counts = Counter(row["category"] for row in comparison_rows)
    summary = {
        "dataset": None,
        "split": "test",
        "num_queries": n,
        "num_gallery_images": len(gids),
        "partial_queries": False,
        "metrics_percent": metrics,
        "top1_errors": {branch: sum(not row[{"BGE": "bge", "TSE": "tse",
                                                "BGE+TSE": "fused"}[branch] + "_top1_correct"]
                                    for row in query_rows) for branch in BRANCHES},
        "complementarity": {
            "bge_correct_tse_wrong": sum(row["bge_correct_tse_wrong"] for row in comparison_rows),
            "tse_correct_bge_wrong": sum(row["tse_correct_bge_wrong"] for row in comparison_rows),
            "fusion_corrects_both": sum(row["fusion_corrects_both"] for row in comparison_rows),
            "fusion_corrects_at_least_one": sum(row["fusion_corrects_at_least_one"] for row in comparison_rows),
            "fusion_introduces_error": sum(row["fusion_introduces_error"] for row in comparison_rows),
            "categories": dict(category_counts),
        },
        "metric_note": "Original utils.metrics.rank; PID equality defines every positive image."
    }
    return summary, query_rows, comparison_rows


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"no rows to write: {path.name}")
    with path.open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render_examples(output_dir, rows, limit):
    """Create review candidates; labels describe outcomes, not error causes."""
    if limit == 0:
        return 0
    from PIL import Image, ImageDraw

    candidates = [row for row in rows if not row["fused_top1_correct"]]
    candidates.sort(key=lambda row: (
        0 if row["category"] == "fusion_introduces_error" else
        1 if row["bge_top1_correct"] != row["tse_top1_correct"] else 2,
        row["fused_best_positive_rank"], row["query_index"]))
    # Prefer different identities for the first review pass.
    selected, seen_pids = [], set()
    for row in candidates:
        if row["pid"] not in seen_pids:
            selected.append(row)
            seen_pids.add(row["pid"])
        if len(selected) == limit:
            break
    if len(selected) < limit:
        selected_indices = {row["query_index"] for row in selected}
        selected.extend(row for row in candidates
                        if row["query_index"] not in selected_indices)
        selected = selected[:limit]
    examples_dir = output_dir / "examples"
    if not candidates:
        return 0
    examples_dir.mkdir(exist_ok=False)
    tile_w, tile_h = 215, 315
    count = 0
    for row in selected:
        ranked = json.loads(row["fused_topk"])
        panels = [
            ("Fused wrong Top-1", row["fused_top1_path"], row["fused_top1_pid"]),
            (f"Best positive, rank {row['fused_best_positive_rank']}",
             row["fused_best_positive_path"], row["pid"]),
        ]
        panels += [(f"Top-{item['rank']}  score {item['score']:.4f}",
                    item["image_path"], item["pid"]) for item in ranked]
        cols = 4
        lines = textwrap.wrap(f"Query {row['query_index']}  PID {row['pid']}: {row['text']}",
                              width=105, break_long_words=False)
        header_h = max(80, 22 * len(lines) + 20)
        canvas = Image.new("RGB", (cols * tile_w, header_h +
                                   ((len(panels) + cols - 1) // cols) * tile_h), "white")
        draw = ImageDraw.Draw(canvas)
        for line_index, line in enumerate(lines):
            draw.text((12, 10 + 21 * line_index), line, fill="black")
        for panel_index, (label, path, pid) in enumerate(panels):
            x = (panel_index % cols) * tile_w
            y = header_h + (panel_index // cols) * tile_h
            with Image.open(path) as image:
                thumbnail = image.convert("RGB")
                thumbnail.thumbnail((tile_w - 12, tile_h - 65))
                canvas.paste(thumbnail, (x + (tile_w - thumbnail.width) // 2, y + 5))
            draw.text((x + 5, y + tile_h - 55), label, fill="black")
            draw.text((x + 5, y + tile_h - 34), f"PID {pid}  {'positive' if pid == row['pid'] else 'negative'}",
                      fill="green" if pid == row["pid"] else "red")
        canvas.save(examples_dir / f"query_{row['query_index']:06d}.png")
        count += 1
    return count


def run(args):
    import torch
    from torch.utils.data import DataLoader, Subset

    sys.path.insert(0, str(PROJECT_ROOT))
    from datasets import build_dataloader
    from model import build_model
    from utils.checkpoint import Checkpointer
    from utils.iotools import load_train_configs
    from utils.metrics import Evaluator

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("original RDE inference uses fp16 modules; choose a CUDA device")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    config = load_train_configs(str(args.config_file))
    if not config:
        raise ValueError(f"empty configuration: {args.config_file}")
    if config.dataset_name != args.dataset_name:
        raise ValueError(f"checkpoint configuration is for {config.dataset_name}, "
                         f"not {args.dataset_name}")
    config.root_dir = str(args.root_dir)
    config.training = False
    img_loader, txt_loader, num_classes = build_dataloader(config)
    full_query_count = len(txt_loader.dataset)
    query_count = min(args.max_queries or full_query_count, full_query_count)
    if query_count == 0:
        raise ValueError("test split contains no text queries")
    if query_count < full_query_count:
        txt_loader = DataLoader(Subset(txt_loader.dataset, range(query_count)),
                                batch_size=txt_loader.batch_size, shuffle=False,
                                num_workers=txt_loader.num_workers)

    module_name, class_name, image_key = DATASET_CLASSES[args.dataset_name]
    dataset_module = __import__(f"datasets.{module_name}", fromlist=[class_name])
    dataset = getattr(dataset_module, class_name)(root=str(args.root_dir), verbose=False)
    queries = build_query_metadata(dataset.test_annos, img_loader.dataset,
                                   txt_loader.dataset.dataset if isinstance(txt_loader.dataset, Subset)
                                   else txt_loader.dataset,
                                   dataset.img_dir, image_key)[:query_count]

    model = build_model(config, num_classes)
    Checkpointer(model).load(f=str(args.checkpoint))
    model = model.to(device).eval()
    evaluator = Evaluator(img_loader, txt_loader)
    with torch.no_grad():
        q_bge, g_bge, qids, gids = evaluator._compute_embedding(model)
        q_tse, g_tse, tse_qids, tse_gids = evaluator._compute_embedding_tse(model)
    if not torch.equal(qids, tse_qids) or not torch.equal(gids, tse_gids):
        raise ValueError("BGE and TSE loader PID orders differ")
    summary, rows, comparisons = compute_diagnostics(
        q_bge, g_bge, q_tse, g_tse, qids, gids,
        img_loader.dataset.img_paths, queries, args.topk, args.score_chunk_size)
    summary["dataset"] = args.dataset_name
    summary["partial_queries"] = query_count < full_query_count
    summary["full_test_query_count"] = full_query_count
    summary["saved_val_dataset"] = getattr(config, "val_dataset", None)
    summary["test_py_metric_comparison"] = "pending external test.py output comparison"
    if summary["partial_queries"]:
        summary["metric_note"] += " Partial-query metrics are not comparable with full test.py metrics."

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "dataset": args.dataset_name,
        "split": "test",
        "saved_val_dataset": getattr(config, "val_dataset", None),
        "checkpoint": str(args.checkpoint),
        "checkpoint_size_bytes": args.checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "config_file": str(args.config_file),
        "config_sha256": sha256_file(args.config_file),
        "root_dir": str(args.root_dir),
        "output_dir": str(args.output_dir),
        "device": str(device),
        "topk": args.topk,
        "max_queries": args.max_queries,
        "analyzed_queries": query_count,
        "full_test_queries": full_query_count,
        "score_chunk_size": args.score_chunk_size,
        "num_visualizations_requested": args.num_visualizations,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "query_rankings.csv", rows)
    write_csv(args.output_dir / "branch_comparison.csv", comparisons)
    summary["visualized_fused_errors"] = render_examples(args.output_dir, rows,
                                                          args.num_visualizations)
    write_json(args.output_dir / "retrieval_summary.json", summary)
    write_json(args.output_dir / "run_metadata.json", metadata)
    print(json.dumps({"output_dir": str(args.output_dir),
                      "metrics_percent": summary["metrics_percent"]}, indent=2))


if __name__ == "__main__":
    run(parse_args())
