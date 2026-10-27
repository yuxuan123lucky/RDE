"""Read-only checkpoint analysis of RDE's two retrieval branches.

Only fresh reports below <repository>/outputs/<dataset>/ are written.
The validation weight grid is never evaluated on Test.
"""

import argparse
import csv
import hashlib
import importlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DATASETS = {"CUHK-PEDES": ("cuhkpedes", "CUHKPEDES"),
            "RSTPReid": ("rstpreid", "RSTPReid"),
            "ICFG-PEDES": ("icfgpedes", "ICFGPEDES")}
ORACLE_NOTE = ("Ground-truth-assisted choice of BGE or TSE Top-1 per query; "
               "analysis only, never a deployable/final performance result. "
               "This is not an upper bound on fusion: fusion can correct both branches.")


def analysis_weights(dataset_name, split):
    if dataset_name not in DATASETS or split not in ("val", "test"):
        raise ValueError("unsupported dataset or split")
    if dataset_name == "ICFG-PEDES" and split == "val":
        raise ValueError("ICFG-PEDES has no independent Val; use --split test for "
                         "fixed 0.5 fusion analysis only, without weight selection")
    return (0.25, 0.50, 0.75) if split == "val" else (0.50,)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", required=True, choices=DATASETS)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--root_dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config_file", type=Path,
                        help="default: configs.yaml beside the checkpoint")
    parser.add_argument("--output_dir", type=Path,
                        help="a NEW run directory under repository/outputs/<dataset>/")
    parser.add_argument("--pretrain_path", type=Path,
                        help="optional existing CLIP file, replacing the saved pretrain path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch_size", type=int,
                        help="default: saved test_batch_size (also for Val)")
    parser.add_argument("--num_workers", type=int, help="default: saved num_workers")
    parser.add_argument("--score_chunk_size", type=int, default=0,
                        help="0: full score matrices, as in RDE; positive: query blocks, full gallery")
    args = parser.parse_args(argv)
    try:
        analysis_weights(args.dataset_name, args.split)
    except ValueError as exc:
        parser.error(str(exc))
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch_size must be positive")
    if args.num_workers is not None and args.num_workers < 0:
        parser.error("--num_workers cannot be negative")
    if args.score_chunk_size < 0 or not 0 <= args.seed < 2 ** 32:
        parser.error("--score_chunk_size must be nonnegative; seed must be in [0, 2**32)")
    return args


def prepare_paths(args):
    """Validate paths without creating directories or changing input files."""
    args.root_dir = args.root_dir.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.config_file = (args.config_file or args.checkpoint.parent / "configs.yaml").expanduser().resolve()
    for label, path, is_dir in (("dataset root", args.root_dir, True),
                                ("checkpoint", args.checkpoint, False),
                                ("configuration", args.config_file, False)):
        if not (path.is_dir() if is_dir else path.is_file()):
            raise ValueError("{} does not exist or has the wrong type: {}".format(label, path))
    allowed = (REPO_ROOT / "outputs" / args.dataset_name).resolve()
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    args.output_dir = (args.output_dir or allowed / ("branch_analysis_" + args.split + "_" + tag)).expanduser().resolve()
    if allowed not in args.output_dir.parents:
        raise ValueError("output must be a NEW subdirectory of {}".format(allowed))
    if args.output_dir.exists():
        raise ValueError("refusing to overwrite existing output: {}".format(args.output_dir))
    for protected in (args.root_dir, args.checkpoint.parent):
        if args.output_dir == protected or protected in args.output_dir.parents:
            raise ValueError("output cannot be inside dataset/checkpoint directory: {}".format(protected))


def select_split(dataset, split):
    """Use the real processed split, never the training-mode Test fallback."""
    data = getattr(dataset, split)
    annos = getattr(dataset, split + "_annos")
    if not annos or not data["image_pids"] or not data["caption_pids"]:
        raise ValueError("{} split is empty; no fallback to Test is permitted".format(split))
    if any(anno["split"] != split for anno in annos):
        raise ValueError("annotation split labels do not match requested " + split)
    if len(data["image_pids"]) != len(data["img_paths"]) or len(data["caption_pids"]) != len(data["captions"]):
        raise ValueError("split PID and image/caption lengths differ")
    if split == "val":
        val_pids = {int(anno["id"]) for anno in annos}
        for other in ("train", "test"):
            if val_pids.intersection(int(anno["id"]) for anno in getattr(dataset, other + "_annos")):
                raise ValueError("Val identities overlap " + other + "; inspect dataset annotations")
    return data


def summarize_predictions(rows):
    """Correctness and identity disagreement are always defined by PID."""
    if not rows:
        raise ValueError("no queries to summarize")
    keys = ("bge_correct", "tse_correct", "fused_correct", "both_correct",
            "only_bge_correct", "only_tse_correct", "both_wrong",
            "top1_pid_disagreement", "top1_image_disagreement",
            "fusion_loses_any_correct", "fusion_corrects_both_wrong")
    counts = dict.fromkeys(keys, 0)
    for row in rows:
        b, t, f = (row[branch + "_top1_pid"] == row["pid"] for branch in ("bge", "tse", "fused"))
        flags = (b, t, f, b and t, b and not t, t and not b, not (b or t),
                 row["bge_top1_pid"] != row["tse_top1_pid"],
                 row["bge_top1_index"] != row["tse_top1_index"],
                 (b or t) and not f, f and not (b or t))
        for key, flag in zip(keys, flags):
            counts[key] += int(flag)
    n = len(rows)
    result = {key: {"count": count, "fraction": count / n, "percent": 100 * count / n}
              for key, count in counts.items()}
    union = counts["both_correct"] + counts["only_bge_correct"] + counts["only_tse_correct"]
    result["oracle_branch_selector"] = {
        "count": union, "R1_percent": 100 * union / n,
        "gain_over_best_single_branch_pp": 100 * (union - max(counts["bge_correct"], counts["tse_correct"])) / n,
        "uses_ground_truth": True, "analysis_only": True, "note": ORACLE_NOTE}
    return result


def check_checkpoint_state(expected, saved):
    """Reject incomplete/wrong-mode checkpoints instead of retaining random weights."""
    if not saved:
        raise ValueError("checkpoint contains no model parameters")
    if all(key.startswith("module.") for key in saved):
        saved = {key[len("module."):]: value for key, value in saved.items()}
    missing, extra = sorted(set(expected) - set(saved)), sorted(set(saved) - set(expected))
    wrong_shapes = [key for key in set(expected) & set(saved)
                    if expected[key].shape != saved[key].shape]
    if missing or extra or wrong_shapes:
        raise ValueError("checkpoint/config architecture mismatch: missing={}, unexpected={}, wrong_shapes={}".format(
            missing[:5], extra[:5], sorted(wrong_shapes)[:5]))
    return saved


def analyze_features(q_bge, g_bge, q_tse, g_tse, qids, gids, data, weights, chunk_size=0):
    import torch
    import torch.nn.functional as F
    from utils.metrics import rank

    n, ng = len(qids), len(gids)
    if n == 0 or ng < 10:
        raise ValueError("need nonempty queries and at least 10 gallery images for original RDE metrics")
    if not (len(q_bge) == len(q_tse) == len(data["captions"]) == len(data["caption_pids"]) == n and
            len(g_bge) == len(g_tse) == len(data["img_paths"]) == len(data["image_pids"]) == ng):
        raise ValueError("feature/annotation counts differ")
    if qids.tolist() != list(data["caption_pids"]) or gids.tolist() != list(data["image_pids"]):
        raise ValueError("feature and annotation PID order differs")
    if set(qids.tolist()) - set(gids.tolist()):
        raise ValueError("one or more query PIDs have no gallery positive")
    for features in (q_bge, g_bge, q_tse, g_tse):
        if not torch.isfinite(features).all().item() or (features.norm(dim=1) == 0).any().item():
            raise ValueError("nonfinite or zero-norm features")
    q_bge, g_bge, q_tse, g_tse = [F.normalize(x, p=2, dim=1) for x in (q_bge, g_bge, q_tse, g_tse)]
    names = ["BGE", "TSE"] + ["fusion_{:.2f}".format(w) for w in weights]
    totals = {name: torch.zeros(5, dtype=torch.float64) for name in names}
    rows = []
    for start in range(0, n, chunk_size or n):
        end = min(start + (chunk_size or n), n)
        bge = q_bge[start:end] @ g_bge.t()
        tse = q_tse[start:end] @ g_tse.t()
        block = [{"query_id": i, "pid": int(qids[i]), "text": data["captions"][i]}
                 for i in range(start, end)]
        for name in names:
            if name == "BGE":
                scores = bge
            elif name == "TSE":
                scores = tse
            else:
                w = float(name.split("_")[1])
                scores = (bge + tse) / 2 if w == 0.5 else w * bge + (1 - w) * tse
            cmc, ap, inp, indices = rank(scores, qids[start:end], gids, max_rank=10, get_mAP=True)
            totals[name] += torch.stack((cmc[0], cmc[4], cmc[9], ap, inp)).double() * (end - start)
            if name not in ("BGE", "TSE", "fusion_0.50"):
                continue
            prefix = {"BGE": "bge", "TSE": "tse", "fusion_0.50": "fused"}[name]
            for local, row in enumerate(block):
                index = int(indices[local, 0])
                row[prefix + "_top1_index"] = index
                row[prefix + "_top1_pid"] = int(gids[index])
                row[prefix + "_top1_correct"] = int(gids[index]) == row["pid"]
                row[prefix + "_top1_path"] = str(data["img_paths"][index])
                row[prefix + "_top1_score"] = float(scores[local, index])
        rows.extend(block)
    metrics = {name: dict(zip(("R1", "R5", "R10", "mAP", "mINP"), (total / n).tolist()))
               for name, total in totals.items()}
    outcomes = summarize_predictions(rows)
    for name, prefix in (("BGE", "bge"), ("TSE", "tse"), ("fusion_0.50", "fused")):
        if abs(metrics[name]["R1"] - outcomes[prefix + "_correct"]["percent"]) > 1e-4:
            raise AssertionError("RDE R1 and per-query PID correctness disagree for " + name)
    return metrics, outcomes, rows


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def set_seed(seed):
    """Same single-GPU state as train.py; do this before loaders/model creation."""
    import numpy as np
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def existing_pretrain(config, override):
    """Prevent the original CLIP builder from downloading/writing any weights."""
    from model.clip_model import _MODELS

    name = str(override.expanduser().resolve()) if override else config.pretrain_choice
    path = Path(name).expanduser()
    if name in _MODELS:
        url = _MODELS[name]
        path = Path.home() / ".cache" / "clip" / url.rsplit("/", 1)[-1]
        if path.is_file() and sha256_file(path) != url.split("/")[-2]:
            raise ValueError("cached official CLIP checksum mismatch; specify a verified --pretrain_path")
    if not path.is_file():
        raise ValueError("existing CLIP file not found: {}; specify --pretrain_path (no download is performed)".format(path))
    return str(path.resolve())


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    prepare_paths(args)
    sys.path.insert(0, str(PROJECT_ROOT))
    import torch
    from torch.utils.data import DataLoader
    from datasets.bases import ImageDataset, TextDataset
    from datasets.build import build_transforms
    from model import build_model
    from utils.iotools import load_train_configs
    from utils.metrics import Evaluator

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("original RDE contains fp16 modules; real checkpoint inference requires CUDA")
    torch.cuda.set_device(device)
    set_seed(args.seed)
    config = load_train_configs(str(args.config_file))
    if not config or config.dataset_name != args.dataset_name:
        raise ValueError("configuration dataset_name must match --dataset_name")
    if config.noisy_rate != 0.0:
        raise ValueError("this analysis is for 0.0 extra artificial noise checkpoints")
    saved_config = dict(config)
    config.pretrain_choice = existing_pretrain(config, args.pretrain_path)
    config.root_dir = str(args.root_dir)
    config.training = False
    batch_size = args.batch_size or config.test_batch_size
    workers = config.num_workers if args.num_workers is None else args.num_workers
    module, cls = DATASETS[args.dataset_name]
    dataset = getattr(importlib.import_module("datasets." + module), cls)(root=config.root_dir, verbose=False)
    data = select_split(dataset, args.split)
    transform = build_transforms(img_size=config.img_size, is_train=False)
    images = ImageDataset(data["image_pids"], data["img_paths"], transform)
    texts = TextDataset(data["caption_pids"], data["captions"], text_length=config.text_length)
    img_loader = DataLoader(images, batch_size=batch_size, shuffle=False, num_workers=workers)
    txt_loader = DataLoader(texts, batch_size=batch_size, shuffle=False, num_workers=workers)
    model = build_model(config, len(dataset.train_id_container))
    checkpoint = torch.load(str(args.checkpoint), map_location="cpu")
    state = check_checkpoint_state(model.state_dict(), checkpoint["model"])
    model.load_state_dict(state, strict=True)
    del state, checkpoint
    model = model.to(device).eval()
    evaluator = Evaluator(img_loader, txt_loader)
    with torch.no_grad():
        qb, gb, qids, gids = evaluator._compute_embedding(model)
        qt, gt, tqids, tgids = evaluator._compute_embedding_tse(model)
    if not torch.equal(qids, tqids) or not torch.equal(gids, tgids):
        raise AssertionError("BGE/TSE feature PID order differs")
    weights = analysis_weights(args.dataset_name, args.split)
    metrics, outcomes, rows = analyze_features(qb, gb, qt, gt, qids, gids, data, weights, args.score_chunk_size)
    weight_rows = [{"alpha_bge": w, "alpha_tse": 1 - w, **metrics["fusion_{:.2f}".format(w)]}
                   for w in weights]
    summary = {"dataset": args.dataset_name, "split": args.split,
               "num_queries": len(qids), "num_gallery": len(gids), "full_split": True,
               "metrics_percent": metrics, "query_outcomes": outcomes,
               "validation_weight_results": weight_rows if args.split == "val" else [],
               "weight_selection": "not performed; Val grid only" if args.split == "val" else "DISABLED ON TEST",
               "formula": "alpha_bge * cosine_BGE + (1-alpha_bge) * cosine_TSE",
               "correctness": "all gallery images with query PID are positives",
               "inference_state": "model.eval() + torch.no_grad()"}
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                             capture_output=True, text=True, check=True).stdout.strip(),
                "git_status": subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal"],
                                             cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.splitlines(),
                "analysis_script_sha256": sha256_file(Path(__file__)),
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "saved_config": saved_config, "effective_pretrain_path": config.pretrain_choice,
                "checkpoint_sha256": sha256_file(args.checkpoint), "config_sha256": sha256_file(args.config_file),
                "tse_enhancement": getattr(config, "tse_enhancement", "none"),
                "saved_val_dataset": getattr(config, "val_dataset", None),
                "seed": args.seed, "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "batch_size": batch_size, "num_workers": workers,
                "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                "feature_dimensions": {"BGE": qb.shape[1], "TSE": qt.shape[1]},
                "metric_note": "Original utils.metrics.rank; query blocks may change near-tie floating point scores/reduction. "
                               "Use --score_chunk_size 0 and matching batch size for reference comparison.",
                "selection_note": "saved_val_dataset records configuration, not verified historical model selection. "
                                  "Val analysis does not undo previous Test-based checkpoint selection."}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for filename, value in (("branch_summary.json", summary), ("run_metadata.json", metadata)):
        with (args.output_dir / filename).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    write_csv(args.output_dir / "query_comparison.csv", rows)
    write_csv(args.output_dir / "fusion_metrics.csv", weight_rows)
    print(json.dumps({"output_dir": str(args.output_dir), **summary}, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    run(parse_args())
