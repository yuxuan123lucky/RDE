# RDE branch complementarity analysis

`analyze_branch_complementarity.py` only loads existing checkpoints. It does
not create training checkpoints, optimize parameters, inject noise, change
the model, or append to training/test logs. It supports Baseline (`none`),
Adapter, and Delta Gate through the checkpoint's **matching** `configs.yaml`.

## Protocol

- CUHK-PEDES and RSTPReid default to their actual `dataset.val` annotations.
  Empty/mislabeled Val or Val identity overlap with Train/Test is an error.
- Val reports BGE, TSE, and `alpha * BGE + (1-alpha) * TSE` for BGE weights
  0.25, 0.50, 0.75. It does not automatically choose a weight or save a model.
- `--split test` reports only BGE, TSE, and the original `(BGE + TSE) / 2`.
  There is no Test weight grid, including for ICFG-PEDES. ICFG Val is rejected.
- Every query and every gallery image of the selected split is used. Correct
  matches are all images with the query's PID, not just its source image.
- Identity disagreement and gallery-row disagreement are separate statistics:
  different Top-1 images with the same PID can both be correct.
- The oracle branch-selector R@1 counts queries where **either** branch's
  Top-1 is correct. It uses ground truth, is analysis only, and must never be
  reported as final model performance. Fusion can exceed this number by
  correcting queries for which both individual branches have wrong Top-1.

The script uses the original `build_transforms(is_train=False)`,
`ImageDataset`, `TextDataset`, `Evaluator` feature extractors, `F.normalize`,
and `utils.metrics.rank`. No training dataset/dataloader is built. Model
state is strictly checked against checkpoint keys/shapes after removing an
optional leading `module.`; wrong-mode/incomplete weights fail clearly.
Seed defaults to 1, with the original cuDNN `deterministic=True` and
`benchmark=True`. CUDA is required for real RDE fp16 inference; `--help` and
dependency-free tests work without PyTorch. Only existing CLIP files are
accepted, with no downloads. Use `--pretrain_path` if the saved path differs
on the server. A named official CLIP weight may use an existing verified
`~/.cache/clip` file.

## Server examples

From `/root/siton-data-gaominData/ZYX/RDE/2024-CVPR-RDE`, in the existing RDE
environment, set the following to **actual existing** files. No new weights
are created. The default output path is
`/root/siton-data-gaominData/ZYX/RDE/outputs/<dataset>/branch_analysis_<split>_<UTC timestamp>/`.
Each run is new; existing output directories are rejected. An optional
`--output_dir` must also be a new directory below `outputs/<dataset>/`, outside
the input checkpoint directory and dataset root.

```bash
CKPT=/absolute/path/to/CUHK-PEDES/run/best.pth
CLIP=/absolute/path/to/existing/ViT-B-16.pt
python tools/analyze_branch_complementarity.py \
  --dataset_name CUHK-PEDES --split val \
  --root_dir /root/siton-data-gaominData/ZYX/datasets \
  --checkpoint "$CKPT" --config_file "$(dirname "$CKPT")/configs.yaml" \
  --pretrain_path "$CLIP" --device cuda:0

CKPT=/absolute/path/to/RSTPReid/run/best.pth
python tools/analyze_branch_complementarity.py \
  --dataset_name RSTPReid --split val \
  --root_dir /root/siton-data-gaominData/ZYX/datasets \
  --checkpoint "$CKPT" --config_file "$(dirname "$CKPT")/configs.yaml" \
  --pretrain_path "$CLIP" --device cuda:0
```

For a fixed-protocol Test check, explicitly use `--split test`. ICFG-PEDES
also requires `--split test`; do not use it to tune weights. The default
inference batch size is the YAML's `test_batch_size`, not training batch size.
Keep it unchanged when comparing against `test.py`; batch size, seed,
environment, and near ties can affect fp16 results. `--score_chunk_size 0`
(default) uses full score matrices, matching original evaluation. For a large
gallery (especially ICFG), `--score_chunk_size 64` bounds ranking memory
without shrinking the gallery, but floating point kernel/reduction differences
can affect near ties. Compare reference metrics before drawing conclusions.

## Output and interpretation

- `branch_summary.json`: R@1/5/10, mAP, mINP in percent; query counts and
  proportions for both correct, only BGE, only TSE, both wrong, disagreements,
  fusion corrections/losses, and the clearly labeled oracle diagnostic.
- `fusion_metrics.csv`: fixed weights and metrics; three Val rows, one Test row.
- `query_comparison.csv`: same-query text/PID, Top-1 PID/path/score/correctness
  for BGE, TSE, and 0.5 fusion. Paths are references, never copied images.
- `run_metadata.json`: input hashes, Git commit, saved configuration, effective
  CLIP path, split, device, seed, cuDNN flags, actual batch size and dimensions.

Strict state loading cannot detect a wrong YAML residual scale because that
Python float is not in the checkpoint. Always pair checkpoint and YAML from
the same run. Metadata records `saved_val_dataset`; it does not prove historical
checkpoint selection. Val analysis of a previously Test-selected checkpoint
does not undo that protocol risk. A nonzero complementary-query count shows
different branch successes, not an attribute-level cause or proof a new gate
will improve retrieval. Human review and independent training are still needed.

## Tests

```bash
python -B -m unittest discover -s tools -p 'test_branch_complementarity.py' -v
```

Protocol/PID/output-safety tests use only the standard library. The metric
test compares all weight formulas and multiple-PID-positive ranking against
original `rank`, including query blocking. It is explicitly **skipped**, not
passed, without PyTorch. No real dataset evaluation is part of these tests.
