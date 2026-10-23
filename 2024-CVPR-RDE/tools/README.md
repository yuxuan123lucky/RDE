# RDE retrieval diagnostics

`retrieval_diagnostics.py` loads an existing RDE `best.pth` and its sibling
`configs.yaml`. It evaluates the selected **val** or **test** split with the existing
`Evaluator` encoders and `utils.metrics.rank`. A gallery image is positive when
its PID equals the text query PID; the original caption/image link is exported
separately. No training takes place.

The script refuses an existing output directory and never writes over a
checkpoint or into the dataset directory. Use a new directory under the server's
`outputs/<dataset>/diagnostics/` for every run. Set `CKPT` to the actual `best.pth` path;
verify that the same directory contains its matching `configs.yaml`.

```bash
cd /root/siton-data-gaominData/ZYX/RDE/2024-CVPR-RDE  # adjust to the server checkout
RUN=$(date +%Y%m%d_%H%M%S)
CKPT=/absolute/path/to/CUHK-PEDES/baseline-run/best.pth
python tools/retrieval_diagnostics.py \
  --dataset_name CUHK-PEDES \
  --split val \
  --root_dir /root/siton-data-gaominData/ZYX/datasets \
  --checkpoint "$CKPT" \
  --output_dir /root/siton-data-gaominData/ZYX/RDE/outputs/CUHK-PEDES/diagnostics/val_${RUN} \
  --device cuda:0
```

```bash
RUN=$(date +%Y%m%d_%H%M%S)
CKPT=/absolute/path/to/ICFG-PEDES/baseline-run/best.pth
python tools/retrieval_diagnostics.py \
  --dataset_name ICFG-PEDES \
  --split test \
  --root_dir /root/siton-data-gaominData/ZYX/datasets \
  --checkpoint "$CKPT" \
  --output_dir /root/siton-data-gaominData/ZYX/RDE/outputs/ICFG-PEDES/diagnostics/test_${RUN} \
  --device cuda:0
```

```bash
RUN=$(date +%Y%m%d_%H%M%S)
CKPT=/absolute/path/to/RSTPReid/baseline-run/best.pth
python tools/retrieval_diagnostics.py \
  --dataset_name RSTPReid \
  --split val \
  --root_dir /root/siton-data-gaominData/ZYX/datasets \
  --checkpoint "$CKPT" \
  --output_dir /root/siton-data-gaominData/ZYX/RDE/outputs/RSTPReid/diagnostics/val_${RUN} \
  --device cuda:0
```

`--split` defaults to `test` for compatibility. Select `val` explicitly for
CUHK-PEDES and RSTPReid validation diagnostics. Confirm that the chosen split
has annotations in the server dataset; the script fails rather than using a
different split. `--topk` defaults to 10. `--max_queries N` analyzes only the first N selected-split text
queries while retaining the full gallery; its metrics are **not** comparable
with full `test.py` results. `--score_chunk_size` defaults to 64 and controls
CPU similarity/ranking memory. `--num_visualizations` defaults to 24 fused
Top-1 errors; use 0 to omit PNGs. The output directory contains
`retrieval_summary.json`, `query_rankings.csv`, `branch_comparison.csv`,
`run_metadata.json`, and, when errors are available, `examples/*.png`.
`query_index` is the text query ID in the selected split. Each query row contains
all three branch Top-K rankings, scores, PIDs, best-positive ranks and the
original caption/image link. Summary counts separate both-correct, BGE-only,
TSE-only and both-wrong cases, plus branch-specific fusion losses.
Use a CUDA device for the original RDE fp16 inference path.

For a full **test** run, compare all three branches' R@1/5/10, mAP, and mINP in
`retrieval_summary.json` with `test.py` for the **same checkpoint, YAML and
test split**. `test.py` does not evaluate Val. The diagnostic uses the same
normalization, average fusion, and `rank()` function; stop and investigate any
discrepancy. The available
CUHK-PEDES and ICFG-PEDES baseline best R@1 values are 75.50% and 67.43%,
respectively; these are reference observations, not locally verified results.
The images are review candidates, not automatic explanations of error causes.
The saved training option `val_dataset` is included in both JSON reports;
the baseline default is `test`, which means `best.pth` may have been selected
using the test split under the original protocol.

Local, data-free checks:

```bash
python -B -m unittest discover -s tools -p 'test_*.py'
```
