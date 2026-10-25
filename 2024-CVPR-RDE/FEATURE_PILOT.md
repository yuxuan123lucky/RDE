# TSE feature pilot

This branch keeps the original RDE BGE, CCD, TAL, data splits, and retrieval
metrics. It adds two optional, modality-specific enhancements after the
original 1024-dimensional TSE image/text pooling. The plain residual adapter
is a control inspired by [CLIP-Adapter](https://github.com/gaopengcuhk/CLIP-Adapter),
not a claimed new method.

| `--tse_enhancement` | TSE path | Extra parameters with bottleneck 256 |
| --- | --- | ---: |
| `none` | Original TSE, with no enhancer modules or calls | 0 |
| `adapter` | TSE + 0.1 × Adapter(TSE) | 1,055,232 |
| `delta_gate` | TSE + 0.1 × sigmoid(Gate([Projected BGE, TSE, absolute difference])) × Adapter(TSE) | 4,217,856 |

The adapter is LayerNorm → Linear(1024, bottleneck) → GELU →
Linear(bottleneck, 1024). The last Linear starts with small nonzero weights
and zero bias, so the initial residual is small while every layer can receive
gradients. Image and text use separate modules. The gate only combines BGE and
TSE from the **same** image or text; image and text towers remain independent.
The enhanced TSE output is float32 and remains 1024-dimensional. The new
modules run in float32 even though original CLIP and TSE Linear weights use
fp16. `--adapter_bottleneck` defaults to 256 and `--adapter_scale` to 0.1.
All trainable enhancer parameters are registered in the optimizer. They use
the original TSE layer learning rate (0.001), with the existing optimizer's
weight decay for weights and zero weight decay for biases. The `none` mode
does not add parameters or change any original parameter group.

The original CCD code is unchanged, but it sees enhanced TSE per-sample
losses when either enhancement is enabled. Its estimated clean/noisy split
can therefore change. TAL and retrieval metrics still use their original
functions, now receiving enhanced TSE features. BGE features are unchanged.

`train.py` saves these CLI options in each run's `configs.yaml`; `test.py`
loads the mode and dimensions from the matching YAML. Always evaluate a new
checkpoint with its own YAML. An old Baseline YAML has no enhancement option
and selects `none`. The original checkpoint loader keeps newly initialized
enhancer parameters when loading an old RDE checkpoint and restores them
from a matching new checkpoint. For weight-only initialization of an
experiment from an old Baseline `best.pth`, add
`--init_checkpoint /absolute/path/to/baseline/best.pth` to `train.py`. This
loads model weights before the optimizer is constructed and starts a new run
at epoch 1; it cannot be combined with `--resume`, which restores optimizer
state for a same-architecture continuation.

## Server commands

Set `PRETRAIN` to the actual local ViT-B/16 weight file. The command below
uses the original 0% noise training options from `run_rde.sh`, with output
placed in a new versioned directory. Set `DATASET` to `CUHK-PEDES`,
`ICFG-PEDES`, or `RSTPReid`; set `MODE` to `none`, `adapter`, or `delta_gate`.
Choose a fresh version label and run directory for every experiment.
`train.py` appends `<dataset>/<timestamp>_<name>_<loss>` to `RUN_ROOT`, so
different modes have separate checkpoint and log directories. The default
training seed remains 1 and the default `val_dataset` remains `test` for
the original Baseline protocol. Use the same existing `PRETRAIN` file for
both modes and omit `--init_checkpoint` in both runs to share the original
CLIP initialization source. If using a Baseline checkpoint instead, pass
the same `--init_checkpoint` path to both modes.

```bash
cd /root/siton-data-gaominData/ZYX/RDE/2024-CVPR-RDE
DATASET=CUHK-PEDES
MODE=adapter
VERSION=feature_pilot_v1
PRETRAIN=/absolute/path/to/ViT-B-16.pt
RUN_ROOT="/root/siton-data-gaominData/ZYX/RDE/outputs/${VERSION}/${MODE}"

CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset_name "$DATASET" \
  --root_dir /root/siton-data-gaominData/ZYX/datasets \
  --pretrain_choice "$PRETRAIN" \
  --output_dir "$RUN_ROOT" \
  --name "RDE_${MODE}" \
  --noisy_rate 0.0 \
  --noisy_file "./noiseindex/${DATASET}_0.0.npy" \
  --loss_names 'TAL+sr0.3_tau0.015_margin0.1_n0.0' \
  --tau 0.015 --margin 0.1 --select_ratio 0.3 \
  --batch_size 64 --num_epoch 60 --img_aug --txt_aug \
  --tse_enhancement "$MODE" \
  --adapter_bottleneck 256 --adapter_scale 0.1
```

For a trained run, set `CKPT` to its actual `best.pth` and use the YAML
saved beside it:

```bash
CKPT=/absolute/path/to/feature_pilot_run/best.pth
python test.py \
  --config_file "$(dirname "$CKPT")/configs.yaml" \
  --checkpoint "$CKPT" \
  --root_dir /root/siton-data-gaominData/ZYX/datasets
```

Run the CPU smoke tests on a machine with PyTorch before a full training run:

```bash
python -B -m unittest discover -s tests -p 'test_*.py' -v
```
