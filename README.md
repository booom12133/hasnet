# HASNet: reproducible dual-view X-ray multi-label recognition

Official implementation for:

> **Dual-View X-ray Multi-Label Recognition via Role-Separated Cross-View Fusion and Decision-Level Refinement**
>
> Huizhen Jia, Cailong Zhou, Peng Fu, and Tonghan Wang

HASNet separates shared dual-view representation, high-level cross-view fusion
(VSC + DCAF), and decision-level refinement (SRC-Head). This release contains
the complete proposed model, paper ablations, original reference baselines,
fixed DvXray splits, DDP/EMA training, calibration, robustness, profiling, and
three-seed aggregation code.

## Reproducibility scope

- Full HASNet and all VSC/DCAF/SRC ablations are implemented through one model
  class and one training engine.
- The paper protocol is encoded in
  [`configs/paper/full_convnext.yaml`](configs/paper/full_convnext.yaml).
- Evaluation uses no test-time augmentation.
- Checkpoints record the resolved config, split hashes, seed, environment,
  raw/EMA weights, optimizer, scaler, scheduler, epoch, and Git commit.
- Dataset images and third-party implementations are not redistributed.
  See [`docs/THIRD_PARTY_BASELINES.md`](docs/THIRD_PARTY_BASELINES.md).

## Installation

Python 3.10 is recommended.

```bash
conda env create -f environment.yml
conda activate hasnet
pip install -e .[dev]
```

Alternatively:

```bash
pip install -r requirements.txt
pip install -e .
```

Verify the installation before training:

```bash
python scripts/check_splits.py
pytest -q
python scripts/profile_model.py --config configs/paper/full_convnext.yaml
```

The full ConvNeXt-Tiny configuration reports approximately **30.97 M**
parameters and **12.123 G** FLOPs for one synchronized 256×256 OL/SD pair.
FLOPs are measured with THOP. The script separately reports the direct
`model.parameters()` total and THOP's module-hook parameter total.

## Dataset

Download DvXray from the
[official repository](https://github.com/Mbwslib/DvXray) and place the images
under:

```text
data/DvXray/
```

The expected layout is:

```text
HASNet/
├── data/
│   └── DvXray/
│       ├── N03892_OL.png
│       ├── N03892_SD.png
│       └── ...
└── splits/
    └── dvxray/
        ├── train.txt
        ├── val.txt
        └── test.txt
```

Only paths and multi-hot labels are consumed. Bounding-box fields remain in
the split files for traceability but are not used for image-level recognition.
The released split sizes are 11,200/3,200/1,600 and are checked for pair-level
overlap and SHA-256 integrity by `scripts/check_splits.py`.

## Exact training protocol

The headline configuration uses:

- two RTX 4090 GPUs with PyTorch DDP;
- ConvNeXt-Tiny with ImageNet-1K initialization;
- independently sampled OL/SD training augmentation;
- 256×256 inputs, 60 epochs, global batch size 64;
- AdamW, learning rate `7e-5`, weight decay `1e-3`;
- six-epoch linear warmup and epoch-wise cosine annealing;
- AMP, gradient clipping at 5.0, and EMA;
- ASL with `gamma_neg=4`, `gamma_pos=0`, and clipping `0.05`;
- auxiliary loss weight `alpha=0.7`;
- one 256-channel BiFPN layer;
- seeds 3407, 3408, and 3409 for the canonical release runs.

Train one seed:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train.py \
  --config configs/paper/full_convnext.yaml \
  --seed 3408 \
  --data-root . \
  --output-dir outputs
```

Print the canonical three-seed command matrix:

```bash
python scripts/reproduce.py --group full
```

Execute it:

```bash
python scripts/reproduce.py --group full --execute
```

Available groups are `full`, `backbones`, `core_ablation`,
`dcaf_ablation`, and `src_ablation`. Aggregate completed runs with:

```bash
python scripts/aggregate_runs.py \
  --input-dir outputs \
  --output results/generated/three_seed_summary.csv
```

## Evaluation

```bash
python scripts/evaluate.py \
  --config configs/paper/full_convnext.yaml \
  --checkpoint outputs/hasnet_full_convnext/seed_3408/best.pt \
  --split test \
  --weights ema
```

This writes both a JSON summary and an NPZ archive containing logits,
probabilities, targets, sample indices, and provenance metadata.

## Paper analysis

Calibration metrics are class-wise Brier, binary NLL, and 10-bin ECE over the
full `[0,1]` range, followed by macro averaging.

```bash
python scripts/calibrate.py \
  --protocol crossfit \
  --fit-predictions outputs/evaluation/main_only_val_clean.npz \
  --folds 5 --seed 3408
```

For a fully held-out calibration analysis, supply separate fitting and
evaluation archives with `--protocol heldout`.

Robustness uses exactly five validation conditions: clean, OL missing, SD
missing, OL blur, and SD blur. Missing sets the selected model-input tensor to
zero; blur uses a fixed 21×21 Gaussian kernel with sigma 5.

```bash
python scripts/robustness.py \
  --config configs/paper/full_convnext.yaml \
  --checkpoint outputs/hasnet_full_convnext/seed_3408/best.pt
```

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the
configuration-to-table map and required evidence for an archival release.

## License and citation

Original code in this repository is licensed under the MIT License. DvXray and
third-party methods retain their own terms. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and
[`CITATION.cff`](CITATION.cff).
