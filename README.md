# HASNet: dual-view X-ray multi-label recognition

Official implementation for:

> **Dual-View X-ray Multi-Label Recognition via Role-Separated Cross-View Fusion and Decision-Level Refinement**
>
> Huizhen Jia, Cailong Zhou, Peng Fu, and Tonghan Wang

This repository provides the official implementation of HASNet for dual-view
X-ray image-level multi-label recognition. It includes the full proposed model,
training and evaluation code, the DvXray split files used by the main
experiment, and the converted LDXray train/validation/test manifests used for
external evaluation.

HASNet uses a shared dual-stream backbone followed by VSC, per-view coordinate
attention, CVFUSE, LKA, and the MAIN, AUX/GATE, and BiFPN components of
SRC-Head. Additional ablation and analysis experiments reported in the paper
were conducted using the same research codebase but are not supported as
separate reproduction protocols in this public release.

Dataset images, checkpoints, logs, and third-party model implementations are
not redistributed.

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

Verify the installation without downloading pretrained weights:

```bash
python scripts/check_splits.py
python -m pytest -q
python scripts/profile_model.py --config configs/paper/full_convnext.yaml
```

For one synchronized 256×256 OL/SD pair, Full HASNet has approximately:

- Params: **30.973 M**
- MACs: **12.123 G**
- FLOPs: **24.246 G**

THOP's operation count is reported as MACs. FLOPs use the convention
`1 MAC = 2 FLOPs`. Parameter totals are counted directly from all registered
model parameters.

## Dataset

### DvXray

Download DvXray from the
[official repository](https://github.com/Mbwslib/DvXray) and place the images
so the relative paths in `splits/dvxray/*.txt` resolve from the selected data
root. A typical layout is:

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

The fixed DvXray split used in the paper contains 11,200/3,200/1,600
synchronized pairs for train/validation/test. Bounding-box fields remain in
the manifests for traceability but are not used for image-level recognition.

### External evaluation / LDXray

[LDXray](https://github.com/rstao-bjtu/LDXray) was originally released for
paired dual-view object detection. For image-level multi-label recognition,
bounding-box coordinates are discarded as supervision and each synchronized
OL/SD pair receives a 12-class multi-hot vector containing every annotated
prohibited-item category present in the pair.

One synchronized View A/View B pair is the indivisible split unit. The official
36,849-pair test split is preserved unchanged. Only the 110,148-pair official
training split is divided by pair-level iterative multi-label stratification at
a 9:1 ratio with split seed 42; `ceil(110,148 × 10%)` gives 11,015 validation
pairs. The released manifests therefore contain 99,133 training pairs, 11,015
validation pairs, and 36,849 test pairs. The manifest and model-output class
order is:

`MP, OL, PC1, LA, GL, PC2, TA, BL, CO, NL, UM, CG`

The corresponding indices are `0: MP`, `1: OL`, `2: PC1`, `3: LA`, `4: GL`,
`5: PC2`, `6: TA`, `7: BL`, `8: CO`, `9: NL`, `10: UM`, and `11: CG`.
For presentation and category analysis, Table 3 reports classes in the order
`MP, OL, PC1, PC2, LA, GL, TA, BL, NL, CO, UM, CG`, using the permutation
`[0, 1, 2, 5, 3, 4, 6, 7, 9, 8, 10, 11]` from model-output order. This is a
reporting/presentation-only reorder; it does not change manifests, labels,
predictions, per-class AP values, or macro mAP.

All methods reported in the paper's LDXray comparison use these converted
manifests. Raw LDXray images are not included. See
[`splits/ldxray/README.md`](splits/ldxray/README.md) for details.

### Training on LDXray

Place the official LDXray download so `dataset/train_A`, `dataset/train_B`,
`dataset/test_A`, and `dataset/test_B` resolve below `DATA_ROOT`, then run:

```bash
DATA_ROOT=/path/to/LDXray bash scripts/run_ldxray.sh
```

The Full HASNet architecture is unchanged for LDXray; only the classifier label
space and dataset-specific training configuration change. LDXray uses 12
image-level categories, and bounding-box coordinates are never used as
supervision. The default global batch remains 64 on both two GPUs and one GPU.
The released LDXray model training seed is 3408; it is independent of split
seed 42.

Evaluate an LDXray checkpoint with the shared evaluation entry point:

```bash
python scripts/evaluate.py \
  --config configs/paper/full_convnext_ldxray.yaml \
  --checkpoint <LDXRAY_CHECKPOINT> \
  --split test \
  --data-root /path/to/LDXray \
  --weights ema
```

## Released training configuration

The released DvXray configuration uses:

- shared ConvNeXt-Tiny initialized from ImageNet-1K;
- independently sampled OL/SD training augmentation;
- 256×256 inputs, 60 epochs, and global batch size 64;
- AdamW with learning rate `7e-5` and weight decay `1e-3`;
- six-epoch linear warmup followed by epoch-wise cosine annealing;
- AMP, gradient clipping at 5.0, and EMA;
- ASL with `gamma_neg=4`, `gamma_pos=0`, and clip `0.05`;
- auxiliary loss weight `alpha=0.7`;
- one 256-channel BiFPN layer;
- seed `3407` and deterministic execution enabled;
- no training or test-time augmentation at evaluation.

Run the released two-GPU entry point:

```bash
bash scripts/run_full.sh
```

For one GPU, keep the same global batch size with:

```bash
NPROC_PER_NODE=1 bash scripts/run_full.sh
```

The equivalent direct command is:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train.py \
  --config configs/paper/full_convnext.yaml \
  --seed 3407 \
  --data-root . \
  --output-dir outputs
```

The released script fixes the random seed and deterministic execution settings
for reproducible public training. Minor numerical differences may still occur
across software and hardware environments.

## Evaluation

```bash
python scripts/evaluate.py \
  --config configs/paper/full_convnext.yaml \
  --checkpoint outputs/hasnet_full_convnext/seed_3407/best.pt \
  --split test \
  --weights ema
```

Evaluation writes a JSON summary and an NPZ archive containing logits,
probabilities, targets, sample indices, and provenance metadata. Test-time
augmentation is disabled in both released configurations.

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the exact released
configuration map.

## License and citation

Original code in this repository is licensed under the MIT License. DvXray and
LDXray retain their own terms. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)
and [`CITATION.cff`](CITATION.cff).
