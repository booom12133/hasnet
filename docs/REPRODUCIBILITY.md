# Released configuration map

This public release supports the Full HASNet training and evaluation path on
DvXray and on the converted LDXray image-level task. Dataset images and model
checkpoints are not included.

## Shared model

Both released configurations build the same Full HASNet architecture:

```text
shared ConvNeXt-Tiny dual stream
  -> VSC
  -> per-view CoordAtt
  -> CVFUSE
  -> LKA
  -> SRC-Head (MAIN + AUX/GATE + one-layer 256-channel BiFPN)
```

The auxiliary gate and BiFPN logit coefficients are initialized in
`hasnet/models/hasnet.py` as `0.0` and `0.1`, respectively.

## DvXray

| Item | Released setting |
|---|---|
| Config | `configs/paper/full_convnext.yaml` |
| Launcher | `scripts/run_full.sh` |
| Classes | 15 |
| Split sizes | 11,200 train / 3,200 validation / 1,600 test |
| Input | synchronized OL/SD, 256×256 |
| Epochs / global batch | 60 / 64 |
| Optimizer | AdamW, LR 7e-5, weight decay 1e-3 |
| Schedule | 6-epoch linear warmup + epoch-wise cosine, eta_min=LR/100 |
| AMP / EMA | enabled / enabled |
| EMA | decay 0.9999, warmup 2,000 updates |
| Loss | ASL (gamma-=4, gamma+=0, clip=0.05), auxiliary alpha 0.7 |
| Seed / deterministic | 3407 / enabled |
| TTA | disabled |
| Selection | highest validation mAP from epoch 32 onward |

OL and SD augmentations are sampled independently. With two processes, the
global batch of 64 becomes 32 synchronized pairs per process. The global batch
remains 64 when `NPROC_PER_NODE=1`.

## LDXray

| Item | Released setting |
|---|---|
| Config | `configs/paper/full_convnext_ldxray.yaml` |
| Launcher | `scripts/run_ldxray.sh` |
| Classes / model-output order | 12: MP, OL, PC1, LA, GL, PC2, TA, BL, CO, NL, UM, CG |
| Split sizes | 99,133 train / 11,015 validation / 36,849 test |
| Input | synchronized OL/SD, 256×256 |
| Epochs / global batch | 60 / 64 |
| Optimizer | AdamW, LR 7e-5, weight decay 1e-3 |
| Schedule | 1,050-update linear warmup + epoch-wise cosine, eta_min=LR/100 |
| AMP / EMA | enabled / enabled |
| EMA | decay 0.9999, warmup 2,000 updates |
| Loss | ASL (gamma-=4, gamma+=0, clip=0.05), auxiliary alpha 0.7 |
| Released seed / deterministic | 3408 / enabled |
| TTA | disabled |
| Selection | highest validation mAP from epoch 32 onward |

The LDXray test split is the official test split. Split seed 42 applies only to
the iterative 9:1 train/validation partition and is unrelated to model training
seed 3408. Bounding boxes are not used as supervision.Table 3 of the paper uses a presentation-only class reorder; see `splits/ldxray/README.md` for the mapping.

## Reproducibility note

The launchers fix the released seeds and deterministic execution settings.
Exact numerical identity is not guaranteed across different PyTorch, CUDA,
cuDNN, driver, and GPU combinations.
