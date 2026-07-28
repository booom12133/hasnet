# Reproducibility map

## Protocol invariants

All retrained configurations inherit from
`configs/paper/full_convnext.yaml`. An ablation YAML changes only the named
architectural switch. The data loader, split, optimizer, schedule, loss,
augmentation, EMA, evaluation metrics, and checkpoint selection remain shared.

| Item | Release setting |
|---|---|
| Dataset | DvXray |
| Split sizes | 11,200 train / 3,200 validation / 1,600 test |
| Input | synchronized OL/SD, 256×256 |
| Default backbone | shared ConvNeXt-Tiny |
| Hardware protocol | DDP, 2× RTX 4090 |
| Epochs / global batch | 60 / 64 |
| Optimizer | AdamW |
| Initial LR / weight decay | 7e-5 / 1e-3 |
| Schedule | 6-epoch linear warmup + epoch-wise cosine |
| AMP / EMA | enabled / enabled |
| Loss | ASL (gamma-=4, gamma+=0, clip=0.05) |
| Auxiliary alpha | 0.7 |
| BiFPN | 1 layer, 256 channels |
| TTA | disabled |
| Canonical release seeds | 3407, 3408, 3409 |
| Model selection | highest validation mAP, beginning at epoch 1 |

OL and SD augmentations are sampled independently, matching the experiment
implementation. The global batch of 64 becomes 32 samples per process under
the two-GPU protocol. Worker count is likewise divided across processes.

## Configuration-to-table map

| Manuscript analysis | Configurations / command |
|---|---|
| Full ConvNeXt-Tiny | `configs/paper/full_convnext.yaml` |
| Cross-backbone | `configs/paper/backbones/*.yaml` |
| Core three-stage ablation | `configs/paper/ablations/core/*.yaml` |
| DCAF ablation | `configs/paper/ablations/dcaf/*.yaml` |
| SRC ablation | `configs/paper/ablations/src/*.yaml` |
| OL/SD/Plain/Feature Fusion | `configs/paper/baselines/*.yaml` |
| Late Fusion Avg/Max | `scripts/late_fusion.py` |
| Temperature Scaling | `scripts/calibrate.py` |
| Reliability diagram | `scripts/reliability_diagram.py` |
| One-view degradation | `scripts/robustness.py` |
| Params/FLOPs | `scripts/profile_model.py` |

The core-ablation `base_fusion.yaml` is the bare CVFUSE projection core with
both CoordAtt and LKA disabled. It is intentionally identical in architecture
to the `CVFUSE only` row of the internal DCAF ablation; the independently
trained rows may differ slightly because results are averaged across runs.

## Metric definitions

For each of 15 classes, average precision is computed from the complete ranked
list. mAP is the arithmetic mean of the 15 class AP values.

Macro-Brier, Macro-NLL, and Macro-ECE treat every label as one independent
binary problem. Each value is first computed per class and then averaged over
classes. ECE uses 10 equal-width bins over the complete probability range
`[0,1]`; it does not discard probabilities below 0.5.

Temperature scaling cannot be fitted and scored on the same observations.
The release supports either:

1. `heldout`: fit on one prediction archive and evaluate another; or
2. `crossfit`: five-fold out-of-fold scaling for a validation-only comparison.

The protocol and fitted temperatures are stored in the output JSON.

## Robustness definitions

The five retained settings are clean, OL missing, SD missing, OL blur, and SD
blur. Missing replaces the selected normalized model-input tensor with zeros.
Blur applies `torchvision.transforms.functional.gaussian_blur` with a fixed
21×21 kernel and sigma 5 to the selected normalized tensor. Retention is:

```text
100 × perturbed mAP / clean mAP
```

## Minimum evidence for a tagged release

For every headline model or ablation, archive:

- the three `resolved_config.yaml` files;
- three `summary.json` files and the aggregate mean/std CSV;
- `best.pt` or a stable external weight URL with SHA-256;
- split SHA-256 values;
- the Git commit hash;
- Params/FLOPs JSON;
- exported validation/test predictions needed for calibration or robustness.

Do not replace a missing run with an inferred per-seed value. Results under
`results/paper/` are a transcription of the manuscript tables; generated
evidence belongs under `results/generated/`.
