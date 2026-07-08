# Dual-View X-ray Multi-Label Recognition

This repository contains the cleaned implementation of the proposed **full model** for the paper:

> Dual-View X-ray Multi-Label Recognition via Role-Separated Cross-View Fusion and Decision-Level Refinement

The released code intentionally keeps only the proposed full model: shared dual-stream backbone, VSC pre-fusion calibration, DCAF cross-view semantic fusion, and SRC-Head decision-level refinement. Baseline models, ablation switches, temporary scripts, logs, caches, and third-party copied repositories are not included.

## Requirements

```bash
pip install -r requirements.txt
```

Recommended environment:

- Python 3.10+
- PyTorch 2.0+
- torchvision 0.15+
- CUDA-enabled GPU for training

## Dataset

This code uses the public DvXray dataset.

Please download DvXray from the official repository:

https://github.com/Mbwslib/DvXray

After downloading, place or symlink the dataset images as:

```text
data/DvXray/
```

The provided split files use the following line format:

```text
OL_image_path#SD_image_path#15-dim_multi_hot_label#OL_boxes#SD_boxes
```

Only the OL/SD image paths and multi-label targets are used by this image-level recognition code.

## Training

```bash
python train_full.py --data_root . --epochs 60 --batch_size 64 --img_size 256
```

The default configuration follows the paper's full model setting:

- ConvNeXt-Tiny shared dual-stream backbone
- image size 256
- ASL loss with gamma_neg=4, gamma_pos=0, clip=0.05
- AdamW, lr=7e-5, weight decay=1e-3
- auxiliary loss alpha=0.7
- VSC + DCAF + SRC-Head enabled by design

## Evaluation

```bash
python train_full.py --eval --checkpoint checkpoints/best.pth --data_root .
```

## Params/FLOPs

```bash
python scripts/profile_model.py --backbone convnext --img_size 256
```

## Code availability statement

The source code needed to train and evaluate the proposed full model is available in this repository. The DvXray dataset is publicly available from its original repository. Trained weights are not included by default.

## License

Please add a license after confirming the licensing status of all dependencies and any reused code. MIT or Apache-2.0 is usually suitable for academic code if all included code is original or license-compatible.
