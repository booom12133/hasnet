# Third-party notices

This repository contains only original release code plus calls to PyTorch,
torchvision, timm, NumPy, Pillow, PyYAML, tqdm, THOP, and Matplotlib under their
respective licenses.

The model uses standard ideas described by the following publications:
ConvNeXt, Coordinate Attention, Visual Attention Network/LKA, EfficientDet/
BiFPN, and Asymmetric Loss. Their concepts are cited in the manuscript; no
source file from their reference repositories is copied here.

The DvXray dataset and its images are not distributed. They remain governed by
the terms of the official DvXray repository.

Literature comparator implementations (CHR, AHCR, DAGNet, SXMNet, DOAM,
MVCNN, and GVCNN) are not copied into this MIT-licensed repository. Their
papers and upstream resources are listed in `docs/THIRD_PARTY_BASELINES.md`.
The `hasnet.models.baselines` module contains only the original controls
defined directly in the HASNet manuscript: single-view, Plain, and Feature
Fusion.
