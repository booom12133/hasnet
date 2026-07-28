"""Original, dependency-free reference baselines defined in the manuscript.

These are not reimplementations of third-party literature methods.  They are
the OL-only, SD-only, Plain, and Feature-Fusion controls whose definitions are
fully specified in the paper.
"""

from __future__ import annotations

import torch
from torch import nn

from .hasnet import build_backbone


class _SharedEncoder(nn.Module):
    def __init__(self, backbone: str = "convnext", pretrained: bool = True):
        super().__init__()
        self.stem, self.layer1, self.layer2, self.layer3, self.layer4, channels = build_backbone(
            backbone, pretrained
        )
        self.out_channels = channels[-1]

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        x = self.stem(image)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


class SingleView(_SharedEncoder):
    """OL-only or SD-only classifier; the script selects the input view."""

    def __init__(self, num_classes: int = 15, backbone: str = "convnext", pretrained: bool = True):
        super().__init__(backbone, pretrained)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(self.out_channels, num_classes)
        )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.classifier(self.encode(image)), None


class PlainDual(_SharedEncoder):
    """Shared backbone/classifier and class-wise maximum logit fusion."""

    def __init__(self, num_classes: int = 15, backbone: str = "convnext", pretrained: bool = True):
        super().__init__(backbone, pretrained)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(self.out_channels, num_classes)
        )

    def forward(self, image_ol: torch.Tensor, image_sd: torch.Tensor) -> tuple[torch.Tensor, None]:
        z_ol = self.classifier(self.encode(image_ol))
        z_sd = self.classifier(self.encode(image_sd))
        return torch.maximum(z_ol, z_sd), None


class FeatureFusion(_SharedEncoder):
    """Top-level feature concatenation followed by one linear classifier."""

    def __init__(self, num_classes: int = 15, backbone: str = "convnext", pretrained: bool = True):
        super().__init__(backbone, pretrained)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.out_channels * 2, num_classes),
        )

    def forward(self, image_ol: torch.Tensor, image_sd: torch.Tensor) -> tuple[torch.Tensor, None]:
        fused = torch.cat([self.encode(image_ol), self.encode(image_sd)], dim=1)
        return self.classifier(fused), None
