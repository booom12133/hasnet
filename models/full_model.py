"""Full model for dual-view X-ray multi-label recognition.

This file only keeps the proposed full model used in the paper:
shared dual-stream backbone + VSC + DCAF + SRC-Head.
Ablation switches, baseline models, and temporary experimental modules are intentionally removed.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


def get_activation(name: str = "silu") -> nn.Module:
    name = (name or "identity").lower()
    if name in {"silu", "swish"}:
        return nn.SiLU(inplace=True)
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name in {"identity", "none"}:
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {name}")


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, act: str = "silu"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            get_activation(act),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DWConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, act: str = "silu"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size // 2, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            get_activation(act),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HSwish(nn.Module):
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.relu(x + 3) / 6


class CoordAtt(nn.Module):
    """Coordinate attention used for per-view refinement before cross-view fusion."""

    def __init__(self, inp: int, oup: int | None = None, reduction: int = 32):
        super().__init__()
        oup = inp if oup is None else oup
        mip = max(8, inp // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = HSwish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return identity * a_w * a_h


class CSPRepLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, expansion: float = 1.0, act: str = "silu"):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.local_branch = DWConv(in_channels, hidden_channels, act=act)
        self.channel_branch = ConvBNAct(in_channels, hidden_channels, 1, act=act)
        self.project = ConvBNAct(hidden_channels, out_channels, 1, act=act) if hidden_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.local_branch(x) + self.channel_branch(x))


class CVFUSE(nn.Module):
    """Compact cross-view fusion block inside DCAF."""

    def __init__(self, in_channels: int, out_channels: int, depth_mult: int = 3, act: str = "silu"):
        super().__init__()
        view_channels = in_channels // 2
        self.view_refine = CoordAtt(view_channels, view_channels)
        self.bn = nn.BatchNorm2d(in_channels)
        self.fuse = CSPRepLayer(in_channels, out_channels, expansion=1.0, act=act)

    def forward(self, feat_ol: torch.Tensor, feat_sd: torch.Tensor) -> torch.Tensor:
        feat_ol = self.view_refine(feat_ol)
        feat_sd = self.view_refine(feat_sd)
        fused = torch.cat([feat_ol, feat_sd], dim=1).contiguous()
        return self.fuse(self.bn(fused))


class LKA(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 7, dilation: int = 3):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, kernel_size=5, padding=dilation * 2, groups=dim, dilation=dilation)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.conv1(self.conv_spatial(self.conv0(x))))


class LKABlock(nn.Module):
    def __init__(self, dim: int, expansion: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.lka = LKA(dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = dim * expansion
        self.ffn = nn.Sequential(nn.Conv2d(dim, hidden, 1), nn.SiLU(inplace=True), nn.Conv2d(hidden, dim, 1))

    def _norm(self, x: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w).permute(0, 2, 1)
        x = norm(x)
        return x.permute(0, 2, 1).reshape(b, c, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.lka(self._norm(x, self.norm1))
        x = x + self.ffn(self._norm(x, self.norm2))
        return x


class BiFPNConv(nn.Module):
    def __init__(self, channels: int, act: str = "silu"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            get_activation(act),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class WeightedFusion(nn.Module):
    def __init__(self, num_inputs: int, channels: int, act: str = "silu"):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_inputs))
        self.conv = BiFPNConv(channels, act=act)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        w = F.relu(self.weights)
        out = sum(x * w[i] for i, x in enumerate(inputs)) / (w.sum() + 1e-5)
        return self.conv(out)


class BiFPN(nn.Module):
    """A compact 3-level BiFPN used as the multi-scale support branch in SRC-Head."""

    def __init__(self, channels: int = 256, act: str = "silu"):
        super().__init__()
        self.p5_td = WeightedFusion(2, channels, act)
        self.p4_td = WeightedFusion(3, channels, act)
        self.p3_out = WeightedFusion(2, channels, act)
        self.p4_out = WeightedFusion(2, channels, act)
        self.p5_out = WeightedFusion(2, channels, act)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.downsample = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=2, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            get_activation(act),
        )

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p5_td = self.p5_td([p5, self.downsample(p4)])
        p4_td = self.p4_td([p4, self.upsample(p5_td), self.downsample(p3)])
        p3_out = self.p3_out([p3, self.upsample(p4_td)])
        p4_out = self.p4_out([p4_td, self.downsample(p3_out)])
        p5_out = self.p5_out([p5_td, self.downsample(p4_out)])
        return p3_out, p4_out, p5_out


class ViewStatCalibrator(nn.Module):
    """VSC: residual affine normalization before DCAF."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.affine_bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.res_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = (x - mean).pow(2).mean(dim=(2, 3), keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x + self.res_scale * (x_norm * self.affine_weight + self.affine_bias)


def build_backbone(name: str = "convnext", pretrained: bool = True):
    weights = "IMAGENET1K_V1" if pretrained else None
    name = name.lower()
    if name == "convnext":
        net = models.convnext_tiny(weights=weights).features
        stem = nn.Sequential(net[0])
        layer1 = net[1]
        layer2 = net[2:4]
        layer3 = net[4:6]
        layer4 = net[6:]
        channels = [96, 192, 384, 768]
    elif name == "resnet50":
        net = models.resnet50(weights=weights)
        stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        layer1, layer2, layer3, layer4 = net.layer1, net.layer2, net.layer3, net.layer4
        channels = [256, 512, 1024, 2048]
    elif name == "resnext50":
        net = models.resnext50_32x4d(weights=weights)
        stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        layer1, layer2, layer3, layer4 = net.layer1, net.layer2, net.layer3, net.layer4
        channels = [256, 512, 1024, 2048]
    elif name == "regnetx_3_2gf":
        net = models.regnet_x_3_2gf(weights=weights)
        stem = nn.Sequential(net.stem[0], net.stem[1], net.stem[2])
        layer1, layer2, layer3, layer4 = (
            net.trunk_output.block1,
            net.trunk_output.block2,
            net.trunk_output.block3,
            net.trunk_output.block4,
        )
        channels = [96, 192, 432, 1008]
    else:
        raise ValueError(f"Unsupported backbone: {name}")
    return stem, layer1, layer2, layer3, layer4, channels


class FullModel(nn.Module):
    """Proposed full model: Shared backbone + VSC + DCAF + SRC-Head."""

    def __init__(
        self,
        num_classes: int = 15,
        backbone: str = "convnext",
        pretrained: bool = True,
        fusion_dim_index: int = 2,
        fpn_channels: int = 256,
        act: str = "silu",
    ):
        super().__init__()
        self.stem, self.layer1, self.layer2, self.layer3, self.layer4, channels = build_backbone(backbone, pretrained)
        top_channels = channels[3]
        fusion_dim = channels[fusion_dim_index]

        # Stage 1: lightweight pre-fusion calibration
        self.vsc = ViewStatCalibrator(top_channels)

        # Stage 2: DCAF high-level semantic fusion
        self.dcaf = CVFUSE(top_channels * 2, fusion_dim, depth_mult=3, act=act)
        self.lka = LKABlock(fusion_dim)
        self.main_classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(fusion_dim, num_classes))

        # Stage 3: SRC-Head, view-specific gate and compact multi-scale support
        self.view_classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(top_channels, num_classes))
        self.logit_gate_scale = nn.Parameter(torch.tensor(0.0))

        self.p3_reduce = nn.Sequential(nn.Conv2d(channels[1] * 2, fpn_channels, 1, bias=False), nn.BatchNorm2d(fpn_channels), get_activation(act))
        self.p4_reduce = nn.Sequential(nn.Conv2d(channels[2] * 2, fpn_channels, 1, bias=False), nn.BatchNorm2d(fpn_channels), get_activation(act))
        self.p5_reduce = nn.Sequential(nn.Conv2d(channels[3] * 2, fpn_channels, 1, bias=False), nn.BatchNorm2d(fpn_channels), get_activation(act))
        self.bifpn = BiFPN(fpn_channels, act=act)
        self.bifpn_classifier = nn.Linear(fpn_channels * 3, num_classes)
        self.fpn_logit_scale = nn.Parameter(torch.tensor(0.1))

    def _extract_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return f2, f3, f4

    def forward(self, image_ol: torch.Tensor, image_sd: torch.Tensor, return_debug: bool = False):
        f2_ol, f3_ol, f4_ol = self._extract_features(image_ol)
        f2_sd, f3_sd, f4_sd = self._extract_features(image_sd)

        # MAIN: VSC -> DCAF -> LKA -> classifier
        f4_ol_cal = self.vsc(f4_ol)
        f4_sd_cal = self.vsc(f4_sd)
        fused = self.lka(self.dcaf(f4_ol_cal, f4_sd_cal))
        z_main = self.main_classifier(fused)

        # AUX/GATE: confidence-derived class-wise view weighting
        z_ol = self.view_classifier(f4_ol)
        z_sd = self.view_classifier(f4_sd)
        conf = torch.stack([torch.abs(torch.sigmoid(z_ol) - 0.5), torch.abs(torch.sigmoid(z_sd) - 0.5)], dim=1)
        weights = F.softmax(conf, dim=1)
        w_ol, w_sd = weights[:, 0, :], weights[:, 1, :]
        z_gate = w_ol * z_ol + w_sd * z_sd

        # BiFPN: compact multi-scale support
        p3 = self.p3_reduce(torch.cat([f2_ol, f2_sd], dim=1))
        p4 = self.p4_reduce(torch.cat([f3_ol, f3_sd], dim=1))
        p5 = self.p5_reduce(torch.cat([f4_ol, f4_sd], dim=1))
        p3, p4, p5 = self.bifpn(p3, p4, p5)
        z_bi = self.bifpn_classifier(torch.cat([
            F.adaptive_avg_pool2d(p3, 1).flatten(1),
            F.adaptive_avg_pool2d(p4, 1).flatten(1),
            F.adaptive_avg_pool2d(p5, 1).flatten(1),
        ], dim=1))

        logits = z_main + self.logit_gate_scale * z_gate + self.fpn_logit_scale * z_bi

        aux_logits = (z_ol, z_sd)
        if return_debug:
            return logits, aux_logits, {
                "z_main": z_main,
                "z_gate": z_gate,
                "z_bi": z_bi,
                "w_ol": w_ol,
                "w_sd": w_sd,
                "final_probs": torch.sigmoid(logits),
            }
        return logits, aux_logits
