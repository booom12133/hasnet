"""DvXray dual-view multi-label dataset and paper-exact perturbations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from timm.data import create_transform

from hasnet.constants import CLASS_NAMES, DVXRAY_MEAN, DVXRAY_STD


@dataclass(frozen=True)
class Perturbation:
    """One-view degradation used in the robustness table."""

    kind: str = "clean"
    view: str | None = None
    kernel_size: int = 21
    sigma: float = 5.0

    def __post_init__(self):
        if self.kind not in {"clean", "missing", "blur"}:
            raise ValueError("kind must be clean, missing, or blur")
        if self.kind != "clean" and self.view not in {"ol", "sd"}:
            raise ValueError("view must be ol or sd for a degraded condition")
        if self.kernel_size % 2 != 1:
            raise ValueError("Gaussian kernel_size must be odd")

    @property
    def name(self) -> str:
        return "clean" if self.kind == "clean" else f"{self.view}_{self.kind}"


class DvXrayDataset(Dataset):
    """Read DvXray split files formatted as OL#SD#multi_hot#OL_boxes#SD_boxes."""

    def __init__(
        self,
        split_file: str | Path,
        img_size: int = 256,
        train: bool = True,
        data_root: str | Path = ".",
        perturbation: Perturbation | None = None,
    ):
        self.split_file = Path(split_file)
        self.data_root = Path(data_root)
        with open(self.split_file, "r", encoding="utf-8") as f:
            self.lines = [line.strip() for line in f if line.strip()]
        self.transform = build_transform(train, img_size)
        self.train = train
        self.perturbation = perturbation or Perturbation()
        if train and self.perturbation.kind != "clean":
            raise ValueError("Paper robustness perturbations are evaluation-only")

    def __len__(self) -> int:
        return len(self.lines)

    def _resolve_path(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.data_root / p

    def __getitem__(self, idx: int):
        parts = self.lines[idx].split("#")
        if len(parts) < 3:
            raise ValueError(f"Invalid annotation line: {self.lines[idx]}")
        ol_path = self._resolve_path(parts[0].strip())
        sd_path = self._resolve_path(parts[1].strip())
        labels = np.array([int(v) for v in parts[2].split(",")], dtype=np.float32)
        if labels.shape != (len(CLASS_NAMES),) or not np.isin(labels, [0, 1]).all():
            raise ValueError(
                f"Expected {len(CLASS_NAMES)} binary labels at line {idx + 1}, got {parts[2]!r}"
            )
        if not ol_path.is_file() or not sd_path.is_file():
            missing = ol_path if not ol_path.is_file() else sd_path
            raise FileNotFoundError(
                f"Image not found: {missing}. Set data.root so split paths resolve to data/DvXray/."
            )
        with Image.open(ol_path) as image:
            image_ol = image.convert("RGB")
        with Image.open(sd_path) as image:
            image_sd = image.convert("RGB")
        # This intentionally matches the experiment code: stochastic transforms
        # are sampled independently for OL and SD.
        image_ol = self.transform(image_ol)
        image_sd = self.transform(image_sd)
        image_ol, image_sd = self._apply_perturbation(image_ol, image_sd)
        return {
            "image_ol": image_ol,
            "image_sd": image_sd,
            "target": torch.from_numpy(labels),
            "index": idx,
            "ol_path": str(ol_path),
            "sd_path": str(sd_path),
        }

    def _apply_perturbation(
        self, image_ol: torch.Tensor, image_sd: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        perturbation = self.perturbation
        if perturbation.kind == "clean":
            return image_ol, image_sd
        selected = image_ol if perturbation.view == "ol" else image_sd
        if perturbation.kind == "missing":
            selected = torch.zeros_like(selected)
        else:
            selected = TF.gaussian_blur(
                selected,
                kernel_size=[perturbation.kernel_size, perturbation.kernel_size],
                sigma=[perturbation.sigma, perturbation.sigma],
            )
        if perturbation.view == "ol":
            image_ol = selected
        else:
            image_sd = selected
        return image_ol, image_sd


def build_transform(is_train: bool, img_size: int = 256):
    if is_train:
        return create_transform(
            input_size=img_size,
            is_training=True,
            color_jitter=0.4,
            auto_augment="rand-m9-mstd0.5-inc1",
            re_prob=0.25,
            re_mode="pixel",
            re_count=1,
            interpolation="bicubic",
            mean=DVXRAY_MEAN,
            std=DVXRAY_STD,
        )
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=DVXRAY_MEAN, std=DVXRAY_STD),
    ])
