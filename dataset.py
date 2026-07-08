"""DvXray dual-view multi-label dataset loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from timm.data import create_transform

MEAN = [0.91584104, 0.92976110, 0.93956200]
STD = [0.22090791, 0.18612830, 0.16510210]


class DvXrayDataset(Dataset):
    """Read DvXray split files formatted as OL#SD#multi_hot#OL_boxes#SD_boxes."""

    def __init__(self, split_file: str | Path, img_size: int = 256, train: bool = True, data_root: str | Path = "."):
        self.split_file = Path(split_file)
        self.data_root = Path(data_root)
        with open(self.split_file, "r", encoding="utf-8") as f:
            self.lines = [line.strip() for line in f if line.strip()]
        self.transform = build_transform(train, img_size)

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
        image_ol = Image.open(ol_path).convert("RGB")
        image_sd = Image.open(sd_path).convert("RGB")
        return self.transform(image_ol), self.transform(image_sd), torch.from_numpy(labels)


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
            mean=MEAN,
            std=STD,
        )
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
