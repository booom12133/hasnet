from pathlib import Path

import numpy as np
import torch
from PIL import Image

from hasnet.data import DvXrayDataset, Perturbation


def _make_fixture(tmp_path: Path) -> Path:
    image_dir = tmp_path / "data" / "DvXray"
    image_dir.mkdir(parents=True)
    gradient = np.tile(np.arange(32, dtype=np.uint8), (32, 1)) * 8
    rgb = np.stack([gradient, np.flipud(gradient), gradient], axis=-1)
    Image.fromarray(rgb).save(image_dir / "sample_OL.png")
    Image.fromarray(np.fliplr(rgb)).save(image_dir / "sample_SD.png")
    split = tmp_path / "split.txt"
    split.write_text(
        "./data/DvXray/sample_OL.png#./data/DvXray/sample_SD.png#"
        + ",".join(["1"] + ["0"] * 14)
        + "#[]#[]\n",
        encoding="utf-8",
    )
    return split


def test_missing_is_zero_at_model_input(tmp_path):
    split = _make_fixture(tmp_path)
    dataset = DvXrayDataset(
        split,
        img_size=32,
        train=False,
        data_root=tmp_path,
        perturbation=Perturbation(kind="missing", view="ol"),
    )
    sample = dataset[0]
    assert torch.count_nonzero(sample["image_ol"]) == 0
    assert torch.count_nonzero(sample["image_sd"]) > 0


def test_blur_uses_requested_view_only(tmp_path):
    split = _make_fixture(tmp_path)
    clean = DvXrayDataset(split, img_size=32, train=False, data_root=tmp_path)[0]
    blurred = DvXrayDataset(
        split,
        img_size=32,
        train=False,
        data_root=tmp_path,
        perturbation=Perturbation(kind="blur", view="sd", kernel_size=21, sigma=5),
    )[0]
    assert torch.equal(clean["image_ol"], blurred["image_ol"])
    assert not torch.equal(clean["image_sd"], blurred["image_sd"])
