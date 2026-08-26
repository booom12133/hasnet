from pathlib import Path

import numpy as np
import torch
from PIL import Image

from hasnet.config import load_config
from hasnet.data import DvXrayDataset, Perturbation
from hasnet.distributed import DistributedContext
from hasnet.engine import make_loader


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


def test_ldxray_manifest_uses_twelve_label_vector(tmp_path):
    image_dir = tmp_path / "dataset" / "train_A"
    paired_dir = tmp_path / "dataset" / "train_B"
    image_dir.mkdir(parents=True)
    paired_dir.mkdir(parents=True)
    image = Image.new("RGB", (32, 32), color=(64, 96, 128))
    image.save(image_dir / "000000.jpg")
    image.save(paired_dir / "000000.jpg")
    split = tmp_path / "ldxray.txt"
    split.write_text(
        "dataset/train_A/000000.jpg#dataset/train_B/000000.jpg#"
        + ",".join(["1"] + ["0"] * 11)
        + "#[]#[]\n",
        encoding="utf-8",
    )
    sample = DvXrayDataset(
        split, img_size=32, train=False, data_root=tmp_path, num_classes=12
    )[0]
    assert sample["target"].shape == (12,)


def test_train_drop_last_reaches_distributed_sampler_and_loader(tmp_path):
    context = DistributedContext(
        distributed=True,
        rank=0,
        world_size=2,
        local_rank=0,
        device=torch.device("cpu"),
    )
    cases = (
        ("configs/paper/full_convnext.yaml", 15, True),
        ("configs/paper/full_convnext_ldxray.yaml", 12, False),
    )
    for config_path, num_classes, expected_drop_last in cases:
        manifest = tmp_path / f"{num_classes}_classes.txt"
        labels = ",".join(["1"] + ["0"] * (num_classes - 1))
        manifest.write_text(
            "".join(
                f"view_a_{index}.png#view_b_{index}.png#{labels}#[]#[]\n"
                for index in range(5)
            ),
            encoding="utf-8",
        )
        config = load_config(config_path)
        config["data"]["splits"]["train"] = str(manifest)
        config["data"]["num_workers"] = 0
        config["data"]["persistent_workers"] = False
        loader, sampler = make_loader(config, "train", context=context, train=True)
        assert sampler is not None
        assert sampler.drop_last is expected_drop_last
        assert loader.drop_last is expected_drop_last
        expected_total_size = 4 if expected_drop_last else 6
        assert sampler.total_size == expected_total_size
