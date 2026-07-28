"""Checkpoint I/O with enough metadata to audit a reported run."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import torchvision
from torch import nn

from hasnet.config import config_hash, serializable_config
from hasnet.distributed import unwrap_model


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment_metadata(repository_root: str | Path = ".") -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": commit,
    }


def build_checkpoint(
    *,
    model: nn.Module,
    ema,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_metric: float,
    config: dict,
    split_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "format_version": 2,
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "model": unwrap_model(model).state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": serializable_config(config),
        "config_sha256": config_hash(config),
        "split_sha256": split_hashes,
        "environment": environment_metadata(),
    }


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    # Explicit weights_only=False is required because optimizer/config metadata
    # are part of the reproducibility record.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_model_weights(
    model: nn.Module,
    checkpoint: dict[str, Any],
    weights: str = "ema",
    strict: bool = True,
) -> None:
    if weights == "ema" and checkpoint.get("ema"):
        state = checkpoint["ema"]["module"]
    elif weights == "raw":
        state = checkpoint["model"]
    else:
        if weights not in {"ema", "raw"}:
            raise ValueError("weights must be ema or raw")
        state = checkpoint["model"]
    unwrap_model(model).load_state_dict(state, strict=strict)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
