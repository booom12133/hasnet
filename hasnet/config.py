"""YAML configuration loading, inheritance, validation, and model construction."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from hasnet.models import FeatureFusion, HASNet, PlainDual, SingleView


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        current = yaml.safe_load(handle) or {}
    if not isinstance(current, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    base_entry = current.pop("base", None)
    if base_entry:
        base_path = (path.parent / base_entry).resolve()
        config = _deep_merge(load_config(base_path), current)
    else:
        config = current
    config["_config_path"] = str(path)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    for section in ("data", "model", "train", "evaluation"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Missing configuration section: {section}")
    if config["model"].get("name") not in {"hasnet", "single_view", "plain", "feature_fusion"}:
        raise ValueError(f"Unknown model.name: {config['model'].get('name')}")
    if int(config["data"].get("num_classes", 15)) != 15:
        raise ValueError("DvXray release expects exactly 15 labels")
    global_batch = int(config["train"]["global_batch_size"])
    if global_batch <= 0:
        raise ValueError("train.global_batch_size must be positive")
    if int(config["train"]["epochs"]) <= 0:
        raise ValueError("train.epochs must be positive")


def config_hash(config: dict[str, Any]) -> str:
    payload = {key: value for key, value in config.items() if not key.startswith("_")}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def serializable_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in config.items() if not key.startswith("_")}


def build_model(config: dict[str, Any], pretrained: bool | None = None):
    model_config = deepcopy(config["model"])
    name = model_config.pop("name")
    configured_pretrained = model_config.pop("pretrained", True)
    selected_pretrained = configured_pretrained if pretrained is None else pretrained
    num_classes = int(config["data"].get("num_classes", 15))
    backbone = model_config.pop("backbone", "convnext")
    if name == "hasnet":
        return HASNet(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=selected_pretrained,
            **model_config,
        )
    if name == "single_view":
        model_config.pop("view", None)
        return SingleView(num_classes=num_classes, backbone=backbone, pretrained=selected_pretrained)
    if name == "plain":
        return PlainDual(num_classes=num_classes, backbone=backbone, pretrained=selected_pretrained)
    if name == "feature_fusion":
        return FeatureFusion(num_classes=num_classes, backbone=backbone, pretrained=selected_pretrained)
    raise AssertionError("validate_config should reject unknown models")
