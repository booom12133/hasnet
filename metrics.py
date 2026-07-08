"""Evaluation metrics for multi-label classification."""

from __future__ import annotations

import torch
from torchmetrics.classification import (
    MultilabelAveragePrecision,
    MultilabelAUROC,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)


@torch.no_grad()
def compute_metrics(probs: torch.Tensor, targets: torch.Tensor, num_labels: int = 15, threshold: float = 0.5) -> dict[str, float]:
    probs = probs.detach().float().cpu()
    targets = targets.detach().long().cpu()
    metrics = {
        "mAP": MultilabelAveragePrecision(num_labels=num_labels, average="macro"),
        "AUROC": MultilabelAUROC(num_labels=num_labels, average="macro"),
        "Precision": MultilabelPrecision(num_labels=num_labels, average="macro", threshold=threshold),
        "Recall": MultilabelRecall(num_labels=num_labels, average="macro", threshold=threshold),
        "F1": MultilabelF1Score(num_labels=num_labels, average="macro", threshold=threshold),
    }
    out = {}
    for name, metric in metrics.items():
        try:
            out[name] = float(metric(probs, targets))
        except Exception:
            out[name] = float("nan")
    return out
