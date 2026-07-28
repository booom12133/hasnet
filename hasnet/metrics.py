"""Version-stable multi-label ranking and calibration metrics.

The paper treats the 15 labels as independent one-vs-rest tasks and then
macro-averages class-wise values.  Implementing the definitions directly keeps
results independent of torchmetrics API/version changes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _as_arrays(
    probabilities: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probs = (
        probabilities.detach().float().cpu().numpy()
        if torch.is_tensor(probabilities)
        else np.asarray(probabilities, dtype=np.float32)
    )
    truth = (
        targets.detach().long().cpu().numpy()
        if torch.is_tensor(targets)
        else np.asarray(targets, dtype=np.int64)
    )
    if probs.ndim != 2 or truth.shape != probs.shape:
        raise ValueError(f"Expected matching [N,C] arrays, got {probs.shape} and {truth.shape}")
    if not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
        raise ValueError("Probabilities must be finite and in [0,1]")
    if not np.isin(truth, [0, 1]).all():
        raise ValueError("Targets must be binary")
    return probs.astype(np.float64), truth.astype(np.int64)


def average_precision_per_class(
    probabilities: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
) -> np.ndarray:
    probs, truth = _as_arrays(probabilities, targets)
    values = np.empty(probs.shape[1], dtype=np.float64)
    for class_index in range(probs.shape[1]):
        y = truth[:, class_index]
        positives = int(y.sum())
        if positives == 0:
            values[class_index] = np.nan
            continue
        order = np.argsort(-probs[:, class_index], kind="mergesort")
        ranked = y[order]
        precision = np.cumsum(ranked) / np.arange(1, ranked.size + 1)
        values[class_index] = float(precision[ranked == 1].sum() / positives)
    return values


def auroc_per_class(
    probabilities: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
) -> np.ndarray:
    probs, truth = _as_arrays(probabilities, targets)
    values = np.empty(probs.shape[1], dtype=np.float64)
    for class_index in range(probs.shape[1]):
        y = truth[:, class_index]
        positives = int(y.sum())
        negatives = int((1 - y).sum())
        if positives == 0 or negatives == 0:
            values[class_index] = np.nan
            continue
        order = np.argsort(-probs[:, class_index], kind="mergesort")
        ranked = y[order]
        tpr = np.concatenate([[0.0], np.cumsum(ranked) / positives, [1.0]])
        fpr = np.concatenate([[0.0], np.cumsum(1 - ranked) / negatives, [1.0]])
        values[class_index] = float(
            np.sum((tpr[1:] + tpr[:-1]) * (fpr[1:] - fpr[:-1]) * 0.5)
        )
    return values


def compute_metrics(
    probabilities: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    probs, truth = _as_arrays(probabilities, targets)
    prediction = probs >= threshold
    tp = np.logical_and(prediction, truth == 1).sum(axis=0)
    fp = np.logical_and(prediction, truth == 0).sum(axis=0)
    fn = np.logical_and(~prediction, truth == 1).sum(axis=0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, np.finfo(float).eps)
    per_class_ap = average_precision_per_class(probs, truth)
    per_class_auc = auroc_per_class(probs, truth)
    return {
        "mAP": float(np.nanmean(per_class_ap)),
        "macro_AUROC": float(np.nanmean(per_class_auc)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_F1": float(np.mean(f1)),
        "threshold": float(threshold),
        "per_class_AP": per_class_ap.tolist(),
        "per_class_AUROC": per_class_auc.tolist(),
    }


def compute_calibration_metrics(
    probabilities: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    num_bins: int = 10,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Class-wise Brier/NLL/ECE, macro-averaged over all classes.

    ECE uses ``num_bins`` equal-width bins over the complete [0,1] range.
    """

    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    probs, truth = _as_arrays(probabilities, targets)
    clipped = np.clip(probs, eps, 1.0 - eps)
    brier = np.mean((probs - truth) ** 2, axis=0)
    nll = np.mean(-(truth * np.log(clipped) + (1 - truth) * np.log(1 - clipped)), axis=0)
    bin_ids = np.digitize(probs, np.linspace(0.0, 1.0, num_bins + 1)[1:-1], right=False)
    ece = np.zeros(probs.shape[1], dtype=np.float64)
    for class_index in range(probs.shape[1]):
        for bin_index in range(num_bins):
            mask = bin_ids[:, class_index] == bin_index
            if not mask.any():
                continue
            confidence = float(probs[mask, class_index].mean())
            empirical_rate = float(truth[mask, class_index].mean())
            ece[class_index] += mask.mean() * abs(empirical_rate - confidence)
    return {
        "Macro-Brier": float(brier.mean()),
        "Macro-NLL": float(nll.mean()),
        "Macro-ECE": float(ece.mean()),
        "num_bins": int(num_bins),
        "range": [0.0, 1.0],
        "per_class_Brier": brier.tolist(),
        "per_class_NLL": nll.tolist(),
        "per_class_ECE": ece.tolist(),
    }


def reliability_curve(
    probabilities: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    num_bins: int = 10,
) -> dict[str, list[float | int]]:
    """Macro class-wise reliability curve using the same bins as Macro-ECE."""

    probs, truth = _as_arrays(probabilities, targets)
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_ids = np.digitize(probs, edges[1:-1], right=False)
    confidence, empirical_rate, count = [], [], []
    for bin_index in range(num_bins):
        class_conf, class_rate, class_count = [], [], []
        for class_index in range(probs.shape[1]):
            mask = bin_ids[:, class_index] == bin_index
            if mask.any():
                class_conf.append(float(probs[mask, class_index].mean()))
                class_rate.append(float(truth[mask, class_index].mean()))
                class_count.append(int(mask.sum()))
        confidence.append(float(np.mean(class_conf)) if class_conf else float("nan"))
        empirical_rate.append(float(np.mean(class_rate)) if class_rate else float("nan"))
        count.append(int(np.sum(class_count)))
    return {
        "bin_lower": edges[:-1].tolist(),
        "bin_upper": edges[1:].tolist(),
        "confidence": confidence,
        "empirical_rate": empirical_rate,
        "count": count,
    }
