import numpy as np
import torch

from hasnet.losses import AsymmetricLoss
from hasnet.metrics import (
    average_precision_per_class,
    compute_calibration_metrics,
    compute_metrics,
)


def test_auxiliary_loss_weights_sum_to_one():
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05)
    targets = torch.tensor([[1.0, 0.0]])
    main = torch.tensor([[1.2, -0.3]])
    ol = torch.tensor([[0.7, -0.4]])
    sd = torch.tensor([[1.0, -0.2]])
    alpha = 0.7
    expected = (
        alpha * criterion._loss(main, targets)
        + 0.15 * criterion._loss(ol, targets)
        + 0.15 * criterion._loss(sd, targets)
    ).mean()
    actual = criterion(main, targets, aux_logits=(ol, sd), alpha=alpha)
    assert torch.allclose(actual, expected)


def test_average_precision_and_macro_metrics():
    probs = np.array([[0.9, 0.1], [0.8, 0.7], [0.2, 0.6], [0.1, 0.2]])
    targets = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    per_class = average_precision_per_class(probs, targets)
    assert np.allclose(per_class, [5 / 6, 1.0])
    metrics = compute_metrics(probs, targets)
    assert np.isclose(metrics["mAP"], 11 / 12)


def test_calibration_is_classwise_full_range_ten_bins():
    probs = np.array([[0.1, 0.9], [0.2, 0.8], [0.8, 0.2], [0.9, 0.1]])
    targets = np.array([[0, 1], [0, 1], [1, 0], [1, 0]])
    result = compute_calibration_metrics(probs, targets, num_bins=10)
    assert result["num_bins"] == 10
    assert result["range"] == [0.0, 1.0]
    assert np.isclose(result["Macro-Brier"], 0.025)
    assert np.isclose(result["Macro-ECE"], 0.15)
