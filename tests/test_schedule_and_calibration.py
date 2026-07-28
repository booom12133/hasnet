import torch

from hasnet.calibration import TemperatureScaler
from hasnet.schedule import PaperV1Schedule


def test_warmup_uses_loader_length_not_hardcoded_steps():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = PaperV1Schedule(
        optimizer,
        epochs=4,
        steps_per_epoch=10,
        warmup_epochs=1,
        eta_min_ratio=0.01,
    )
    assert optimizer.param_groups[0]["lr"] == 0.1
    scheduler.step_batch()
    assert optimizer.param_groups[0]["lr"] == 0.2
    for _ in range(9):
        scheduler.step_batch()
    assert optimizer.param_groups[0]["lr"] == 1.0


def test_temperature_is_positive_and_finite():
    logits = torch.tensor(
        [[4.0, -3.0], [2.0, -1.0], [-2.0, 3.0], [-4.0, 1.0]],
        dtype=torch.float32,
    )
    targets = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=torch.float32)
    scaler = TemperatureScaler()
    temperature = scaler.fit(logits, targets, max_iter=30)
    assert 0.0 < temperature < 100.0
    assert torch.isfinite(scaler(logits)).all()
