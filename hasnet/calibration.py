"""Post-hoc temperature scaling for multi-label logits."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TemperatureScaler(nn.Module):
    """A single positive scalar temperature, fit with binary NLL."""

    def __init__(self, initial_temperature: float = 1.0):
        super().__init__()
        if initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive")
        self.log_temperature = nn.Parameter(torch.tensor(initial_temperature).log())

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(1e-3, 100.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        max_iter: int = 100,
    ) -> float:
        logits = logits.detach().double()
        targets = targets.detach().double()
        self.double()
        optimizer = torch.optim.LBFGS(
            [self.log_temperature], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(self(logits), targets)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.float()
        return float(self.temperature.detach().cpu())
