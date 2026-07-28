"""Exponential moving average used for validation and released checkpoints."""

from __future__ import annotations

import math
from copy import deepcopy

import torch
from torch import nn

from hasnet.distributed import unwrap_model


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999, warmup_updates: int = 2000):
        self.module = deepcopy(unwrap_model(model)).eval()
        self.decay = float(decay)
        self.warmup_updates = int(warmup_updates)
        self.updates = 0
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    def _decay(self) -> float:
        return self.decay * (1.0 - math.exp(-self.updates / max(self.warmup_updates, 1)))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = self._decay()
        source = unwrap_model(model).state_dict()
        for key, value in self.module.state_dict().items():
            source_value = source[key].detach()
            if value.dtype.is_floating_point:
                value.mul_(decay).add_(source_value, alpha=1.0 - decay)
            else:
                value.copy_(source_value)

    def state_dict(self) -> dict:
        return {
            "module": self.module.state_dict(),
            "updates": self.updates,
            "decay": self.decay,
            "warmup_updates": self.warmup_updates,
        }

    def load_state_dict(self, state: dict) -> None:
        self.module.load_state_dict(state["module"], strict=True)
        self.updates = int(state.get("updates", 0))
