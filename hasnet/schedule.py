"""Learning-rate schedule matching the experiment-code update order."""

from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR


class PaperV1Schedule:
    """Linear batch warmup plus epoch-wise cosine annealing.

    The first release used six epochs of batch-wise warmup and stepped a
    60-epoch cosine scheduler after every epoch.  ``steps_per_epoch`` replaces
    the old dataset-specific constant (175); for the supplied 11,200/64 split
    it evaluates to exactly the same value.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        epochs: int,
        steps_per_epoch: int,
        warmup_epochs: int = 6,
        eta_min_ratio: float = 0.01,
    ):
        self.optimizer = optimizer
        self.epochs = int(epochs)
        self.steps_per_epoch = int(steps_per_epoch)
        self.warmup_epochs = int(warmup_epochs)
        self.warmup_steps = self.warmup_epochs * self.steps_per_epoch
        self.warmup_step = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.cosine = CosineAnnealingLR(
            optimizer,
            T_max=self.epochs,
            eta_min=min(self.base_lrs) * float(eta_min_ratio),
        )
        if self.warmup_steps > 0:
            self._set_warmup_lr(1)

    def _set_warmup_lr(self, numerator: int) -> None:
        factor = min(1.0, numerator / max(self.warmup_steps, 1))
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor

    def step_batch(self) -> None:
        if self.warmup_step < self.warmup_steps:
            self.warmup_step += 1
            self._set_warmup_lr(self.warmup_step + 1)

    def step_epoch(self) -> None:
        self.cosine.step()

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        return {
            "epochs": self.epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "warmup_epochs": self.warmup_epochs,
            "warmup_steps": self.warmup_steps,
            "warmup_step": self.warmup_step,
            "base_lrs": self.base_lrs,
            "cosine": self.cosine.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.warmup_step = int(state["warmup_step"])
        self.base_lrs = list(state["base_lrs"])
        self.cosine.load_state_dict(state["cosine"])
