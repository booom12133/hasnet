"""Loss functions for dual-view multi-label recognition."""

import torch
from torch import nn


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg: float = 4, gamma_pos: float = 0, clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        probs = torch.sigmoid(logits)
        xs_pos = probs
        xs_neg = 1.0 - probs
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)
        loss = targets * torch.log(xs_pos + self.eps) + (1.0 - targets) * torch.log(xs_neg + self.eps)
        pt = xs_pos * targets + xs_neg * (1.0 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
        return -loss * (1.0 - pt).clamp(min=self.eps).pow(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, aux_logits=None, alpha: float = 0.7) -> torch.Tensor:
        main = self._loss(logits, targets)
        if aux_logits is None:
            return main.mean()
        z_ol, z_sd = aux_logits
        aux_weight = (1.0 - alpha) / 2.0
        total = alpha * main + aux_weight * self._loss(z_ol, targets) + aux_weight * self._loss(z_sd, targets)
        return total.mean()
