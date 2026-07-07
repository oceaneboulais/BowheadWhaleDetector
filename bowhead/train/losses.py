"""Custom loss functions.

FocalLoss
---------
Lin et al. 2017 (RetinaNet).  Reduces the relative loss for well-classified
examples and focuses training on hard, misclassified ones — ideal for the
class imbalance here (prevalence ≈ 0.11, many trivially-easy negatives).

    FL(p_t) = -α_t · (1 − p_t)^γ · log(p_t)

Parameters
----------
gamma : float
    Focusing parameter.  0 → ordinary cross-entropy.  Typical: 1.5–2.0.
alpha : float | None
    Prior probability of the *positive* class used to scale the loss.
    If None the per-class weights passed as ``weight`` to the constructor
    are used (compatible with class-weighted CE).
reduction : str
    "mean" | "sum" | "none"
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | None = None,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.register_buffer("weight", weight)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (B, C) raw logits
        targets : (B,) int64  OR  (B,) float in [0,1] (soft labels from Mixup)
        """
        num_classes = logits.size(1)

        # --- soft-label branch (Mixup) ---------------------------------- #
        if targets.dtype == torch.float32:
            # targets are soft labels in [0,1]; treat as blend of one-hot rows
            log_p = F.log_softmax(logits, dim=1)          # (B, C)
            p     = log_p.exp()
            # build soft one-hot: col 1 = target, col 0 = 1 - target
            soft = torch.zeros_like(logits)
            soft[:, 1] = targets
            soft[:, 0] = 1.0 - targets

            focal_weight = (1.0 - (p * soft).sum(dim=1, keepdim=True)) ** self.gamma
            loss = -(focal_weight * soft * log_p).sum(dim=1)

            if self.alpha is not None:
                at = self.alpha * soft[:, 1] + (1 - self.alpha) * soft[:, 0]
                loss = at * loss
        # --- hard-label branch ------------------------------------------ #
        else:
            log_p = F.log_softmax(logits, dim=1)          # (B, C)
            p     = log_p.exp()
            p_t   = p.gather(1, targets.unsqueeze(1)).squeeze(1)   # (B,)
            log_p_t = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)

            focal_weight = (1.0 - p_t) ** self.gamma
            loss = -focal_weight * log_p_t

            if self.alpha is not None:
                at = torch.where(targets == 1,
                                 torch.tensor(self.alpha, device=logits.device),
                                 torch.tensor(1.0 - self.alpha, device=logits.device))
                loss = at * loss
            elif self.weight is not None:
                w = self.weight.gather(0, targets)
                loss = w * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
