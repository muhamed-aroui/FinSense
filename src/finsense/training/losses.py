"""Loss functions for the Phase 1 head sweep.

Provides:
- Weighted cross-entropy via ``build_weighted_ce``
- Focal loss via ``FocalLoss``
- A factory ``build_loss`` that reads a config dict and returns the loss.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for multi-class classification.

    ``FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)``

    Parameters
    ----------
    gamma:
        Focusing parameter. Higher values down-weight easy examples more.
    alpha:
        Per-class weights (length = num_classes). If ``None``, uniform.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Sequence[float] | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        else:
            self.alpha: torch.Tensor | None = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Parameters
        ----------
        logits: (B, C) raw logits.
        targets: (B,) integer class labels.
        """
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        # Gather the probability of the true class
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        p_t = (probs * targets_one_hot).sum(dim=-1)  # (B,)
        log_p_t = (log_probs * targets_one_hot).sum(dim=-1)  # (B,)

        focal_weight = (1.0 - p_t) ** self.gamma  # (B,)

        loss = -focal_weight * log_p_t  # (B,)

        # Apply per-class alpha weights
        if self.alpha is not None:
            alpha_t = self.alpha[targets]  # (B,)
            loss = alpha_t * loss

        return loss.mean()


def compute_class_weights(labels: Sequence[int], num_classes: int = 3) -> list[float]:
    """Compute balanced class weights (inverse frequency, sklearn-style).

    Returns weights such that ``weight[c] = n_samples / (n_classes * count[c])``.
    """
    from collections import Counter

    counts = Counter(labels)
    n = len(labels)
    return [n / (num_classes * counts[c]) for c in range(num_classes)]


def build_weighted_ce(class_weights: list[float]) -> nn.CrossEntropyLoss:
    """Build a weighted cross-entropy loss from per-class weights."""
    return nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))


def build_loss(
    strategy: str,
    class_weights: list[float] | None = None,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """Factory: build a loss module from a strategy name.

    Parameters
    ----------
    strategy:
        One of ``"baseline"``, ``"weighted_ce"``, ``"focal"``, ``"sampler"``.
        The ``"sampler"`` strategy uses plain CE (the resampling is handled
        by the data loader, not the loss).
    class_weights:
        Required for ``"weighted_ce"`` and ``"focal"``.
    focal_gamma:
        Gamma for focal loss.
    """
    if strategy in ("baseline", "sampler"):
        return nn.CrossEntropyLoss()
    elif strategy == "weighted_ce":
        if class_weights is None:
            raise ValueError("class_weights required for weighted_ce")
        return build_weighted_ce(class_weights)
    elif strategy == "focal":
        if class_weights is None:
            raise ValueError("class_weights required for focal")
        return FocalLoss(gamma=focal_gamma, alpha=class_weights)
    else:
        raise ValueError(f"Unknown strategy {strategy!r}")
