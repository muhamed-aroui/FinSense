"""Classification head architectures for the frozen-backbone sweep.

All heads take (hidden_states, attention_mask) and return logits of
shape ``(batch, num_classes)``.

Variants:
- LinearHead: single linear layer on [CLS]
- MLP2Head: 2-layer MLP on [CLS]
- MLP3Head: 3-layer MLP on [CLS]
- AttentionHead: single-head self-attention pooling + linear
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearHead(nn.Module):
    """Linear probe: ``Linear(hidden_size, num_classes)`` on [CLS]."""

    def __init__(self, hidden_size: int, num_classes: int = 3) -> None:
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        cls_repr = hidden_states[:, 0]  # [CLS] is position 0
        return self.classifier(cls_repr)


class MLP2Head(nn.Module):
    """2-layer MLP on [CLS]: Linear → ReLU → Dropout → Linear."""

    def __init__(
        self,
        hidden_size: int,
        hidden_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        cls_repr = hidden_states[:, 0]
        return self.net(cls_repr)


class MLP3Head(nn.Module):
    """3-layer MLP on [CLS]: Linear → ReLU → Drop → Linear → ReLU → Drop → Linear."""

    def __init__(
        self,
        hidden_size: int,
        hidden_dims: tuple[int, int] = (512, 128),
        num_classes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        d1, d2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(hidden_size, d1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d1, d2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d2, num_classes),
        )

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        cls_repr = hidden_states[:, 0]
        return self.net(cls_repr)


class AttentionHead(nn.Module):
    """Single-head self-attention pooling over all tokens + linear classifier.

    A learnable query vector attends over the token embeddings via
    scaled dot-product attention (masking out padding), producing a
    weighted-average representation that is fed to a linear classifier.
    """

    def __init__(self, hidden_size: int, num_classes: int = 3) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.scale = math.sqrt(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # hidden_states: (B, S, H), attention_mask: (B, S)
        batch_size = hidden_states.size(0)
        query = self.query.expand(batch_size, -1, -1)  # (B, 1, H)

        # Scaled dot-product: (B, 1, S)
        scores = torch.bmm(query, hidden_states.transpose(1, 2)) / self.scale

        # Mask padding positions to -inf
        mask = attention_mask.unsqueeze(1)  # (B, 1, S)
        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)  # (B, 1, S)
        pooled = torch.bmm(weights, hidden_states).squeeze(1)  # (B, H)

        return self.classifier(pooled)


# Registry for config-driven head construction
HEAD_REGISTRY: dict[str, type[nn.Module]] = {
    "linear": LinearHead,
    "mlp2": MLP2Head,
    "mlp3": MLP3Head,
    "attn": AttentionHead,
}


def build_head(head_name: str, hidden_size: int, head_config: dict | None = None) -> nn.Module:
    """Construct a head by name with optional config overrides.

    Parameters
    ----------
    head_name:
        Key in :data:`HEAD_REGISTRY`.
    hidden_size:
        The backbone's hidden dimension (e.g. 768 for DeBERTa-v3-base).
    head_config:
        Extra kwargs forwarded to the head constructor (e.g. ``hidden_dim``,
        ``hidden_dims``, ``dropout``).
    """
    if head_name not in HEAD_REGISTRY:
        raise ValueError(f"Unknown head {head_name!r}. Choose from {list(HEAD_REGISTRY)}")
    cls = HEAD_REGISTRY[head_name]
    kwargs: dict = {"hidden_size": hidden_size}
    if head_config:
        kwargs.update(head_config)
    return cls(**kwargs)
