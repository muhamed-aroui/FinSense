"""Frozen DeBERTa-v3-base backbone wrapper.

Loads the pretrained model, freezes all parameters, and exposes a
forward method that returns the last hidden state for downstream heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel


class FrozenBackbone(nn.Module):
    """Wraps a HF pretrained encoder with all parameters frozen.

    Parameters
    ----------
    model_name:
        HF model id (e.g. ``"microsoft/deberta-v3-base"``).
    """

    def __init__(self, model_name: str = "microsoft/deberta-v3-base") -> None:
        super().__init__()
        self.encoder: PreTrainedModel = AutoModel.from_pretrained(model_name)
        self.hidden_size: int = self.encoder.config.hidden_size
        # Freeze all backbone parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the full last hidden state ``(batch, seq_len, hidden_size)``.

        Runs in no_grad since the backbone is frozen — saves memory by
        not building the autograd graph for the encoder.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state
