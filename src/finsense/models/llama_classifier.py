"""Llama-based sequence classifier with last-token pooling.

Wraps a causal LM (Llama-3.2-1B) for 3-class classification:
1. Extracts hidden states from the last non-padding token.
2. Passes through a trainable linear classifier.

Supports both standard (fp16) and quantized (4-bit via bitsandbytes)
base models, with LoRA/QLoRA adapters attached via PEFT.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class LlamaClassifier(nn.Module):
    """Causal LM + last-token pooling + linear classifier.

    Parameters
    ----------
    model_name:
        HF model id (e.g. ``"meta-llama/Llama-3.2-1B"``).
    num_classes:
        Number of output classes.
    quantization_config:
        If provided, passed to ``from_pretrained`` for 4-bit loading.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B",
        num_classes: int = 3,
        quantization_config: Any | None = None,
    ) -> None:
        super().__init__()

        load_kwargs: dict[str, Any] = {}
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"

        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=True,
            **load_kwargs,
        )
        self.hidden_size: int = self.backbone.config.hidden_size
        self.classifier = nn.Linear(self.hidden_size, num_classes)

        # Freeze the backbone — LoRA adapters (applied externally) remain trainable
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: backbone → last-token pooling → classifier.

        Returns logits of shape ``(batch, num_classes)``.
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Last layer hidden states: (batch, seq_len, hidden_size)
        hidden_states = outputs.hidden_states[-1]

        # Last-token pooling: find the last non-padding position per example
        # attention_mask is 1 for real tokens, 0 for padding
        # Sum along seq dim gives the length; subtract 1 for 0-based index
        seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_token_repr = hidden_states[batch_idx, seq_lengths]  # (batch, hidden_size)

        return self.classifier(last_token_repr)


def build_llama_classifier(
    cfg: dict,
) -> LlamaClassifier:
    """Build a LlamaClassifier from a config dict.

    Handles 4-bit quantization config for QLoRA if ``cfg["quantization"]``
    is present.
    """
    quant_config = None
    if "quantization" in cfg:
        from transformers import BitsAndBytesConfig

        qcfg = cfg["quantization"]
        quant_config = BitsAndBytesConfig(
            load_in_4bit=qcfg.get("load_in_4bit", True),
            bnb_4bit_quant_type=qcfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=getattr(
                torch, qcfg.get("bnb_4bit_compute_dtype", "bfloat16")
            ),
        )

    return LlamaClassifier(
        model_name=cfg["model"],
        num_classes=3,
        quantization_config=quant_config,
    )


def apply_lora(model: LlamaClassifier, cfg: dict) -> LlamaClassifier:
    """Apply LoRA adapters to the backbone via PEFT.

    Modifies the model in-place and returns it. After this call, only
    the LoRA adapter parameters and the classifier head are trainable.
    """
    from peft import LoraConfig, get_peft_model

    lora_cfg = cfg["lora_config"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.backbone = get_peft_model(model.backbone, peft_config)
    return model
