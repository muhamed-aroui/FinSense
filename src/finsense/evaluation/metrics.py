"""Metrics computation for FinSense.

Reports the standard metric set from CLAUDE.md §4.5 / reports §6:
- Macro-F1 (north-star)
- Per-class F1 (Bearish, Bullish, Neutral)
- Confusion matrix
- Balanced accuracy
- Inference latency (ms/example)
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import torch
import torch.nn as nn
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)


def compute_metrics(
    labels: Sequence[int],
    preds: Sequence[int],
) -> dict[str, Any]:
    """Compute the full metric set for a set of predictions.

    Returns
    -------
    Dict with keys:
    - ``macro_f1``: float
    - ``per_class_f1``: list of 3 floats [bearish, bullish, neutral]
    - ``balanced_accuracy``: float
    - ``confusion_matrix``: list of lists (3×3)
    """
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    per_class = f1_score(labels, preds, average=None, zero_division=0)
    bal_acc = balanced_accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])

    return {
        "macro_f1": float(macro_f1),
        "per_class_f1": [float(f) for f in per_class],
        "balanced_accuracy": float(bal_acc),
        "confusion_matrix": cm.tolist(),
    }


def measure_latency(
    backbone: nn.Module,
    head: nn.Module,
    dataset: Any,
    device: torch.device,
    dtype: torch.dtype,
    n_samples: int = 100,
    warmup: int = 10,
) -> float:
    """Measure mean per-example inference latency in milliseconds.

    Uses batch_size=1 on ``n_samples`` examples from ``dataset``,
    excluding the first ``warmup`` examples from timing.
    """
    backbone.eval()
    head.eval()

    n_total = min(warmup + n_samples, len(dataset))
    times: list[float] = []

    with torch.no_grad():
        for i in range(n_total):
            item = dataset[i]
            input_ids = item["input_ids"].unsqueeze(0).to(device)
            attention_mask = item["attention_mask"].unsqueeze(0).to(device)

            # Sync CUDA before timing
            if device.type == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()

            with torch.amp.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
                hidden = backbone(input_ids, attention_mask)
                _ = head(hidden, attention_mask)

            if device.type == "cuda":
                torch.cuda.synchronize()

            t1 = time.perf_counter()

            if i >= warmup:
                times.append((t1 - t0) * 1000.0)  # ms

    backbone.train()
    head.train()

    return sum(times) / len(times) if times else 0.0
