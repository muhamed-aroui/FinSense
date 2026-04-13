"""Weighted random sampler for class-balanced batches."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

from torch.utils.data import WeightedRandomSampler


def build_weighted_sampler(labels: Sequence[int]) -> WeightedRandomSampler:
    """Build a ``WeightedRandomSampler`` with per-example weight = 1/count(label).

    Samples with replacement so that each class appears roughly equally
    often per epoch. The total number of samples per epoch equals
    ``len(labels)`` (same as default).
    """
    counts = Counter(labels)
    per_example_weight = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(
        weights=per_example_weight,
        num_samples=len(labels),
        replacement=True,
    )
