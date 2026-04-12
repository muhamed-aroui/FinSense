"""FinSense canonical train / val / test split definitions.

The Twitter Financial News Sentiment dataset on the HF Hub
(`zeroshot/twitter-financial-news-sentiment`) ships only `train` and
`validation` splits. FinSense renames HF `validation` to `test` and
carves a new stratified validation split out of HF `train` so that
model selection never touches the held-out test set.

The split is materialized as integer index lists persisted to
`splits/phase0_v1.json`. The same indices must be reproduced in any
future Python session that calls :func:`build_phase0_v1`.

This module deliberately depends only on `numpy` + `scikit-learn`
so it can be imported by EDA notebooks before any ML deps are loaded.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.model_selection import train_test_split

#: Canonical Hub repo id for the upstream dataset.
HF_REPO_ID = "zeroshot/twitter-financial-news-sentiment"

#: Project-wide random seed. Do not change without bumping the split
#: file version (`phase0_v1.json` -> `phase0_v2.json`).
SEED = 42

#: Fraction of HF `train` carved into our held-out validation set.
VAL_FRACTION = 0.10

#: Active split file version. Bump (do not overwrite) on any change.
ACTIVE_VERSION = "phase0_v1"

#: Human-readable label mapping. Inferred from dataset inspection
#: (label 0 rows are uniformly bearish: downgrades, cuts to sell).
LABELS: dict[int, str] = {0: "Bearish", 1: "Bullish", 2: "Neutral"}


def build_phase0_v1(hf_train_labels: Sequence[int], hf_val_size: int) -> dict[str, list[int]]:
    """Construct the v1 split index lists deterministically.

    Parameters
    ----------
    hf_train_labels:
        The label column of the upstream HF `train` split, in order.
        Length must equal the size of HF `train`.
    hf_val_size:
        The number of rows in the upstream HF `validation` split.
        These become our held-out `test` set 1:1 in row order.

    Returns
    -------
    A dict with keys ``"train"``, ``"val"``, ``"test"`` mapping to
    lists of integer indices into the respective HF splits. The
    ``"train"`` and ``"val"`` lists index into HF `train`; the
    ``"test"`` list indexes into HF `validation`.
    """
    y = np.asarray(hf_train_labels)
    all_idx = np.arange(len(y))
    train_idx, val_idx = train_test_split(
        all_idx,
        test_size=VAL_FRACTION,
        random_state=SEED,
        stratify=y,
        shuffle=True,
    )
    return {
        "train": sorted(int(i) for i in train_idx),
        "val": sorted(int(i) for i in val_idx),
        "test": list(range(int(hf_val_size))),
    }


def split_file_path(version: str = ACTIVE_VERSION) -> Path:
    """Return the on-disk path of a versioned split file."""
    with resources.as_file(resources.files("finsense.data") / "splits" / f"{version}.json") as p:
        return Path(p)


def load_split_indices(version: str = ACTIVE_VERSION) -> dict[str, list[int]]:
    """Load the persisted split index lists for the given version."""
    path = split_file_path(version)
    with path.open() as fh:
        payload = json.load(fh)
    return {k: list(payload[k]) for k in ("train", "val", "test")}


def save_split_indices(splits: dict[str, list[int]], version: str = ACTIVE_VERSION) -> Path:
    """Persist the split index lists. Refuses to overwrite a different version."""
    path = split_file_path(version)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != splits:
            raise RuntimeError(
                f"Refusing to overwrite {path.name}: contents differ. "
                f"Bump the version (e.g., phase0_v2.json) instead."
            )
    path.write_text(json.dumps(splits, indent=2) + "\n")
    return path
