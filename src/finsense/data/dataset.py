"""Dataset loading, preprocessing, and tokenization for FinSense.

Loads the twitter-financial-news-sentiment dataset from HF Hub,
applies the canonical Phase 0 splits, preprocesses text, and
tokenizes with the DeBERTa-v3-base tokenizer.
"""

from __future__ import annotations

from typing import Any

import torch
from datasets import DatasetDict, load_dataset
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from finsense.data.preprocessing import preprocess
from finsense.data.splits import HF_REPO_ID, load_split_indices


class SentimentDataset(TorchDataset):
    """A map-style PyTorch dataset for sentiment classification.

    Each item returns a dict with ``input_ids``, ``attention_mask``,
    and ``label`` tensors.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 64,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = preprocess(self.texts[idx])
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_splits(
    model_name: str = "microsoft/deberta-v3-base",
    max_length: int = 64,
    split_version: str | None = None,
) -> dict[str, SentimentDataset]:
    """Load train/val/test as :class:`SentimentDataset` instances.

    Parameters
    ----------
    model_name:
        HF model id whose tokenizer to use.
    max_length:
        Maximum token length for padding/truncation.
    split_version:
        Override the split version (default: active version from splits.py).

    Returns
    -------
    Dict with keys ``"train"``, ``"val"``, ``"test"``.
    """
    # Load raw data from HF Hub
    raw: DatasetDict = load_dataset(HF_REPO_ID)  # type: ignore[assignment]

    # Load canonical split indices
    kwargs: dict[str, Any] = {}
    if split_version is not None:
        kwargs["version"] = split_version
    indices = load_split_indices(**kwargs)

    # Apply indices to get the actual rows
    hf_train = raw["train"]
    hf_val = raw["validation"]

    train_texts = [hf_train[i]["text"] for i in indices["train"]]
    train_labels = [hf_train[i]["label"] for i in indices["train"]]

    val_texts = [hf_train[i]["text"] for i in indices["val"]]
    val_labels = [hf_train[i]["label"] for i in indices["val"]]

    test_texts = [hf_val[i]["text"] for i in indices["test"]]
    test_labels = [hf_val[i]["label"] for i in indices["test"]]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return {
        "train": SentimentDataset(train_texts, train_labels, tokenizer, max_length),
        "val": SentimentDataset(val_texts, val_labels, tokenizer, max_length),
        "test": SentimentDataset(test_texts, test_labels, tokenizer, max_length),
    }
