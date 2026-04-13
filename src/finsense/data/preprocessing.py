"""Text preprocessing for FinSense.

Phase 1 preprocessing (pinned, no ablation):
1. Strip URLs (https?://... short-links carry no semantic content).
2. Collapse runs of whitespace to a single space.
3. No lowercasing — DeBERTa-v3 is cased.
4. Cashtags and hashtags kept as-is.
"""

from __future__ import annotations

import re

_URL_RE = re.compile(r"https?://\S+")
_WS_RE = re.compile(r"\s+")


def preprocess(text: str) -> str:
    """Apply the Phase 1 preprocessing pipeline to a single text."""
    text = _URL_RE.sub("", text)
    text = _WS_RE.sub(" ", text).strip()
    return text
