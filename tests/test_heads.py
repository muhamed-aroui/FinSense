"""Unit tests for finsense.models.heads."""

import torch

from finsense.models.heads import (
    AttentionHead,
    LinearHead,
    MLP2Head,
    MLP3Head,
    build_head,
)


BATCH = 4
SEQ_LEN = 16
HIDDEN = 768
NUM_CLASSES = 3


def _make_inputs():
    hidden_states = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    attention_mask = torch.ones(BATCH, SEQ_LEN, dtype=torch.long)
    # Mask out last 4 positions as padding
    attention_mask[:, -4:] = 0
    return hidden_states, attention_mask


def test_linear_head_shape():
    head = LinearHead(HIDDEN, NUM_CLASSES)
    h, m = _make_inputs()
    out = head(h, m)
    assert out.shape == (BATCH, NUM_CLASSES)


def test_mlp2_head_shape():
    head = MLP2Head(HIDDEN, hidden_dim=256, num_classes=NUM_CLASSES)
    h, m = _make_inputs()
    out = head(h, m)
    assert out.shape == (BATCH, NUM_CLASSES)


def test_mlp3_head_shape():
    head = MLP3Head(HIDDEN, hidden_dims=(512, 128), num_classes=NUM_CLASSES)
    h, m = _make_inputs()
    out = head(h, m)
    assert out.shape == (BATCH, NUM_CLASSES)


def test_attn_head_shape():
    head = AttentionHead(HIDDEN, NUM_CLASSES)
    h, m = _make_inputs()
    out = head(h, m)
    assert out.shape == (BATCH, NUM_CLASSES)


def test_attn_head_respects_mask():
    """Attention head should not attend to masked positions."""
    head = AttentionHead(HIDDEN, NUM_CLASSES)
    h, m = _make_inputs()
    # With full mask vs partial mask, outputs should differ
    m_full = torch.ones_like(m)
    out_partial = head(h, m)
    out_full = head(h, m_full)
    assert not torch.allclose(out_partial, out_full), "Mask should affect output"


def test_linear_head_param_count():
    head = LinearHead(HIDDEN, NUM_CLASSES)
    n = sum(p.numel() for p in head.parameters())
    # 768 * 3 + 3 = 2307
    assert n == 768 * 3 + 3


def test_mlp2_param_count():
    head = MLP2Head(HIDDEN, hidden_dim=256, num_classes=NUM_CLASSES)
    n = sum(p.numel() for p in head.parameters())
    expected = (768 * 256 + 256) + (256 * 3 + 3)
    assert n == expected


def test_build_head_registry():
    for name in ("linear", "mlp2", "mlp3", "attn"):
        head = build_head(name, HIDDEN)
        h, m = _make_inputs()
        out = head(h, m)
        assert out.shape == (BATCH, NUM_CLASSES)


def test_build_head_with_config():
    head = build_head("mlp2", HIDDEN, {"hidden_dim": 128, "dropout": 0.2})
    h, m = _make_inputs()
    out = head(h, m)
    assert out.shape == (BATCH, NUM_CLASSES)
    n = sum(p.numel() for p in head.parameters())
    expected = (768 * 128 + 128) + (128 * 3 + 3)
    assert n == expected


def test_build_head_unknown():
    import pytest

    with pytest.raises(ValueError, match="Unknown head"):
        build_head("nonexistent", HIDDEN)


def test_gradient_flows_through_head():
    """Gradients should flow through all head params."""
    head = MLP2Head(HIDDEN, hidden_dim=256, num_classes=NUM_CLASSES)
    h, m = _make_inputs()
    h.requires_grad_(False)  # simulate frozen backbone
    out = head(h, m)
    loss = out.sum()
    loss.backward()
    for name, param in head.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"
