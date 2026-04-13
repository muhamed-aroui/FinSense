"""Unit tests for finsense.training.losses."""

import torch

from finsense.training.losses import (
    FocalLoss,
    build_loss,
    compute_class_weights,
)


def test_compute_class_weights_balanced():
    labels = [0, 0, 1, 1, 1, 2, 2, 2, 2, 2]  # 2:3:5
    weights = compute_class_weights(labels, num_classes=3)
    assert len(weights) == 3
    # Rarest class (0) should have highest weight
    assert weights[0] > weights[1] > weights[2]
    # Check formula: n / (n_classes * count)
    assert abs(weights[0] - 10 / (3 * 2)) < 1e-6
    assert abs(weights[1] - 10 / (3 * 3)) < 1e-6
    assert abs(weights[2] - 10 / (3 * 5)) < 1e-6


def test_focal_loss_shape():
    fl = FocalLoss(gamma=2.0)
    logits = torch.randn(8, 3)
    targets = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    loss = fl(logits, targets)
    assert loss.shape == ()  # scalar
    assert loss.item() > 0


def test_focal_loss_with_alpha():
    fl = FocalLoss(gamma=2.0, alpha=[2.0, 1.5, 0.5])
    logits = torch.randn(8, 3)
    targets = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    loss = fl(logits, targets)
    assert loss.shape == ()
    assert loss.item() > 0


def test_focal_loss_gradient():
    fl = FocalLoss(gamma=2.0)
    logits = torch.randn(4, 3, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 0])
    loss = fl(logits, targets)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0


def test_focal_gamma_zero_approximates_ce():
    """Focal loss with gamma=0 should approximate standard CE."""
    torch.manual_seed(42)
    logits = torch.randn(32, 3)
    targets = torch.randint(0, 3, (32,))
    fl = FocalLoss(gamma=0.0)
    ce = torch.nn.CrossEntropyLoss()
    focal_val = fl(logits, targets).item()
    ce_val = ce(logits, targets).item()
    assert abs(focal_val - ce_val) < 1e-4


def test_build_loss_baseline():
    loss = build_loss("baseline")
    assert isinstance(loss, torch.nn.CrossEntropyLoss)


def test_build_loss_weighted_ce():
    loss = build_loss("weighted_ce", class_weights=[2.0, 1.5, 0.5])
    assert isinstance(loss, torch.nn.CrossEntropyLoss)


def test_build_loss_focal():
    loss = build_loss("focal", class_weights=[2.0, 1.5, 0.5], focal_gamma=2.0)
    assert isinstance(loss, FocalLoss)


def test_build_loss_sampler():
    loss = build_loss("sampler")
    assert isinstance(loss, torch.nn.CrossEntropyLoss)


def test_build_loss_unknown():
    import pytest

    with pytest.raises(ValueError, match="Unknown strategy"):
        build_loss("unknown_strategy")
