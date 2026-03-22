"""
Unit tests for training/loss.py.
Test the custom loss function used for training, which
is a modified binary cross-entropy with clipping to
prevent extreme penalties for very wrong predictions.
"""

from training.loss import clipped_sigmoid_xent
import sys
import math
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_benign_floor_clips_low_loss():
    """
    Easy CORRECT benign predictions (logit << 0) should have their loss
    raised to the floor value softplus(clip_neg) = log(2) ≈ 0.693.

    logit=-5 means model confidently predicts benign.
    label=0  means truth IS benign — correct prediction.
    Without clipping: loss = -log(1 - sigmoid(-5)) ≈ 0.0067 (very low).
    With clipping:    loss = softplus(0) = log(2) ≈ 0.693 (floored).
    """
    logits = torch.tensor([-5.0, -5.0])
    labels = torch.tensor([0.0, 0.0])
    loss = clipped_sigmoid_xent(logits, labels, clip_neg=0.0)
    expected = math.log(2.0)  # softplus(0) = log(1 + exp(0)) = log(2)
    assert torch.allclose(loss, torch.tensor([expected, expected]), atol=1e-4)


def test_benign_no_clip_when_logit_above_threshold():
    """
    When logit > clip_neg (model predicts pathogenic but truth is benign),
    the clip does NOT apply — loss is the standard high BCE value.
    """
    logits = torch.tensor([5.0, 5.0])
    labels = torch.tensor([0.0, 0.0])
    loss_clipped = clipped_sigmoid_xent(logits, labels, clip_neg=0.0)
    # Compute expected standard BCE manually: -log(1 - sigmoid(5))
    prob = torch.sigmoid(torch.tensor(5.0))
    expected = -(torch.log(1 - prob)).item()
    assert torch.allclose(loss_clipped, torch.tensor([expected, expected]), atol=1e-4)


def test_pathogenic_clipping():
    """
    Very wrong pathogenic predictions (logit << clip_pos) should be floored
    to softplus(-clip_pos) = softplus(1) = log(1 + e) ≈ 1.313.
    """
    logits = torch.tensor([-5.0, -5.0])
    labels = torch.tensor([1.0, 1.0])
    loss = clipped_sigmoid_xent(logits, labels, clip_pos=-1.0)
    expected = math.log(1 + math.exp(1.0))  # softplus(-clip_pos) = softplus(1)
    assert torch.allclose(loss, torch.tensor([expected, expected]), atol=1e-4)


def test_weights_applied():
    logits = torch.tensor([0.0, 0.0])
    labels = torch.tensor([1.0, 1.0])
    weights = torch.tensor([1.0, 0.5])
    loss_w = clipped_sigmoid_xent(logits, labels, weights=weights)
    loss_no = clipped_sigmoid_xent(logits, labels)
    assert loss_w[0] == loss_no[0]
    assert torch.isclose(loss_w[1], loss_no[1] * 0.5)


def test_standard_bce_when_clipping_never_triggers():
    """
    With clip thresholds pushed far negative so clipping never fires,
    loss must match standard BCE exactly.

    The benign clip fires when logit < clip_neg.
    The pathogenic clip fires when logit < clip_pos.
    Setting both to -1e9 means a logit would have to be below -1e9
    to trigger either clip — impossible for normally-distributed logits.
    """
    import torch.nn.functional as F
    torch.manual_seed(0)
    logits = torch.randn(50)
    labels = (torch.rand(50) > 0.5).float()
    # clip_neg=-1e9: benign clip never triggers (no normal logit is below -1e9)
    # clip_pos=-1e9: pathogenic clip never triggers for the same reason
    loss_clip = clipped_sigmoid_xent(logits, labels, clip_neg=-1e9, clip_pos=-1e9)
    loss_bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    assert torch.allclose(loss_clip, loss_bce, atol=1e-5)


def test_no_overflow_with_extreme_clip_values():
    """_softplus must not overflow for extreme clip thresholds."""
    logits = torch.randn(10)
    labels = (torch.rand(10) > 0.5).float()
    # Should not raise OverflowError
    loss = clipped_sigmoid_xent(logits, labels, clip_neg=1e9, clip_pos=-1e9)
    assert torch.all(torch.isfinite(loss))


def test_default_clip_values_match_paper():
    """Default clip_neg=0.0 and clip_pos=-1.0 match the AM paper (Methods)."""
    logits = torch.tensor([-2.0])
    labels = torch.tensor([0.0])
    loss = clipped_sigmoid_xent(logits, labels)
    # logit=-2 < clip_neg=0 → floored to log(2)
    assert torch.isclose(loss, torch.tensor(math.log(2.0)), atol=1e-5)
