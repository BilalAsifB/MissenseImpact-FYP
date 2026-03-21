"""
Unit tests for training/loss.py.
Test the custom loss function used for training, which 
is a modified binary cross-entropy with clipping to 
prevent extreme penalties for very wrong predictions.
"""

import sys
from pathlib import Path
import torch
import pytest
sys.path.insert(0, str(Path(__file__).parent.parent))
from training.loss import clipped_sigmoid_xent


def test_benign_clipping():
    """Easy benign predictions (logit >> 0) should be clipped, not penalised."""
    logits = torch.tensor([ 5.0,  5.0])  # very confident benign
    labels = torch.tensor([ 0.0,  0.0])
    loss   = clipped_sigmoid_xent(logits, labels, clip_neg=0.0)
    # All logits > clip_neg=0, so loss = constant log(exp(0)+1) = log(2)
    import math
    expected = math.log(math.exp(0.0) + 1)
    assert torch.allclose(loss, torch.tensor([expected, expected]), atol=1e-4)


def test_pathogenic_clipping():
    """Very wrong pathogenic predictions should be clipped at clip_pos."""
    logits = torch.tensor([-5.0, -5.0])  # predicted benign but label=1
    labels = torch.tensor([ 1.0,  1.0])
    loss   = clipped_sigmoid_xent(logits, labels, clip_pos=-1.0)
    import math
    expected = math.log(math.exp(1.0) + 1)   # log(exp(-(-1))+1)
    assert torch.allclose(loss, torch.tensor([expected, expected]), atol=1e-4)


def test_weights_applied():
    logits  = torch.tensor([0.0, 0.0])
    labels  = torch.tensor([1.0, 1.0])
    weights = torch.tensor([1.0, 0.5])
    loss_w  = clipped_sigmoid_xent(logits, labels, weights=weights)
    loss_no = clipped_sigmoid_xent(logits, labels)
    assert loss_w[0] == loss_no[0]
    assert torch.isclose(loss_w[1], loss_no[1] * 0.5)


def test_standard_bce_range():
    """Without clipping triggers, loss should match standard BCE."""
    import torch.nn.functional as F
    logits = torch.randn(100)
    labels = (torch.rand(100) > 0.5).float()
    # Use extreme clip values so clipping never triggers
    loss_clip = clipped_sigmoid_xent(logits, labels,
                                     clip_neg=-1e9, clip_pos=-1e9)
    loss_bce  = F.binary_cross_entropy_with_logits(logits, labels,
                                                    reduction="none")
    assert torch.allclose(loss_clip, loss_bce, atol=1e-5)
