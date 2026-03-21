"""
AlphaMissense clipped sigmoid cross-entropy.
"""

import math
from typing import Optional
import torch
import torch.nn.functional as F


def _softplus(x: float) -> float:
    """
    Numerically stable softplus: log(1 + exp(x)).

    Uses the identity log(1+exp(x)) = x + log(1+exp(-x)) for large x,
    avoiding overflow when x is very large (e.g. 1e9).
    """
    if x > 30:
        return x  # log(1 + exp(x)) ≈ x for large x
    if x < -30:
        return math.exp(x)  # log(1 + exp(x)) ≈ exp(x) for very negative x
    return math.log(1 + math.exp(x))


def clipped_sigmoid_xent(
    logits:   torch.Tensor,
    labels:   torch.Tensor,
    clip_neg: float = 0.0,
    clip_pos: float = -1.0,
    weights:  Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    AlphaMissense clipped sigmoid cross-entropy (Methods, equation in paper).

    The clipping applies a *floor* on loss for two cases:

    Benign variants (label=0): when logit < clip_neg (= 0 by default), the model
    is confidently predicting benign. AM clips the loss floor to softplus(clip_neg)
    = log(2) ≈ 0.693. This prevents over-rewarding extreme benign confidence and
    reflects that low-frequency variants could still be pathogenic.

    Pathogenic variants (label=1): when logit < clip_pos (= -1 by default), the
    model is very wrong. AM clips the loss ceiling to softplus(-clip_pos) = log(2)
    ≈ 0.693. This reflects that "pathogenic" labels in training are sampled
    unobserved variants — noisy proxies, not curated ground truth.
    """
    prob = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
    loss = -labels * torch.log(prob) - (1 - labels) * torch.log(1 - prob)

    # Floor for easy-correct benign predictions (logit << clip_neg)
    loss = torch.where(
        ((1 - labels) > 0.5) & (logits < clip_neg),
        torch.full_like(loss, _softplus(clip_neg)),
        loss,
    )
    # Floor for very-wrong pathogenic predictions (logit << clip_pos)
    loss = torch.where(
        (labels > 0.5) & (logits < clip_pos),
        torch.full_like(loss, _softplus(-clip_pos)),
        loss,
    )

    if weights is not None:
        loss = loss * weights
    return loss
