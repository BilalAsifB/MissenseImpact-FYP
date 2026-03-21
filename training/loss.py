"""
AlphaMissense clipped sigmoid cross-entropy.
"""

import math
from typing import Optional
import torch
import torch.nn.functional as F


def clipped_sigmoid_xent(
    logits:  torch.Tensor,
    labels:  torch.Tensor,
    clip_neg: float = 0.0,
    clip_pos: float = -1.0,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    AlphaMissense loss (Methods, equation in paper).

    Clips benign loss at logit=0 and pathogenic loss at logit=-1.
    This prevents over-penalising uncertain labels (unobserved variants
    sampled as pathogenic proxies may not truly be pathogenic).
    """
    prob = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
    loss = -labels * torch.log(prob) - (1 - labels) * torch.log(1 - prob)

    loss = torch.where(
        ((1 - labels) > 0.5) & (logits < clip_neg),
        torch.full_like(loss, math.log(math.exp(clip_neg) + 1)),
        loss,
    )
    loss = torch.where(
        (labels > 0.5) & (logits < clip_pos),
        torch.full_like(loss, math.log(math.exp(-clip_pos) + 1)),
        loss,
    )
    if weights is not None:
        loss = loss * weights
    return loss
