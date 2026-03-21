"""
Ref/alt projection and differencing.
"""

import torch
import torch.nn as nn


def _projection(esm_dim: int, proj_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(esm_dim, proj_dim),
        nn.LayerNorm(proj_dim),
        nn.GELU(),
        nn.Dropout(dropout),
    )


class VariantFusion(nn.Module):
    """
    Projects ref and alt embeddings separately then computes their difference.

    Separate projections (not shared weights) are important: the model should
    learn that the ref context and alt context play different roles in
    determining pathogenicity. Sharing weights would force them to be
    treated symmetrically, which they are not biologically.
    """

    def __init__(
        self, esm_dim: int = 1280, proj_dim: int = 512, dropout: float = 0.1
    ):
        super().__init__()
        self.ref_proj = _projection(esm_dim, proj_dim, dropout)
        self.alt_proj = _projection(esm_dim, proj_dim, dropout)

    def forward(
        self, ref_emb: torch.Tensor, alt_emb: torch.Tensor
    ) -> torch.Tensor:
        return self.ref_proj(ref_emb) - self.alt_proj(alt_emb)
