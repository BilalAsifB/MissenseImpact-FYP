"""
Pathogenicity MLP classifier.
"""

import torch
import torch.nn as nn


class PathogenicityHead(nn.Module):
    """MLP on top of the variant diff embedding → scalar pathogenicity logit."""

    def __init__(self, proj_dim: int = 512, hidden_dim: int = 256,
                 dropout: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, diff_emb: torch.Tensor) -> torch.Tensor:
        return self.mlp(diff_emb).squeeze(-1)   # (B,)
