"""
ESM-1b encoder wrapper.
"""

from __future__ import annotations
from typing import Literal, Optional
import torch
import torch.nn as nn
from transformers import EsmModel

ESM_DIM = 1280   # ESM-1b hidden dimension


class ESMBackbone(nn.Module):
    """
    Wraps ESM-1b and returns a fixed-size embedding per variant.

    freeze_layers: int
        Number of ESM transformer layers (out of 33) to freeze.
        33 = fully frozen (linear probe mode).
        30 = freeze bottom 30, train top 3 (recommended starting point).
        0  = fully unfrozen (expensive, risk of catastrophic forgetting).

    pooling: "variant_pos" | "mean" | "cls"
        How to reduce (batch, seq_len, 1280) → (batch, 1280).
        "variant_pos" is most principled for single-variant effect prediction:
        it reads the embedding at the exact mutated position, capturing both
        the local residue context and the global sequence context injected by
        ESM's self-attention.
    """

    def __init__(
        self,
        model_name:    str   = "facebook/esm1b_t33_650M_UR50S",
        freeze_layers: int   = 30,
        pooling:       Literal["variant_pos", "mean", "cls"] = "variant_pos",
    ):
        super().__init__()
        self.pooling = pooling
        self.esm = EsmModel.from_pretrained(model_name, add_pooling_layer=False)
        self._freeze(freeze_layers)

    def _freeze(self, n: int):
        for p in self.esm.embeddings.parameters():
            p.requires_grad = False
        for i, layer in enumerate(self.esm.encoder.layer):
            if i < n:
                for p in layer.parameters():
                    p.requires_grad = False

    def forward(
        self,
        input_ids:        torch.Tensor,   # (B, L)
        attention_mask:   torch.Tensor,   # (B, L)
        variant_positions: Optional[torch.Tensor] = None,  # (B,)
    ) -> torch.Tensor:                    # (B, 1280)
        hidden = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state               # (B, L, 1280)

        if self.pooling == "variant_pos":
            assert variant_positions is not None
            idx = variant_positions.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
            return hidden.gather(1, idx).squeeze(1)

        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        if self.pooling == "cls":
            return hidden[:, 0, :]

        raise ValueError(f"Unknown pooling: {self.pooling}")
    