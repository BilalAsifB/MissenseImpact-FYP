"""
ESM-1b encoder + masked-LM head wrapper.

This module exposes the pre-trained masked-language-model head that ships
with ``EsmForMaskedLM``. The variant-scoring model in
``model/esm_missense.py`` reads vocabulary logits at a masked variant
position (logit_diff scoring, AlphaMissense-style), and the auxiliary
training objective re-uses the same head for a standard masked-LM loss
to keep the backbone close to the pre-trained manifold.
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
from transformers import EsmForMaskedLM

ESM_DIM = 1280   # ESM-1b hidden dimension


class ESMBackbone(nn.Module):
    """
    Wraps ``EsmForMaskedLM`` and returns both the per-token hidden state
    and the per-token vocabulary logits produced by the pre-trained LM
    head. The encoder and LM head are the same objects HuggingFace ships
    under ``EsmForMaskedLM.esm`` and ``EsmForMaskedLM.lm_head``.

    Parameters
    ----------
    model_name:
        HuggingFace model ID for the ESM checkpoint.
    freeze_layers:
        Number of bottom transformer layers (out of 33 for ESM-1b) to
        freeze. Embeddings are always frozen; the LM head is always
        trainable so the auxiliary MLM loss can regularize it alongside
        the variant-position logit-diff.

        - ``33`` = fully frozen encoder (linear probe via the LM head).
        - ``30`` = freeze bottom 30, train top 3 (recommended default).
        - ``0``  = fully unfrozen (expensive, risk of catastrophic forgetting).
    """

    def __init__(
        self,
        model_name: str = "facebook/esm1b_t33_650M_UR50S",
        freeze_layers: int = 30,
    ):
        super().__init__()
        mlm = EsmForMaskedLM.from_pretrained(model_name)
        self.esm = mlm.esm
        self.lm_head = mlm.lm_head
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
        input_ids: torch.Tensor,        # (B, L)
        attention_mask: torch.Tensor,   # (B, L)
        return_hidden: bool = False,
    ) -> dict:
        """Return per-token vocabulary logits and (optionally) hidden states.

        Output keys:
          - ``logits``: (B, L, V) — unnormalised LM-head logits.
          - ``hidden``: (B, L, D) — last hidden state (only if ``return_hidden``).
        """
        enc_out = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = enc_out.last_hidden_state
        logits = self.lm_head(hidden)
        out = {"logits": logits}
        if return_hidden:
            out["hidden"] = hidden
        return out

    # Kept for external callers that only want hidden states. Not used by
    # the current pathogenicity scorer but safe to import.
    def hidden_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        variant_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        if variant_positions is None:
            return hidden
        idx = variant_positions.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
        return hidden.gather(1, idx).squeeze(1)
