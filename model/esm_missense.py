"""
Top-level ESM-Missense model with AlphaMissense-style LM-head scoring.

Scoring
-------
For each variant at (protein, position, ref_aa, alt_aa):

  1. Take the tokenised reference sequence.
  2. Replace the variant position with ESM's ``<mask>`` token.
  3. Run ESM-1b + its pre-trained masked-LM head, producing
     vocabulary logits of shape (B, L, V).
  4. Read ``lm_logits[variant_pos, ref_token]`` and
     ``lm_logits[variant_pos, alt_token]``.
  5. The pathogenicity logit is ``logit_ref - logit_alt``. Under the
     clipped sigmoid cross-entropy loss, a positive value means "the LM
     prefers the reference" (i.e. the alternate is disfavoured /
     pathogenic), matching AlphaMissense's ``logit_diff`` convention.

Auxiliary loss
--------------
During training we additionally mask ``mlm_mask_prob`` of the real
residues (excluding the variant position and ESM special tokens) and
compute a standard masked-LM cross-entropy loss on them. This
regularizes the backbone + LM head back toward the ESM-1b pre-training
manifold — analogous to the masked-MSA loss AlphaMissense retains
during fine-tuning.

Backward compatibility
----------------------
The module returns the same top-level dict signature the previous
MLP-head model produced (``logit``, ``pathogenicity``), plus optional
``mlm_logits`` / ``mlm_labels`` / ``mlm_loss`` used by the trainer when
``mlm_lambda > 0``. Existing callers that only read ``out["logit"]`` or
``out["pathogenicity"]`` work unchanged.
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone import ESMBackbone
from training.loss import clipped_sigmoid_xent

# ESM-1b special-token IDs. These are fixed by the ESM-1b tokenizer and are
# only exposed as constructor arguments for testability.
ESM_PAD_TOKEN_ID = 1
ESM_CLS_TOKEN_ID = 0
ESM_EOS_TOKEN_ID = 2
ESM_MASK_TOKEN_ID = 32


class ESMMissense(nn.Module):
    """
    AlphaMissense-style LM-head-scoring ESM-1b variant classifier.

    Parameters
    ----------
    esm_model_name:
        HuggingFace model ID for the ESM checkpoint.
    freeze_esm_layers:
        Passed to :class:`ESMBackbone`.
    mlm_mask_prob:
        Fraction of non-variant, non-special residues to mask for the
        auxiliary masked-LM loss at training time. ``0.0`` disables the
        aux loss (only the variant position is masked and only the
        pathogenicity logit-diff is produced).
    mlm_max_masks:
        Upper bound on the number of auxiliary MLM positions per sample.
        Protects memory for long sequences.
    pad_token_id / mask_token_id / cls_token_id / eos_token_id:
        Override the ESM-1b defaults; useful for unit tests that wire in
        a small tokenizer.
    """

    def __init__(
        self,
        esm_model_name: str = "facebook/esm1b_t33_650M_UR50S",
        freeze_esm_layers: int = 30,
        mlm_mask_prob: float = 0.15,
        mlm_max_masks: int = 80,
        pad_token_id: int = ESM_PAD_TOKEN_ID,
        mask_token_id: int = ESM_MASK_TOKEN_ID,
        cls_token_id: int = ESM_CLS_TOKEN_ID,
        eos_token_id: int = ESM_EOS_TOKEN_ID,
        # Legacy keyword arguments kept so existing JSON configs and CLIs
        # do not break; they have no effect in the new scoring head.
        esm_pooling: Optional[str] = None,
        proj_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        del esm_pooling, proj_dim, hidden_dim, dropout  # intentionally unused
        self.backbone = ESMBackbone(esm_model_name, freeze_esm_layers)
        self.mlm_mask_prob = float(mlm_mask_prob)
        self.mlm_max_masks = int(mlm_max_masks)
        self.pad_token_id = int(pad_token_id)
        self.mask_token_id = int(mask_token_id)
        self.cls_token_id = int(cls_token_id)
        self.eos_token_id = int(eos_token_id)

    # ------------------------------------------------------------------
    # Masking helpers
    # ------------------------------------------------------------------
    def _build_mlm_labels(
        self,
        input_ids: torch.Tensor,       # (B, L)
        attention_mask: torch.Tensor,  # (B, L)
        variant_position: torch.Tensor,  # (B,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mask a random subset of non-variant, non-special tokens.

        Returns a ``(masked_input_ids, mlm_labels)`` pair; positions that
        should not contribute to the MLM loss are labelled ``-100``.
        """
        B, L = input_ids.shape
        device = input_ids.device
        labels = torch.full_like(input_ids, -100)
        masked_ids = input_ids.clone()

        # Start from "real residue" positions: attended and not a special
        # token and not the variant itself.
        is_real = attention_mask.bool()
        is_real &= input_ids != self.cls_token_id
        is_real &= input_ids != self.eos_token_id
        is_real &= input_ids != self.pad_token_id
        is_real &= input_ids != self.mask_token_id
        arange_b = torch.arange(B, device=device)
        is_real[arange_b, variant_position] = False

        if self.mlm_mask_prob <= 0.0:
            return masked_ids, labels

        rand = torch.rand(B, L, device=device)
        to_mask = is_real & (rand < self.mlm_mask_prob)

        # Cap at ``mlm_max_masks`` per sample to keep memory bounded.
        if self.mlm_max_masks > 0:
            counts = to_mask.sum(dim=1)
            overflow = counts > self.mlm_max_masks
            if overflow.any():
                for i in torch.nonzero(overflow, as_tuple=False).flatten().tolist():
                    idx = torch.nonzero(to_mask[i], as_tuple=False).flatten()
                    keep = idx[torch.randperm(idx.numel(), device=device)[:self.mlm_max_masks]]
                    row_mask = torch.zeros(L, dtype=torch.bool, device=device)
                    row_mask[keep] = True
                    to_mask[i] = row_mask

        labels[to_mask] = input_ids[to_mask]
        masked_ids[to_mask] = self.mask_token_id
        return masked_ids, labels

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, batch: dict) -> dict:
        input_ids = batch["ref_input_ids"]
        attention_mask = batch["ref_attention_mask"]
        variant_position = batch["variant_position"]
        B = input_ids.size(0)
        arange_b = torch.arange(B, device=input_ids.device)

        # Auxiliary MLM masking only at training time.
        want_mlm = self.training and self.mlm_mask_prob > 0.0
        if want_mlm:
            masked_ids, mlm_labels = self._build_mlm_labels(
                input_ids, attention_mask, variant_position
            )
        else:
            masked_ids = input_ids.clone()
            mlm_labels = None

        # Always mask the variant position so the LM-head readout is a
        # proper masked-LM logit (not the log-prob of the unmasked input).
        masked_ids[arange_b, variant_position] = self.mask_token_id

        enc_out = self.backbone(masked_ids, attention_mask)
        lm_logits = enc_out["logits"]   # (B, L, V)

        # Variant-position vocabulary logits → (B, V).
        vp_logits = lm_logits[arange_b, variant_position]

        ref_tok = batch.get("ref_token_id")
        alt_tok = batch.get("alt_token_id")
        if ref_tok is None or alt_tok is None:
            raise KeyError(
                "ESMMissense requires 'ref_token_id' and 'alt_token_id' in the "
                "batch (see data/pipeline.py). Update data pipelines or "
                "collate_fn to pass them through."
            )

        logit = (
            vp_logits.gather(1, ref_tok.view(-1, 1)).squeeze(1)
            - vp_logits.gather(1, alt_tok.view(-1, 1)).squeeze(1)
        )

        out = {
            "logit": logit,
            "pathogenicity": torch.sigmoid(logit),
            "variant_logits": vp_logits,
        }
        if mlm_labels is not None:
            out["mlm_logits"] = lm_logits
            out["mlm_labels"] = mlm_labels
            out["mlm_loss"] = F.cross_entropy(
                lm_logits.view(-1, lm_logits.size(-1)),
                mlm_labels.view(-1),
                ignore_index=-100,
            )
        return out

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def compute_loss(
        self,
        batch: dict,
        mlm_lambda: float = 0.0,
    ) -> tuple[torch.Tensor, dict]:
        """Return the scalar training loss and a metric dict.

        ``mlm_lambda`` weights the auxiliary masked-LM loss. Set to
        ``0.0`` to recover a pure pathogenicity objective; set > 0 to
        also regularize the backbone with a standard masked-LM loss on
        ``mlm_mask_prob`` of the non-variant residues.
        """
        out = self.forward(batch)
        var_loss = clipped_sigmoid_xent(
            out["logit"], batch["labels"], weights=batch.get("weights"),
        ).mean()

        metrics = {
            "loss": var_loss.item(),
            "variant_loss": var_loss.item(),
            "pathogenicity_mean": out["pathogenicity"].mean().item(),
        }
        if mlm_lambda > 0.0 and "mlm_loss" in out:
            mlm_loss = out["mlm_loss"]
            loss = var_loss + mlm_lambda * mlm_loss
            metrics["mlm_loss"] = mlm_loss.item()
            metrics["loss"] = loss.item()
            return loss, metrics

        return var_loss, metrics

    def predict(self, batch: dict) -> torch.Tensor:
        """Returns calibrated pathogenicity scores (0–1)."""
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                return self.forward(batch)["pathogenicity"]
        finally:
            if was_training:
                self.train()
