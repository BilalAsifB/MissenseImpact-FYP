"""
Top-level ESM-Missense model.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from model.backbone import ESMBackbone, ESM_DIM
from model.fusion   import VariantFusion
from model.head     import PathogenicityHead
from training.loss  import clipped_sigmoid_xent


class ESMMissense(nn.Module):
    """
    Full ESM-1b based missense variant pathogenicity predictor.

    Forward pass:
        1. Run ESM-1b on reference sequence  → ref_emb  (B, 1280)
        2. Run ESM-1b on alternate sequence  → alt_emb  (B, 1280)
        3. Project both and compute diff     → diff_emb (B, proj_dim)
        4. MLP classifier on diff            → logit    (B,)

    Memory note: running ESM twice doubles GPU memory usage.
    Use the packed forward (option B in comments) if memory constrained.
    """

    def __init__(
        self,
        esm_model_name: str   = "facebook/esm1b_t33_650M_UR50S",
        freeze_esm_layers: int = 30,
        esm_pooling:    str   = "variant_pos",
        proj_dim:       int   = 512,
        hidden_dim:     int   = 256,
        dropout:        float = 0.1,
    ):
        super().__init__()
        self.backbone   = ESMBackbone(esm_model_name, freeze_esm_layers, esm_pooling)
        self.fusion     = VariantFusion(ESM_DIM, proj_dim, dropout)
        self.classifier = PathogenicityHead(proj_dim, hidden_dim, dropout)

    def forward(self, batch: dict) -> dict:
        var_pos = batch["variant_position"]

        ref_emb = self.backbone(batch["ref_input_ids"],
                                batch["ref_attention_mask"], var_pos)
        alt_emb = self.backbone(batch["alt_input_ids"],
                                batch["alt_attention_mask"], var_pos)

        diff_emb = self.fusion(ref_emb, alt_emb)
        logit    = self.classifier(diff_emb)

        return {
            "logit":         logit,
            "pathogenicity": torch.sigmoid(logit),
            "ref_emb":       ref_emb,
            "alt_emb":       alt_emb,
            "diff_emb":      diff_emb,
        }

    def compute_loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        out     = self.forward(batch)
        loss    = clipped_sigmoid_xent(
            out["logit"], batch["labels"],
            weights=batch.get("weights"),
        ).mean()
        return loss, {"loss": loss.item(),
                      "pathogenicity_mean": out["pathogenicity"].mean().item()}

    def predict(self, batch: dict) -> torch.Tensor:
        """Returns calibrated pathogenicity scores (0–1)."""
        with torch.no_grad():
            return self.forward(batch)["pathogenicity"]
