"""
PyTorch Dataset + collate function.
"""

from __future__ import annotations
import torch
import pandas as pd
from torch.utils.data import Dataset
from data.pipeline import DataPipeline, ProteinVariant, ESM_MODEL_NAME


class SASVariantDataset(Dataset):
    """
    Loads the training CSV produced by scripts/build_training_data.py.

    Required columns:  protein_id, sequence, position, reference_aa,
                       alternate_aa, label
    Optional columns:  weight, source
    """

    REQUIRED = {"protein_id", "sequence", "position",
                "ref_aa", "alt_aa", "label"}

    def __init__(self, csv_path: str, max_length: int = 1024,
                 tokenizer_name: str = ESM_MODEL_NAME):
        self.df = pd.read_csv(csv_path)
        missing = self.REQUIRED - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")
        self.pipeline = DataPipeline(tokenizer_name, max_length)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        variant = ProteinVariant(
            protein_id=str(row["protein_id"]),
            sequence=str(row["sequence"]),
            position=int(row["position"]),
            reference_aa=str(row["ref_aa"]),
            alternate_aa=str(row["alt_aa"]),
            label=int(row["label"]),
            weight=float(row.get("weight", 1.0)),
            source=str(row.get("source", "")),
        )
        return self.pipeline.process(variant)


def collate_variants(batch: list[dict]) -> dict:
    """Pad variable-length sequences to longest in batch."""
    def pad(tensors: list) -> tuple[torch.Tensor, torch.Tensor]:
        seqs = [t.squeeze(0) for t in tensors]
        max_len = max(s.size(0) for s in seqs)
        pad_id = 1   # ESM pad token
        ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
        mask = torch.zeros(len(seqs), max_len, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :s.size(0)] = s
            mask[i, :s.size(0)] = 1
        return ids, mask

    ref_ids, ref_mask = pad([b["ref_input_ids"] for b in batch])
    alt_ids, alt_mask = pad([b["alt_input_ids"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
    weights = torch.tensor([b["weight"] for b in batch], dtype=torch.float32)
    var_pos = torch.tensor([b["variant_position"] for b in batch], dtype=torch.long)

    return {
        "ref_input_ids": ref_ids,
        "ref_attention_mask": ref_mask,
        "alt_input_ids": alt_ids,
        "alt_attention_mask": alt_mask,
        "variant_position": var_pos,
        "labels": labels,
        "weights": weights,
    }
