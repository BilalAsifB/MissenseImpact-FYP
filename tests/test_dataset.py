"""
Unit tests for data/dataset.py.
"""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from data import dataset as dataset_mod


class _FakePipeline:
    def __init__(self, tokenizer_name: str, max_length: int):
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length

    def process(self, variant):
        return {
            "protein_id": variant.protein_id,
            "position": variant.position,
            "reference_aa": variant.reference_aa,
            "alternate_aa": variant.alternate_aa,
            "label": variant.label,
            "weight": variant.weight,
            "source": variant.source,
            "ref_input_ids": torch.tensor([[0, 5, 2]], dtype=torch.long),
            "ref_attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            "alt_input_ids": torch.tensor([[0, 6, 2]], dtype=torch.long),
            "alt_attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            "variant_position": 2,
        }


def test_dataset_init_raises_on_missing_columns(tmp_path):
    csv = tmp_path / "bad.csv"
    pd.DataFrame([{"protein_id": "P1", "sequence": "ACD"}]).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="CSV missing columns"):
        dataset_mod.SASVariantDataset(str(csv))


def test_dataset_len_and_getitem_with_optional_columns(monkeypatch, tmp_path):
    monkeypatch.setattr(dataset_mod, "DataPipeline", _FakePipeline)
    csv = tmp_path / "ok.csv"
    pd.DataFrame(
        [
            {
                "protein_id": "P1",
                "sequence": "ACDE",
                "position": 2,
                "ref_aa": "C",
                "alt_aa": "V",
                "label": 1,
                "weight": 0.7,
                "source": "gnomad",
            }
        ]
    ).to_csv(csv, index=False)

    ds = dataset_mod.SASVariantDataset(str(csv), max_length=256, tokenizer_name="stub_tok")
    assert len(ds) == 1
    assert ds.pipeline.max_length == 256
    assert ds.pipeline.tokenizer_name == "stub_tok"

    item = ds[0]
    assert item["protein_id"] == "P1"
    assert item["position"] == 2
    assert item["reference_aa"] == "C"
    assert item["alternate_aa"] == "V"
    assert item["label"] == 1
    assert item["weight"] == 0.7
    assert item["source"] == "gnomad"


def test_dataset_getitem_uses_default_weight_and_source(monkeypatch, tmp_path):
    monkeypatch.setattr(dataset_mod, "DataPipeline", _FakePipeline)
    csv = tmp_path / "ok_default.csv"
    pd.DataFrame(
        [
            {
                "protein_id": "P2",
                "sequence": "ACDE",
                "position": 1,
                "ref_aa": "A",
                "alt_aa": "G",
                "label": 0,
            }
        ]
    ).to_csv(csv, index=False)

    ds = dataset_mod.SASVariantDataset(str(csv))
    item = ds[0]
    assert item["weight"] == 1.0
    assert item["source"] == ""


def test_collate_variants_pads_and_stacks_consistently():
    batch = [
        {
            "ref_input_ids": torch.tensor([[0, 5, 2]], dtype=torch.long),
            "alt_input_ids": torch.tensor([[0, 7, 2]], dtype=torch.long),
            "label": 0,
            "weight": 1.0,
            "variant_position": 2,
        },
        {
            "ref_input_ids": torch.tensor([[0, 8, 9, 2]], dtype=torch.long),
            "alt_input_ids": torch.tensor([[0, 8, 4, 2]], dtype=torch.long),
            "label": 1,
            "weight": 0.5,
            "variant_position": 3,
        },
    ]

    out = dataset_mod.collate_variants(batch)

    assert out["ref_input_ids"].shape == (2, 4)
    assert out["alt_input_ids"].shape == (2, 4)
    assert out["ref_attention_mask"].shape == (2, 4)
    assert out["alt_attention_mask"].shape == (2, 4)

    # First sample was length 3, so it should be padded with ESM pad token=1.
    assert out["ref_input_ids"][0, 3].item() == 1
    assert out["alt_input_ids"][0, 3].item() == 1
    assert out["ref_attention_mask"][0].tolist() == [1, 1, 1, 0]
    assert out["alt_attention_mask"][0].tolist() == [1, 1, 1, 0]

    assert out["labels"].dtype == torch.float32
    assert out["weights"].dtype == torch.float32
    assert out["variant_position"].dtype == torch.long
    assert out["labels"].tolist() == [0.0, 1.0]
    assert out["weights"].tolist() == [1.0, 0.5]
