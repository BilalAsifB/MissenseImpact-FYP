"""
Unit tests for data/pipeline.py.
Covers ProteinVariant validation, DataPipeline tokenisation,
center-cropping for long sequences, and the alternate_sequence property.
"""

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.pipeline import ProteinVariant, DataPipeline

# 20-residue test sequence — one of each standard amino acid in alphabetical order
# Actual character map (1-based):
#  1=A  2=C  3=D  4=E  5=F  6=G  7=H  8=I  9=K  10=L
# 11=M 12=N 13=P 14=Q 15=R 16=S 17=T 18=V 19=W 20=Y
SEQ = "ACDEFGHIKLMNPQRSTVWY"


def test_variant_creates_correctly():
    v = ProteinVariant("P001", SEQ, 1, "A", "C")
    assert v.protein_id == "P001"
    assert v.position == 1
    assert v.reference_aa == "A"
    assert v.alternate_aa == "C"


def test_alternate_sequence_single_change():
    v = ProteinVariant("P001", SEQ, 1, "A", "G")
    assert v.alternate_sequence[0] == "G"
    assert v.alternate_sequence[1:] == SEQ[1:]


def test_alternate_sequence_middle():
    # pos 10 = 'L', pos 9 = 'K'
    v = ProteinVariant("P001", SEQ, 10, "L", "K")
    assert v.alternate_sequence[9]  == "K"
    assert v.alternate_sequence[:9] == SEQ[:9]
    assert v.alternate_sequence[10:] == SEQ[10:]


def test_alternate_sequence_last():
    v = ProteinVariant("P001", SEQ, 20, "Y", "A")
    assert v.alternate_sequence[-1] == "A"
    assert v.alternate_sequence[:-1] == SEQ[:-1]


def test_position_out_of_range_low():
    with pytest.raises(ValueError, match="out of range"):
        ProteinVariant("P001", SEQ, 0, "A", "C")


def test_position_out_of_range_high():
    with pytest.raises(ValueError, match="out of range"):
        ProteinVariant("P001", SEQ, len(SEQ) + 1, "A", "C")


def test_position_at_last_residue():
    v = ProteinVariant("P001", SEQ, len(SEQ), "Y", "A")
    assert v.alternate_sequence[-1] == "A"


def test_ref_aa_mismatch_raises():
    # pos 1 = 'A', passing 'G' should raise
    with pytest.raises(ValueError):
        ProteinVariant("P001", SEQ, 1, "G", "C")


def test_synonymous_raises():
    with pytest.raises(ValueError, match="synonymous"):
        ProteinVariant("P001", SEQ, 1, "A", "A")


def test_empty_sequence_raises():
    with pytest.raises(ValueError, match="empty"):
        ProteinVariant("P001", "", 1, "A", "C")


def test_label_and_weight_defaults():
    v = ProteinVariant("P001", SEQ, 1, "A", "C")
    assert v.label is None
    assert v.weight == 1.0
    assert v.source == ""


def test_pipeline_output_keys():
    pipeline = DataPipeline()
    # pos 6 = 'G', substitute with 'A'
    v = ProteinVariant("P001", SEQ, 6, "G", "A", label=0)
    sample = pipeline.process(v)
    required = {
        "ref_input_ids", "ref_attention_mask",
        "alt_input_ids", "alt_attention_mask",
        "variant_position", "protein_id", "label", "weight",
    }
    assert required.issubset(sample.keys())


def test_pipeline_variant_position_shifted():
    """variant_position = pos_0 + 1 to account for ESM <cls> token."""
    pipeline = DataPipeline()
    # pos 6 = 'G'
    v = ProteinVariant("P001", SEQ, 6, "G", "A", label=0)
    sample = pipeline.process(v)
    # 1-based position 6 → 0-based index 5 → tokenised position 5+1 = 6
    assert sample["variant_position"] == 6


def test_pipeline_ref_alt_differ():
    pipeline = DataPipeline()
    # pos 3 = 'D', substitute with 'E'
    v = ProteinVariant("P001", SEQ, 3, "D", "E", label=0)
    sample = pipeline.process(v)
    assert not (sample["ref_input_ids"] == sample["alt_input_ids"]).all()


def test_pipeline_sequence_length_within_limit():
    pipeline = DataPipeline(max_length=1024)
    v = ProteinVariant("P001", SEQ, 1, "A", "C", label=1)
    sample = pipeline.process(v)
    assert sample["ref_input_ids"].size(1) <= 1024
    assert sample["alt_input_ids"].size(1) <= 1024


def test_pipeline_crops_long_sequence():
    # Build a 2000-residue sequence with a unique residue at position 1000
    long_seq = "A" * 999 + "G" + "A" * 1000
    pipeline = DataPipeline(max_length=128)
    v = ProteinVariant("P001", long_seq, 1000, "G", "V", label=1)
    sample = pipeline.process(v)
    assert sample["ref_input_ids"].size(1) <= 128


def test_pipeline_cropped_position_in_range():
    long_seq = "A" * 999 + "G" + "A" * 1000
    pipeline = DataPipeline(max_length=128)
    v = ProteinVariant("P001", long_seq, 1000, "G", "V", label=1)
    sample = pipeline.process(v)
    seq_len = sample["ref_input_ids"].size(1)
    assert 1 <= sample["variant_position"] < seq_len


def test_pipeline_preserves_label():
    pipeline = DataPipeline()
    for label in [0, 1]:
        v = ProteinVariant("P001", SEQ, 1, "A", "C", label=label)
        assert pipeline.process(v)["label"] == label


def test_pipeline_preserves_weight():
    pipeline = DataPipeline()
    v = ProteinVariant("P001", SEQ, 1, "A", "C", label=0, weight=0.4)
    assert pipeline.process(v)["weight"] == 0.4
