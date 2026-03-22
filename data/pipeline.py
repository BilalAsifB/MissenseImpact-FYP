"""
ProteinVariant dataclass + ESM-1b tokenisation.
"""

from __future__ import annotations
import dataclasses
from typing import Optional
from transformers import EsmTokenizer

ESM_MODEL_NAME = "facebook/esm1b_t33_650M_UR50S"
ESM_MAX_TOKENS = 1024
ESM_MAX_RESIDUES = ESM_MAX_TOKENS - 2


@dataclasses.dataclass(frozen=True)
class ProteinVariant:
    protein_id: str
    sequence: str
    position: int          # 1-based
    reference_aa: str
    alternate_aa: str
    label: Optional[int] = None
    weight: float = 1.0
    source: str = ""

    def __post_init__(self):
        if not self.sequence:
            raise ValueError("sequence cannot be empty")
        if not (1 <= self.position <= len(self.sequence)):
            raise ValueError(
                f"{self.protein_id}: position {self.position} out of range "
                f"[1, {len(self.sequence)}]")
        if self.sequence[self.position - 1] != self.reference_aa:
            raise ValueError(
                f"{self.protein_id} pos {self.position}: sequence has "
                f"'{self.sequence[self.position - 1]}' but ref='{self.reference_aa}'")
        if self.reference_aa == self.alternate_aa:
            raise ValueError("reference_aa == alternate_aa (synonymous)")

    @property
    def alternate_sequence(self) -> str:
        s = list(self.sequence)
        s[self.position - 1] = self.alternate_aa
        return "".join(s)


class DataPipeline:
    """Tokenises ref + alt sequences for ESM-1b input."""

    def __init__(self, tokenizer_name: str = ESM_MODEL_NAME,
                 max_length: int = ESM_MAX_TOKENS):
        self.tokenizer = EsmTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def process(self, variant: ProteinVariant) -> dict:
        ref_seq, alt_seq = variant.sequence, variant.alternate_sequence
        pos_0 = variant.position - 1

        usable = self.max_length - 2
        if len(ref_seq) > usable:
            half = usable // 2
            start = max(0, pos_0 - half)
            end = min(len(ref_seq), start + usable)
            start = max(0, end - usable)
            ref_seq = ref_seq[start:end]
            alt_seq = alt_seq[start:end]
            pos_0 = pos_0 - start

        def tok(seq):
            return self.tokenizer(seq, return_tensors="pt",
                                  padding=False, truncation=False,
                                  add_special_tokens=True)

        r, a = tok(ref_seq), tok(alt_seq)
        return {
            "ref_input_ids": r["input_ids"],
            "ref_attention_mask": r["attention_mask"],
            "alt_input_ids": a["input_ids"],
            "alt_attention_mask": a["attention_mask"],
            "variant_position": pos_0 + 1,   # +1 for <cls> token
            "protein_id": variant.protein_id,
            "label": variant.label,
            "weight": variant.weight,
            "source": variant.source,
        }
