"""
ProteinVariant dataclass + ESM-1b tokenisation.

Produces the tensors needed by the LM-head scoring head
(model/esm_missense.py): a single reference-sequence tokenisation, the
tokeniser IDs of the reference and alternate amino acids at the variant
site, and the variant position inside the tokenised (cls-offset) sequence.

The alternate-sequence tokens are still emitted so that legacy consumers
and tests keep working, but the new model does not read them.
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
    """Tokenises ref + alt sequences for ESM-1b input.

    The model only needs the reference-sequence tokenisation (it scores the
    variant via the LM head at a masked variant position), but we still
    emit the alternate tokens so legacy consumers and tests keep working.
    """

    def __init__(self, tokenizer_name: str = ESM_MODEL_NAME,
                 max_length: int = ESM_MAX_TOKENS):
        self.tokenizer = EsmTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def _token_id(self, aa: str) -> int:
        """Resolve an amino-acid token to its tokenizer vocabulary ID.

        ESM's tokenizer maps each single-letter AA to exactly one ID; we
        look it up via convert_tokens_to_ids so the mapping stays
        implementation-agnostic.
        """
        tid = self.tokenizer.convert_tokens_to_ids(aa)
        unk_id = getattr(self.tokenizer, "unk_token_id", None)
        if tid is None or (unk_id is not None and tid == unk_id):
            raise ValueError(f"Tokenizer has no dedicated token for AA {aa!r}")
        return int(tid)

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
            "ref_token_id": self._token_id(variant.reference_aa),
            "alt_token_id": self._token_id(variant.alternate_aa),
            "protein_id": variant.protein_id,
            "label": variant.label,
            "weight": variant.weight,
            "source": variant.source,
        }
