"""
Codon-level single-nucleotide mutation-rate priors for proxy sampling.

AlphaMissense (Cheng et al. 2023, Methods) resamples pathogenic proxies so that
their trinucleotide mutation-rate spectrum matches that of the benign set. The
goal is to prevent the model from learning "rare substitution = pathogenic"
instead of the actual pathogenicity signal.

Full trinucleotide-context matching requires a reference genome FASTA
(pyfaidx on hg38, etc.) to look up flanking bases for every variant.
The benign CSVs produced by data/post_vep.py do not preserve flanking
context, so we approximate the AM scheme by marginalising over codon
positions instead of trinucleotide contexts.

For a candidate (ref_aa -> alt_aa) substitution we:
  1. Enumerate every codon encoding ref_aa.
  2. For each of the three codon positions, enumerate every single-nucleotide
     substitution and translate the resulting codon.
  3. Keep the substitutions whose translation equals alt_aa.
  4. Weight each kept substitution by a 12-type strand-specific SNV rate and
     average over the ref_aa codon set.

This gives a scalar weight w(ref_aa, alt_aa) that is:
  - zero for AA pairs that cannot be reached via a single SNV (very rare in
    the wild; the AM paper also excludes multi-SNV changes from proxy
    sampling)
  - proportional to the expected population-wide mutation probability of the
    AA change, marginalised over which synonymous codon encodes ref_aa
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable


# Standard genetic code (stop codons mapped to "*").
CODON_TABLE: dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

ALL_AAS: tuple[str, ...] = tuple("ACDEFGHIKLMNPQRSTVWY")
_NT_ALPHABET: tuple[str, ...] = ("A", "C", "G", "T")


# Strand-specific 12-type SNV relative rates.
#
# Approximate marginal rates from Samocha et al. 2014 (Nat Genet) after
# collapsing trinucleotide contexts. Values are relative (they do not need
# to sum to one — only their ratios matter for weighted sampling).
#
# Transitions (Ts) are much more common than transversions (Tv), and the
# CpG-related C>T / G>A substitutions are ~3x more common than non-CpG Ts
# because of spontaneous 5mC deamination. Since we do not know the
# trinucleotide context here, we use the overall (CpG-averaged) rate for
# C>T and G>A, which is still meaningfully higher than other Ts.
SNV_RATE_12TYPE: dict[tuple[str, str], float] = {
    ("A", "C"): 0.44,
    ("A", "G"): 1.00,   # Ts
    ("A", "T"): 0.37,
    ("C", "A"): 0.74,
    ("C", "G"): 0.42,
    ("C", "T"): 3.50,   # Ts (CpG-averaged)
    ("G", "A"): 3.50,   # Ts (CpG-averaged; reverse-complement of C>T)
    ("G", "C"): 0.42,
    ("G", "T"): 0.74,
    ("T", "A"): 0.37,
    ("T", "C"): 1.00,   # Ts
    ("T", "G"): 0.44,
}


@lru_cache(maxsize=None)
def _aa_to_codons() -> dict[str, tuple[str, ...]]:
    table: dict[str, list[str]] = {aa: [] for aa in ALL_AAS}
    for codon, aa in CODON_TABLE.items():
        if aa in table:
            table[aa].append(codon)
    return {aa: tuple(sorted(codons)) for aa, codons in table.items()}


def _single_nt_paths(ref_codon: str, alt_aa: str) -> list[tuple[str, str]]:
    """All (ref_nt, alt_nt) single-position substitutions of `ref_codon`
    whose translation is `alt_aa`. Stop codons are excluded."""
    paths: list[tuple[str, str]] = []
    for pos in range(3):
        ref_nt = ref_codon[pos]
        for alt_nt in _NT_ALPHABET:
            if alt_nt == ref_nt:
                continue
            alt_codon = ref_codon[:pos] + alt_nt + ref_codon[pos + 1:]
            if CODON_TABLE[alt_codon] == alt_aa:
                paths.append((ref_nt, alt_nt))
    return paths


@lru_cache(maxsize=None)
def aa_substitution_weight(ref_aa: str, alt_aa: str) -> float:
    """
    Relative population mutation-rate weight for the substitution
    `ref_aa -> alt_aa`, marginalised over synonymous codons of `ref_aa`.

    Returns 0.0 when the substitution cannot be produced by a single
    nucleotide change in any codon encoding `ref_aa` (i.e. requires two
    or more simultaneous SNVs, which is astronomically rare in
    population data).
    """
    if ref_aa == alt_aa:
        return 0.0
    if ref_aa not in _aa_to_codons() or alt_aa not in _aa_to_codons():
        return 0.0

    ref_codons = _aa_to_codons()[ref_aa]
    if not ref_codons:
        return 0.0

    total = 0.0
    for ref_codon in ref_codons:
        for ref_nt, alt_nt in _single_nt_paths(ref_codon, alt_aa):
            total += SNV_RATE_12TYPE[(ref_nt, alt_nt)]
    return total / len(ref_codons)


def aa_weights_for_pool(ref_aa: str, pool: Iterable[str]) -> list[float]:
    """Weights for each candidate alt AA in `pool` given a reference AA.

    Preserves the order of `pool`. Entries with weight 0 (no single-nt
    path from ref_aa) remain 0 — callers decide how to handle all-zero
    weight vectors (e.g. fall back to uniform).
    """
    return [aa_substitution_weight(ref_aa, alt) for alt in pool]
