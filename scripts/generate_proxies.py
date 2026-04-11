#!/usr/bin/env python3
"""
scripts/generate_proxies.py

Generate synthetic pathogenic proxy variants following the exact
AlphaMissense method (Cheng et al. 2023, Science — Methods section).

Method:
    For each benign variant at (protein, position, ref_aa, alt_aa_observed),
    sample ONE amino acid substitution at the same position that has NOT
    been observed in the population dataset, and label it pathogenic (1).

    Exclusion set: any substitution already observed in your full SAS
    benign dataset (gnomAD SAS + SG10K + IndiGen + 1000G SAS) is excluded.
    Weight: proxy inherits the MAF weight of its paired benign variant.

Usage:
    python scripts/generate_proxies.py \
        --benign_csv  /path/missense-dataset/benign_all.csv \
        --output_csv  /path/missense-dataset/pathogenic_proxies.csv \
        --seed        42
"""
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

ALL_AAS = list("ACDEFGHIKLMNPQRSTVWY")


def generate_proxies(benign: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Generate pathogenic proxies with strict duplicate prevention.

    For each benign locus (protein_id, position, ref_aa):
      1. Build pool = all AAs except ref_aa and any alt_aa already
         observed at this locus in the benign population.
      2. Validate rows with sequence checks when sequence is present.
      3. Sample proxy alts without replacement within each locus group.
      4. Emit label=1 rows inheriting metadata from matched benign rows.

    Guarantees:
      - no proxy overlaps with benign on (protein_id, position, alt_aa)
      - no proxy duplicates on (protein_id, position, ref_aa, alt_aa)
    """
    rng = np.random.default_rng(seed)

    required = {"protein_id", "position", "ref_aa", "alt_aa", "weight"}
    missing = sorted(required - set(benign.columns))
    if missing:
        raise ValueError(f"Input benign data is missing required columns: {missing}")

    dup_key = ["protein_id", "position", "ref_aa", "alt_aa"]
    locus_key = ["protein_id", "position", "ref_aa"]

    work = benign.copy()
    work["protein_id"] = work["protein_id"].astype(str).str.strip()
    work["position"] = pd.to_numeric(work["position"], errors="coerce")
    work["ref_aa"] = work["ref_aa"].astype(str).str.strip().str.upper()
    work["alt_aa"] = work["alt_aa"].astype(str).str.strip().str.upper()

    valid_aa = set(ALL_AAS)
    invalid_mask = (
        (work["protein_id"] == "")
        | (work["protein_id"].str.lower() == "nan")
        | work["position"].isna()
        | ~work["ref_aa"].isin(valid_aa)
        | ~work["alt_aa"].isin(valid_aa)
        | (work["ref_aa"] == work["alt_aa"])
    )
    n_invalid = int(invalid_mask.sum())
    if n_invalid > 0:
        log.warning("Dropping %d benign rows with invalid position/AA fields", n_invalid)
    work = work.loc[~invalid_mask].copy()
    work["position"] = work["position"].astype(int)

    n_in = len(work)
    work = work.drop_duplicates(subset=dup_key, keep="first").reset_index(drop=True)
    n_input_dups = n_in - len(work)
    if n_input_dups > 0:
        log.info("Removed %d duplicate benign rows on key %s", n_input_dups, dup_key)

    # Build exclusion set: all observed (protein_id, position, alt_aa) tuples.
    # Any AA in this set cannot be a pathogenic proxy — it has been tolerated
    # in at least one South Asian individual.
    log.info("Building exclusion set from %d benign variants...", len(work))
    observed = set(zip(work["protein_id"], work["position"], work["alt_aa"]))
    log.info("Exclusion set size: %d unique (protein, position, alt_aa) tuples", len(observed))

    observed_by_locus = (
        work.groupby(locus_key, sort=False)["alt_aa"]
        .agg(lambda s: set(s.values))
        .to_dict()
    )

    def _clean_value(v, default=""):
        if pd.isna(v):
            return default
        return v

    proxies = []
    skipped_empty = 0
    skipped_seq = 0
    skipped_capacity = 0

    for locus, grp in work.groupby(locus_key, sort=False):
        pid, pos, ref = locus
        alt_seen = observed_by_locus[locus]

        # Pool: all AAs that are (a) not reference and
        # (b) not already observed as benign at this locus.
        pool = [aa for aa in ALL_AAS if aa != ref and aa not in alt_seen]

        if not pool:
            # Every possible substitution at this locus has been observed.
            skipped_empty += len(grp)
            continue

        valid_rows = []
        for row in grp.itertuples(index=False):
            seq = _clean_value(getattr(row, "sequence", ""), "")
            seq = str(seq)
            if seq and (pos < 1 or pos > len(seq) or seq[pos - 1] != ref):
                skipped_seq += 1
                continue
            valid_rows.append(row)

        if not valid_rows:
            continue

        n_draw = min(len(valid_rows), len(pool))
        if n_draw < len(valid_rows):
            # Not enough unique unobserved substitutions to assign one per row.
            skipped_capacity += len(valid_rows) - n_draw

        sampled_alts = rng.choice(pool, size=n_draw, replace=False).tolist()

        for row, sampled_alt in zip(valid_rows[:n_draw], sampled_alts):
            weight = _clean_value(getattr(row, "weight", 1.0), 1.0)
            try:
                weight = float(weight)
            except Exception:
                weight = 1.0

            proxies.append({
                "protein_id": pid,
                "sequence": str(_clean_value(getattr(row, "sequence", ""), "")),
                "position": int(pos),
                "ref_aa": ref,
                "alt_aa": sampled_alt,
                "label": 1,
                "weight": weight,   # inherit MAF weight
                "source": "proxy",
                "gene_symbol": _clean_value(getattr(row, "gene_symbol", ""), ""),
                "transcript_id": _clean_value(getattr(row, "transcript_id", ""), ""),
                "mane_select": _clean_value(getattr(row, "mane_select", ""), ""),
                "af": None,
                "ac": None,
                "an": None,
                "ar2": None,
                "is_imputed": False,
                "chrom": _clean_value(getattr(row, "chrom", ""), ""),
                "pos": None,
                "hgvsp": "",
                "existing_var": "",
            })

    proxy_df = pd.DataFrame(proxies)

    # Hard QC: proxy-proxy duplicates must be zero.
    proxy_dups = int(proxy_df.duplicated(subset=dup_key).sum()) if len(proxy_df) else 0
    if proxy_dups > 0:
        raise AssertionError(
            f"Proxy generation created {proxy_dups} duplicate rows on key {dup_key}."
        )

    # Hard QC: proxy must not overlap with observed benign substitutions.
    overlap = 0
    if len(proxy_df):
        overlap = sum(
            (pid, pos, alt) in observed
            for pid, pos, alt in proxy_df[["protein_id", "position", "alt_aa"]].itertuples(index=False, name=None)
        )
    if overlap > 0:
        raise AssertionError(f"Proxy generation created {overlap} benign-overlapping substitutions.")

    total = len(work) + len(proxy_df)
    log.info("Generated:             %d proxies (label=1)", len(proxy_df))
    log.info("Skipped (pool empty):  %d  (all 19 substitutions observed in SAS)", skipped_empty)
    log.info("Skipped (seq invalid): %d", skipped_seq)
    log.info("Skipped (pool limit):  %d  (not enough unique unobserved substitutions)", skipped_capacity)
    if total > 0:
        log.info("Class balance:         %.1f%% benign / %.1f%% proxy",
                 100 * len(work) / total, 100 * len(proxy_df) / total)
    else:
        log.info("Class balance:         n/a (empty cleaned benign input)")

    # Sample pool size distribution for diagnostic logging
    if len(work) > 0:
        sample = work.sample(min(10000, len(work)), random_state=0)
        sample_pools = []
        for row in sample.itertuples(index=False):
            locus = (row.protein_id, int(row.position), row.ref_aa)
            seen = observed_by_locus.get(locus, set())
            n = sum(1 for aa in ALL_AAS if aa != row.ref_aa and aa not in seen)
            sample_pools.append(n)

        arr = np.array(sample_pools)
        if arr.size > 0:
            log.info("Pool size (10k sample): mean=%.1f  median=%.0f  min=%d  max=%d",
                     arr.mean(), np.median(arr), arr.min(), arr.max())
    else:
        log.info("Pool size diagnostics skipped (empty cleaned benign input)")

    return proxy_df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benign_csv", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    log.info("Loading %s", args.benign_csv)
    benign = pd.read_csv(args.benign_csv)
    log.info("Loaded %d benign variants", len(benign))

    proxy_df = generate_proxies(benign, seed=args.seed)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    proxy_df.to_csv(args.output_csv, index=False)
    log.info("Saved %d proxies → %s", len(proxy_df), args.output_csv)


if __name__ == "__main__":
    main()
