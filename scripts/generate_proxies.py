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
    Generate one pathogenic proxy per benign variant.

    For each benign (protein_id, position, ref_aa, alt_aa_observed):
      1. Build pool = all AAs except ref_aa and any alt_aa already
         observed at this (protein_id, position) in the population.
      2. If pool is empty, skip (all substitutions observed — very rare).
      3. Sample one AA uniformly at random from pool.
      4. Emit as label=1, inheriting the benign variant's MAF weight.
    """
    rng = np.random.default_rng(seed)

    # Build exclusion set: all observed (protein_id, position, alt_aa) tuples.
    # Any AA in this set cannot be a pathogenic proxy — it has been tolerated
    # in at least one South Asian individual.
    log.info("Building exclusion set from %d benign variants...", len(benign))
    observed = set(zip(benign["protein_id"], benign["position"], benign["alt_aa"]))
    log.info("Exclusion set size: %d unique (protein, position, alt_aa) tuples", len(observed))

    proxies = []
    skipped_empty = 0
    skipped_seq = 0

    for _, row in benign.iterrows():
        ref = row["ref_aa"]
        pid = row["protein_id"]
        pos = int(row["position"])
        seq = str(row.get("sequence", ""))

        # Validate sequence integrity
        if seq and (pos < 1 or pos > len(seq) or seq[pos - 1] != ref):
            skipped_seq += 1
            continue

        # Pool: all AAs that are (a) not the reference and
        # (b) not already observed as benign at this position
        pool = [
            aa for aa in ALL_AAS
            if aa != ref and (pid, pos, aa) not in observed
        ]

        if not pool:
            # Every possible substitution at this position has been observed
            # in the South Asian population — cannot generate a proxy here
            skipped_empty += 1
            continue

        # Uniform random sample from unobserved substitutions
        sampled_alt = pool[rng.integers(0, len(pool))]

        proxies.append({
            "protein_id": pid,
            "sequence": seq,
            "position": pos,
            "ref_aa": ref,
            "alt_aa": sampled_alt,
            "label": 1,
            "weight": float(row["weight"]),   # inherit MAF weight
            "source": "proxy",
            "gene_symbol": row.get("gene_symbol", ""),
            "transcript_id": row.get("transcript_id", ""),
            "mane_select": row.get("mane_select", ""),
            "af": None,
            "ac": None,
            "an": None,
            "ar2": None,
            "is_imputed": False,
            "chrom": row.get("chrom", ""),
            "pos": None,
            "hgvsp": "",
            "existing_var": "",
        })

    proxy_df = pd.DataFrame(proxies)

    total = len(benign) + len(proxy_df)
    log.info("Generated:             %d proxies (label=1)", len(proxy_df))
    log.info("Skipped (pool empty):  %d  (all 19 substitutions observed in SAS)", skipped_empty)
    log.info("Skipped (seq invalid): %d", skipped_seq)
    log.info("Class balance:         %.1f%% benign / %.1f%% proxy",
             100 * len(benign) / total, 100 * len(proxy_df) / total)

    # Sample pool size distribution for diagnostic logging
    sample_pools = []
    for _, row in benign.sample(min(10000, len(benign)), random_state=0).iterrows():
        ref = row["ref_aa"]
        pid = row["protein_id"]
        pos = int(row["position"])
        n = sum(
            1 for aa in ALL_AAS
            if aa != ref and (pid, pos, aa) not in observed
        )
        sample_pools.append(n)
    arr = np.array(sample_pools)
    log.info("Pool size (10k sample): mean=%.1f  median=%.0f  min=%d  max=%d",
             arr.mean(), np.median(arr), arr.min(), arr.max())

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
