#!/usr/bin/env python3
"""
scripts/combine_and_split.py

Combine benign variants + pathogenic proxies, then re-run the
protein-level train/val/test split so that both label classes for
each protein stay in the same split.

Usage:
    python scripts/combine_and_split.py \
        --benign_csv  /path/missense-dataset/benign_all.csv \
        --proxy_csv   /path/missense-dataset/pathogenic_proxies.csv \
        --output_dir  /path/missense-dataset/ \
        --val_frac    0.1 \
        --test_frac   0.1
"""
from data.splits import split_by_protein, save_splits
import sys
import argparse
import logging
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benign_csv", required=True)
    p.add_argument("--proxy_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    log.info("Loading benign variants...")
    benign = pd.read_csv(args.benign_csv)
    benign["label"] = 0
    log.info("  %d benign variants", len(benign))

    log.info("Loading pathogenic proxies...")
    proxies = pd.read_csv(args.proxy_csv)
    proxies["label"] = 1
    log.info("  %d pathogenic proxies", len(proxies))

    # Combine — both classes for each protein must stay together
    combined = pd.concat([benign, proxies], ignore_index=True)
    log.info("Combined: %d total variants", len(combined))

    # Verify class balance
    counts = combined["label"].value_counts()
    log.info(
        "Class balance: %d benign (%.1f%%) / %d proxy (%.1f%%)",
        counts.get(0, 0), 100 * counts.get(0, 0) / len(combined),
        counts.get(1, 0), 100 * counts.get(1, 0) / len(combined),
    )

    # Split at protein level — all variants for a protein (both label=0
    # and label=1) go to the same split, preventing any data leakage
    log.info("Splitting at protein level...")
    train, val, test = split_by_protein(
        combined, args.val_frac, args.test_frac, seed=args.seed
    )

    # Sanity check: both labels present in each split
    for name, df in [("train", train), ("val", val), ("test", test)]:
        n0 = (df["label"] == 0).sum()
        n1 = (df["label"] == 1).sum()
        log.info("  %s: %d benign + %d proxy = %d total", name, n0, n1, len(df))

    save_splits(train, val, test, args.output_dir)
    log.info("Done. Splits saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
