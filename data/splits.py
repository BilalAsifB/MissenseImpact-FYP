"""
Position-aware train/val/test splitting.

The AM paper removes all training variants whose protein position appears
in any validation or test set (Supplementary Note, leakage type 2).
This module enforces that constraint so evaluation is clean.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np


def split_by_protein(
    df: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split at protein level: all variants of a protein go to the same split.
    This prevents any sequence-level data leakage between splits.

    Returns (train_df, val_df, test_df).
    """
    rng = np.random.default_rng(seed)
    proteins = np.asarray(df["protein_id"].unique(), dtype=object)
    rng.shuffle(proteins)

    n_val = max(1, int(len(proteins) * val_frac))
    n_test = max(1, int(len(proteins) * test_frac))

    val_proteins = set(proteins[:n_val])
    test_proteins = set(proteins[n_val:n_val + n_test])
    train_proteins = set(proteins[n_val + n_test:])

    train = df[df["protein_id"].isin(train_proteins)].copy()
    val = df[df["protein_id"].isin(val_proteins)].copy()
    test = df[df["protein_id"].isin(test_proteins)].copy()

    # Enforce position decontamination: remove training variants
    # at positions that appear in val or test for the same protein
    eval_positions = set(
        zip(pd.concat([val, test])["protein_id"],
            pd.concat([val, test])["position"])
    )
    contaminated = train.apply(
        lambda r: (r["protein_id"], r["position"]) in eval_positions, axis=1)
    n_removed = contaminated.sum()
    train = train[~contaminated].reset_index(drop=True)

    print("Split summary:")
    print(f"  Train: {len(train):,} variants ({len(train_proteins):,} proteins)")
    print(f"  Val:   {len(val):,}   variants ({len(val_proteins):,} proteins)")
    print(f"  Test:  {len(test):,}  variants ({len(test_proteins):,} proteins)")
    print(f"  Removed {n_removed:,} training variants at eval positions")

    return train, val, test


def save_splits(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
    output_dir: str,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    val.to_csv(out / "val.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    print(f"Splits saved to {out}/")
