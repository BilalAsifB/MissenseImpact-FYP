"""
Unit tests for data/splits.py.
"""

from __future__ import annotations

import pandas as pd

from data.splits import save_splits, split_by_protein


def _make_split_df():
    rows = []
    proteins = ["P1", "P2", "P3", "P4", "P5", "P6"]
    for p in proteins:
        rows.append({"protein_id": p, "position": 10, "label": 0, "id": f"{p}_a"})
        rows.append({"protein_id": p, "position": 20, "label": 1, "id": f"{p}_b"})
    return pd.DataFrame(rows)


def test_split_by_protein_separates_proteins_across_splits():
    df = _make_split_df()
    train, val, test = split_by_protein(df, val_frac=0.2, test_frac=0.2, seed=123)

    train_p = set(train["protein_id"])
    val_p = set(val["protein_id"])
    test_p = set(test["protein_id"])

    assert train_p.isdisjoint(val_p)
    assert train_p.isdisjoint(test_p)
    assert val_p.isdisjoint(test_p)
    assert train_p | val_p | test_p == set(df["protein_id"])


def test_split_by_protein_is_reproducible_for_seed():
    df = _make_split_df()
    t1, v1, te1 = split_by_protein(df, val_frac=0.33, test_frac=0.33, seed=42)
    t2, v2, te2 = split_by_protein(df, val_frac=0.33, test_frac=0.33, seed=42)

    assert set(t1["id"]) == set(t2["id"])
    assert set(v1["id"]) == set(v2["id"])
    assert set(te1["id"]) == set(te2["id"])


def test_split_by_protein_removes_train_positions_seen_in_eval():
    # Construct a frame where same protein has repeated positions across rows.
    df = pd.DataFrame(
        [
            {"protein_id": "P1", "position": 10, "id": "p1_10_a"},
            {"protein_id": "P1", "position": 10, "id": "p1_10_b"},
            {"protein_id": "P2", "position": 10, "id": "p2_10_a"},
            {"protein_id": "P3", "position": 30, "id": "p3_30_a"},
            {"protein_id": "P4", "position": 40, "id": "p4_40_a"},
            {"protein_id": "P5", "position": 50, "id": "p5_50_a"},
        ]
    )

    train, val, test = split_by_protein(df, val_frac=0.2, test_frac=0.2, seed=1)
    eval_positions = set(zip(pd.concat([val, test])["protein_id"], pd.concat([val, test])["position"]))
    train_positions = set(zip(train["protein_id"], train["position"]))
    assert train_positions.isdisjoint(eval_positions)


def test_split_by_protein_minimum_one_val_and_test():
    df = pd.DataFrame(
        [
            {"protein_id": "P1", "position": 1},
            {"protein_id": "P2", "position": 2},
            {"protein_id": "P3", "position": 3},
        ]
    )
    train, val, test = split_by_protein(df, val_frac=0.01, test_frac=0.01, seed=7)
    assert len(val["protein_id"].unique()) >= 1
    assert len(test["protein_id"].unique()) >= 1


def test_save_splits_writes_expected_files(tmp_path):
    train = pd.DataFrame([{"x": 1}])
    val = pd.DataFrame([{"x": 2}])
    test = pd.DataFrame([{"x": 3}])

    save_splits(train, val, test, str(tmp_path))

    train_p = tmp_path / "train.csv"
    val_p = tmp_path / "val.csv"
    test_p = tmp_path / "test.csv"

    assert train_p.exists()
    assert val_p.exists()
    assert test_p.exists()
    assert pd.read_csv(train_p)["x"].tolist() == [1]
    assert pd.read_csv(val_p)["x"].tolist() == [2]
    assert pd.read_csv(test_p)["x"].tolist() == [3]
