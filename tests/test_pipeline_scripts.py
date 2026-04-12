"""
Unit tests for scripts orchestrating the data pipeline.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from scripts import build_training_data, combine_and_split


def test_build_training_data_main_orchestrates_steps(monkeypatch, tmp_path):
    args = SimpleNamespace(
        gnomad_dir=str(tmp_path / "gnomad"),
        sg10k_dir=str(tmp_path / "sg10k"),
        indigen_dir=str(tmp_path / "indigen"),
        thousandg_dir=str(tmp_path / "thousandg"),
        output_dir=str(tmp_path / "out"),
        seq_cache=str(tmp_path / "seq.json"),
        gene_cache=str(tmp_path / "gene.json"),
        checkpoint_dir=str(tmp_path / "ckpt"),
        chromosomes=["chr1", "chr2"],
        min_an=150,
        val_frac=0.2,
        test_frac=0.3,
    )
    monkeypatch.setattr(
        build_training_data.argparse.ArgumentParser,
        "parse_args",
        lambda self: args,
    )

    benign_df = pd.DataFrame([{"protein_id": "P1", "position": 10}])
    train_df = pd.DataFrame([{"protein_id": "P1"}])
    val_df = pd.DataFrame([{"protein_id": "P2"}])
    test_df = pd.DataFrame([{"protein_id": "P3"}])
    calls = {"build": None, "split": None, "save": None}

    def _fake_build_training_csv(**kwargs):
        calls["build"] = kwargs
        return benign_df

    def _fake_split_by_protein(df, val_frac, test_frac):
        calls["split"] = (df.copy(), val_frac, test_frac)
        return train_df, val_df, test_df

    def _fake_save_splits(train, val, test, output_dir):
        calls["save"] = (train.copy(), val.copy(), test.copy(), output_dir)

    monkeypatch.setattr(build_training_data, "build_training_csv", _fake_build_training_csv)
    monkeypatch.setattr(build_training_data, "split_by_protein", _fake_split_by_protein)
    monkeypatch.setattr(build_training_data, "save_splits", _fake_save_splits)

    build_training_data.main()

    assert calls["build"]["annotated_dirs"] == {
        "gnomad": args.gnomad_dir,
        "sg10k": args.sg10k_dir,
        "indigen": args.indigen_dir,
        "1000g": args.thousandg_dir,
    }
    assert calls["build"]["output_csv"].endswith("benign_all.csv")
    assert calls["build"]["chromosomes"] == ["chr1", "chr2"]
    assert calls["build"]["min_an"] == 150

    split_df, val_frac, test_frac = calls["split"]
    assert split_df.equals(benign_df)
    assert val_frac == 0.2
    assert test_frac == 0.3

    saved_train, saved_val, saved_test, saved_out_dir = calls["save"]
    assert saved_train.equals(train_df)
    assert saved_val.equals(val_df)
    assert saved_test.equals(test_df)
    assert saved_out_dir == args.output_dir


def test_combine_and_split_main_merges_labels_and_splits(monkeypatch, tmp_path):
    args = SimpleNamespace(
        benign_csv=str(tmp_path / "benign.csv"),
        proxy_csv=str(tmp_path / "proxy.csv"),
        output_dir=str(tmp_path / "out"),
        val_frac=0.1,
        test_frac=0.2,
        seed=77,
    )
    monkeypatch.setattr(
        combine_and_split.argparse.ArgumentParser,
        "parse_args",
        lambda self: args,
    )

    benign = pd.DataFrame([{"protein_id": "P1", "position": 1}])
    proxy = pd.DataFrame([{"protein_id": "P2", "position": 2}])
    calls = {"split_input": None, "save": None}

    def _fake_read_csv(path):
        if path == args.benign_csv:
            return benign.copy()
        if path == args.proxy_csv:
            return proxy.copy()
        raise AssertionError(f"Unexpected path: {path}")

    def _fake_split(df, val_frac, test_frac, seed):
        calls["split_input"] = (df.copy(), val_frac, test_frac, seed)
        return df.iloc[:1].copy(), df.iloc[1:2].copy(), df.iloc[0:0].copy()

    def _fake_save_splits(train, val, test, output_dir):
        calls["save"] = (train.copy(), val.copy(), test.copy(), output_dir)

    monkeypatch.setattr(combine_and_split.pd, "read_csv", _fake_read_csv)
    monkeypatch.setattr(combine_and_split, "split_by_protein", _fake_split)
    monkeypatch.setattr(combine_and_split, "save_splits", _fake_save_splits)

    combine_and_split.main()

    combined, val_frac, test_frac, seed = calls["split_input"]
    assert set(combined["label"]) == {0, 1}
    assert val_frac == 0.1
    assert test_frac == 0.2
    assert seed == 77

    _, _, _, saved_out_dir = calls["save"]
    assert saved_out_dir == args.output_dir
