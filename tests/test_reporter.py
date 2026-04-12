"""
Unit tests for evaluation/reporter.py.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

# Prevent invalid inherited backend (e.g., matplotlib_inline) during test collection.
os.environ["MPLBACKEND"] = "Agg"

from evaluation import reporter as reporter_mod
from evaluation.metrics import EvalResult, VariantPredictions


def _preds():
    return VariantPredictions(
        scores=np.array([0.1, 0.8, 0.2, 0.9], dtype=np.float32),
        labels=np.array([0, 1, 0, 1], dtype=np.int32),
        gene_ids=np.array(["G1", "G1", "G2", "G2"]),
        positions=np.array([1, 2, 3, 4], dtype=np.int32),
        source="toy",
    )


def _result():
    return EvalResult(
        auroc=0.9,
        auroc_ci=(0.8, 0.95),
        auprc=0.88,
        brier=0.1,
        ece=0.02,
        gene_bias_auroc=0.7,
        debiased_auroc=0.85,
        gene_auroc_mean=0.82,
        path_thresh_90p=0.75,
        benign_thresh_90p=0.25,
        frac_ambiguous=0.1,
        n_variants=4,
        n_genes=2,
        per_gene=pd.DataFrame([{"gene": "G1", "auroc": 0.9, "n_pos": 1, "n_neg": 1}]),
    )


def test_generate_all_saves_json_without_matplotlib(monkeypatch, tmp_path):
    monkeypatch.setattr(reporter_mod, "HAS_MPL", False)
    rep = reporter_mod.Reporter(output_dir=str(tmp_path))
    rep.generate_all(_result(), _preds(), model_name="ModelX")

    p = tmp_path / "eval_summary.json"
    assert p.exists()
    payload = json.loads(p.read_text())
    assert payload["model"] == "ModelX"
    assert payload["auroc"] == 0.9


def test_generate_all_invokes_plotters_when_matplotlib_available(monkeypatch, tmp_path):
    monkeypatch.setattr(reporter_mod, "HAS_MPL", True)
    rep = reporter_mod.Reporter(output_dir=str(tmp_path))
    calls = {"roc": 0, "pr": 0, "cal": 0, "dist": 0, "bias": 0, "pg": 0}

    monkeypatch.setattr(rep, "plot_roc", lambda *a, **k: calls.__setitem__("roc", calls["roc"] + 1))
    monkeypatch.setattr(rep, "plot_pr", lambda *a, **k: calls.__setitem__("pr", calls["pr"] + 1))
    monkeypatch.setattr(rep, "plot_calibration", lambda *a, **k: calls.__setitem__("cal", calls["cal"] + 1))
    monkeypatch.setattr(rep, "plot_score_dist", lambda *a, **k: calls.__setitem__("dist", calls["dist"] + 1))
    monkeypatch.setattr(rep, "plot_gene_bias", lambda *a, **k: calls.__setitem__("bias", calls["bias"] + 1))
    monkeypatch.setattr(rep, "plot_per_gene", lambda *a, **k: calls.__setitem__("pg", calls["pg"] + 1))

    rep.generate_all(_result(), _preds(), model_name="ModelX", calibrated_probs=np.array([0.1, 0.2]))

    assert calls == {"roc": 1, "pr": 1, "cal": 1, "dist": 1, "bias": 1, "pg": 1}


def test_save_json_contains_expected_keys(tmp_path):
    rep = reporter_mod.Reporter(output_dir=str(tmp_path))
    rep._save_json(_result(), "ModelX")
    payload = json.loads((tmp_path / "eval_summary.json").read_text())
    assert set(
        [
            "model",
            "auroc",
            "auroc_ci",
            "auprc",
            "gene_bias_auroc",
            "debiased_auroc",
            "brier",
            "ece",
            "frac_ambiguous",
            "path_thresh",
            "benign_thresh",
            "n_variants",
            "n_genes",
        ]
    ).issubset(set(payload.keys()))
