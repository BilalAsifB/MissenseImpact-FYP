"""
Test the evaluation metrics used to assess model performance,
includingauROC, auPRC, Brier score, ECE, and gene bias.
Also test the calibration fitting and application functions.
"""

import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.metrics import (
    VariantPredictions, evaluate, gene_bias_auroc,
    balance_per_gene, fit_calibration, apply_calibration,
)


def _make_preds(n=200, seed=0):
    rng = np.random.default_rng(seed)
    scores = rng.standard_normal(n).astype(np.float32)
    labels = (rng.random(n) > 0.5).astype(np.int32)
    genes  = rng.choice(["GENE_A","GENE_B","GENE_C","GENE_D"], n)
    pos    = rng.integers(1, 500, n)
    return VariantPredictions(scores, labels, genes, pos)


def test_evaluate_runs():
    preds  = _make_preds()
    result = evaluate(preds, n_bootstrap=10)
    assert 0.0 <= result.auroc  <= 1.0
    assert 0.0 <= result.auprc  <= 1.0
    assert 0.0 <= result.brier  <= 1.0
    assert 0.0 <= result.ece    <= 1.0
    assert 0.0 <= result.frac_ambiguous <= 1.0


def test_gene_bias_auroc_range():
    preds = _make_preds()
    bias  = gene_bias_auroc(preds)
    assert 0.0 <= bias <= 1.0


def test_balance_per_gene():
    preds = _make_preds(400)
    bal   = balance_per_gene(preds)
    df_b  = __import__("pandas").DataFrame({"g": bal.gene_ids, "l": bal.labels})
    for _, g in df_b.groupby("g"):
        assert (g.l==1).sum() == (g.l==0).sum()


def test_calibration_roundtrip():
    rng    = np.random.default_rng(7)
    logits = rng.standard_normal(500)
    labels = (1/(1+np.exp(-logits)) > 0.5).astype(int)
    c1, c0 = fit_calibration(logits, labels)
    probs   = apply_calibration(logits, c1, c0)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_summary_prints():
    preds  = _make_preds()
    result = evaluate(preds, n_bootstrap=5)
    s = result.summary()
    assert "auROC" in s
    assert "Gene-bias" in s
