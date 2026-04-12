"""
Unit tests for evaluation/benchmark.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from evaluation import benchmark as bm
from evaluation.metrics import EvalResult


class _FakePipeline:
    def process(self, variant):
        ids = torch.tensor([[0, 5, 2]], dtype=torch.long)
        mask = torch.tensor([[1, 1, 1]], dtype=torch.long)
        return {
            "ref_input_ids": ids,
            "ref_attention_mask": mask,
            "alt_input_ids": ids.clone(),
            "alt_attention_mask": mask.clone(),
            "variant_position": 1,
            "label": variant.label,
            "weight": 1.0,
        }


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return super().eval()

    def forward(self, batch):
        n = batch["labels"].shape[0]
        logits = torch.arange(1, n + 1, dtype=torch.float32, device=batch["labels"].device)
        return {"logit": logits}


def _benchmark_df():
    return pd.DataFrame(
        [
            {"protein_id": "P1", "sequence": "ACD", "position": 1, "reference_aa": "A", "alternate_aa": "V", "label": 0},
            # Invalid ref aa -> ProteinVariant validation error -> NaN in inference.
            {"protein_id": "P2", "sequence": "ACD", "position": 1, "reference_aa": "G", "alternate_aa": "V", "label": 1},
            {"protein_id": "P3", "sequence": "TTA", "position": 2, "reference_aa": "T", "alternate_aa": "A", "label": 1},
        ]
    )


def test_run_inference_marks_invalid_rows_as_nan():
    model = _FakeModel()
    pipeline = _FakePipeline()
    df = _benchmark_df()

    logits = bm.run_inference(model, df, pipeline, device="cpu", batch_size=2)

    assert model.eval_called
    assert len(logits) == 3
    assert np.isfinite(logits[0])
    assert np.isnan(logits[1])
    assert np.isfinite(logits[2])


def test_run_inference_all_invalid_chunk_returns_all_nan():
    model = _FakeModel()
    pipeline = _FakePipeline()
    df = pd.DataFrame(
        [
            {"protein_id": "P1", "sequence": "ACD", "position": 1, "reference_aa": "G", "alternate_aa": "V", "label": 0},
            {"protein_id": "P2", "sequence": "ACD", "position": 1, "reference_aa": "G", "alternate_aa": "V", "label": 1},
        ]
    )

    logits = bm.run_inference(model, df, pipeline, device="cpu", batch_size=8)
    assert np.isnan(logits).all()


def test_benchmark_suite_calibrate_uses_non_nan_logits(monkeypatch, tmp_path):
    suite = bm.BenchmarkSuite(data_dir=str(tmp_path), n_bootstrap=5)
    val_df = pd.DataFrame({"label": [0, 1, 0]})
    fit_args = {}

    monkeypatch.setattr(bm.pd, "read_csv", lambda _p: val_df)
    monkeypatch.setattr(bm, "run_inference", lambda *args, **kwargs: np.array([0.1, np.nan, 0.3]))

    def _fake_fit(logits, labels):
        fit_args["logits"] = logits
        fit_args["labels"] = labels
        return 1.23, -0.45

    monkeypatch.setattr(bm, "fit_calibration", _fake_fit)

    suite.calibrate(model=None, pipeline=None, val_csv="val.csv", device="cpu", batch_size=2)

    assert suite._cal_c1 == 1.23
    assert suite._cal_c0 == -0.45
    assert np.allclose(fit_args["logits"], np.array([0.1, 0.3]))
    assert np.array_equal(fit_args["labels"], np.array([0, 0]))


def test_run_one_returns_none_for_missing_csv(tmp_path):
    suite = bm.BenchmarkSuite(data_dir=str(tmp_path), n_bootstrap=3)
    out = suite.run_one(model=None, pipeline=None, name="missing_benchmark", device="cpu")
    assert out is None


def test_run_one_applies_calibration_and_filters_nan(monkeypatch, tmp_path):
    df = pd.DataFrame(
        [
            {"protein_id": "P1", "sequence": "ACD", "position": 1, "reference_aa": "A", "alternate_aa": "V", "label": 0},
            {"protein_id": "P2", "sequence": "ACD", "position": 1, "reference_aa": "A", "alternate_aa": "V", "label": 1},
            {"protein_id": "P3", "sequence": "TTA", "position": 2, "reference_aa": "T", "alternate_aa": "A", "label": 1},
        ]
    )
    (tmp_path / "toy.csv").write_text(df.to_csv(index=False))

    suite = bm.BenchmarkSuite(data_dir=str(tmp_path), n_bootstrap=3)
    suite._cal_c1, suite._cal_c0 = 2.0, 0.5
    captured = {}

    monkeypatch.setattr(bm, "run_inference", lambda *args, **kwargs: np.array([0.2, np.nan, 0.4], dtype=np.float32))
    monkeypatch.setattr(bm, "apply_calibration", lambda scores, c1, c0: scores + 1.0)

    def _fake_evaluate(preds, n_bootstrap):
        captured["scores"] = preds.scores
        captured["labels"] = preds.labels
        captured["gene_ids"] = preds.gene_ids
        return EvalResult(auroc=0.9, auprc=0.8, n_variants=len(preds.scores), n_genes=len(np.unique(preds.gene_ids)))

    monkeypatch.setattr(bm, "evaluate", _fake_evaluate)

    res = suite.run_one(model=None, pipeline=None, name="toy", device="cpu", batch_size=2)

    assert isinstance(res, EvalResult)
    assert np.allclose(captured["scores"], np.array([1.2, 1.4], dtype=np.float32))
    assert np.array_equal(captured["labels"], np.array([0, 1]))
    # Should fall back to protein_id because gene_id column is absent.
    assert np.array_equal(captured["gene_ids"], np.array(["P1", "P3"]))


def test_run_all_collects_only_successful_benchmarks(monkeypatch, tmp_path):
    suite = bm.BenchmarkSuite(data_dir=str(tmp_path), n_bootstrap=3)

    def _fake_run_one(model, pipeline, name, device="cuda", batch_size=32):
        if name == "skip":
            return None
        return EvalResult(
            auroc=0.81,
            auroc_ci=(0.7, 0.9),
            auprc=0.75,
            gene_bias_auroc=0.6,
            debiased_auroc=0.78,
            gene_auroc_mean=0.74,
            brier=0.2,
            ece=0.05,
            frac_ambiguous=0.12,
            n_variants=100,
            n_genes=20,
        )

    monkeypatch.setattr(suite, "run_one", _fake_run_one)
    out = suite.run_all(model=None, pipeline=None, benchmarks=["keep", "skip"], device="cpu", batch_size=4)

    assert len(out) == 1
    assert out.iloc[0]["benchmark"] == "keep"
    assert out.iloc[0]["auroc"] == 0.81
