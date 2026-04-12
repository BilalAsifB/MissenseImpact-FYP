"""
Additional unit tests for ThresholdLearner in evaluation/threshold_calibration.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.threshold_calibration import ThresholdLearner, ThresholdBundle, AM_PRECISION_TARGET
from evaluation import threshold_calibration as tc


def _val_df():
    return pd.DataFrame(
        [
            {"protein_id": "P1", "sequence": "ACD", "position": 1, "reference_aa": "A", "alternate_aa": "V", "label": 0},
            {"protein_id": "P2", "sequence": "ACD", "position": 2, "reference_aa": "C", "alternate_aa": "D", "label": 1},
            {"protein_id": "P3", "sequence": "ACD", "position": 3, "reference_aa": "D", "alternate_aa": "E", "label": 0},
            {"protein_id": "P4", "sequence": "ACD", "position": 1, "reference_aa": "A", "alternate_aa": "C", "label": 1},
        ]
    )


def test_threshold_learner_fit_validates_required_columns(tmp_path):
    learner = ThresholdLearner(model=None, pipeline=None, device="cpu", batch_size=2)
    p = tmp_path / "val.csv"
    pd.DataFrame([{"protein_id": "P1"}]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="val_csv missing columns"):
        learner.fit(str(p))


def test_threshold_learner_fit_raises_for_single_class(monkeypatch, tmp_path):
    p = tmp_path / "val.csv"
    df = _val_df().copy()
    df["label"] = 1
    df.to_csv(p, index=False)

    monkeypatch.setattr(tc, "derive_thresholds", lambda scores, labels, target: (0.8, 0.9, 0.2, 0.9, 0.1))
    monkeypatch.setattr(tc, "roc_auc_score", lambda y, s: 0.8)

    import evaluation.benchmark as bm
    import evaluation.metrics as em

    monkeypatch.setattr(bm, "run_inference", lambda *args, **kwargs: np.array([0.1, 0.2, 0.3, 0.4]))
    monkeypatch.setattr(em, "fit_calibration", lambda logits, labels: (1.0, 0.0))
    monkeypatch.setattr(em, "apply_calibration", lambda logits, c1, c0: logits)

    learner = ThresholdLearner(model=None, pipeline=None, device="cpu", batch_size=2)
    with pytest.raises(ValueError, match="must contain both pathogenic"):
        learner.fit(str(p))


def test_threshold_learner_fit_happy_path_with_am_comparison(monkeypatch, tmp_path):
    p = tmp_path / "val.csv"
    _val_df().to_csv(p, index=False)

    import evaluation.benchmark as bm
    import evaluation.metrics as em

    monkeypatch.setattr(bm, "run_inference", lambda *args, **kwargs: np.array([0.1, np.nan, 0.3, 0.9], dtype=np.float32))
    monkeypatch.setattr(em, "fit_calibration", lambda logits, labels: (2.0, -0.5))
    monkeypatch.setattr(em, "apply_calibration", lambda logits, c1, c0: np.array([0.2, 0.4, 0.8], dtype=np.float32))
    monkeypatch.setattr(tc, "roc_auc_score", lambda y, s: 0.77)
    monkeypatch.setattr(tc, "derive_thresholds", lambda scores, labels, target: (0.7, 0.92, 0.25, 0.91, 0.2))

    learner = ThresholdLearner(model=None, pipeline=None, device="cpu", batch_size=2)
    bundle = learner.fit(str(p), precision_target=AM_PRECISION_TARGET, compare_to_am=True)

    assert isinstance(bundle, ThresholdBundle)
    assert bundle.path_thresh == 0.7
    assert bundle.benign_thresh == 0.25
    assert bundle.path_precision == 0.92
    assert bundle.benign_precision == 0.91
    assert bundle.frac_ambiguous == 0.2
    assert bundle.n_variants == 3  # one NaN filtered out
    assert bundle.n_pathogenic == 1
    assert bundle.n_benign == 2
    assert bundle.am_path_thresh is not None
    assert bundle.path_thresh_delta is not None


def test_threshold_learner_fit_without_am_compare(monkeypatch, tmp_path):
    p = tmp_path / "val.csv"
    _val_df().to_csv(p, index=False)

    import evaluation.benchmark as bm
    import evaluation.metrics as em

    monkeypatch.setattr(bm, "run_inference", lambda *args, **kwargs: np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))
    monkeypatch.setattr(em, "fit_calibration", lambda logits, labels: (1.0, 0.0))
    monkeypatch.setattr(em, "apply_calibration", lambda logits, c1, c0: logits)
    monkeypatch.setattr(tc, "roc_auc_score", lambda y, s: 0.7)
    monkeypatch.setattr(tc, "derive_thresholds", lambda scores, labels, target: (0.6, 0.9, 0.2, 0.9, 0.15))

    learner = ThresholdLearner(model=None, pipeline=None, device="cpu", batch_size=2)
    bundle = learner.fit(str(p), compare_to_am=False)

    assert bundle.path_thresh == 0.6
    assert bundle.am_path_thresh is None
