"""
Unit tests for evaluation/threshold_calibration.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.threshold_calibration import (
    ThresholdBundle,
    derive_thresholds,
    AM_PATH_THRESH,
    AM_BEN_THRESH,
    AM_PRECISION_TARGET,
)


def _synthetic_scores(n=400, seed=0):
    """
    Generate synthetic calibrated scores and labels where the positive
    class has higher scores. Simulates a reasonably good model.
    """
    rng = np.random.default_rng(seed)
    labels = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.int32)
    # Benign scores centred at 0.3, pathogenic at 0.7
    scores = np.where(
        labels == 1,
        rng.beta(5, 2, n).astype(np.float32),   # peaks near 0.8
        rng.beta(2, 5, n).astype(np.float32),   # peaks near 0.2
    )
    return scores, labels


# ── derive_thresholds ──────────────────────────────────────────────────────

class TestDeriveThresholds:
    def test_returns_five_values(self):
        scores, labels = _synthetic_scores()
        result = derive_thresholds(scores, labels)
        assert len(result) == 5

    def test_path_thresh_achieves_target_precision(self):
        scores, labels = _synthetic_scores()
        path_t, path_prec, _, _, _ = derive_thresholds(scores, labels, target=0.90)
        # Precision at the threshold must be >= 90%
        assert path_prec >= 0.90 - 1e-6

    def test_benign_thresh_achieves_target_precision(self):
        scores, labels = _synthetic_scores()
        _, _, ben_t, ben_prec, _ = derive_thresholds(scores, labels, target=0.90)
        assert ben_prec >= 0.90 - 1e-6

    def test_path_thresh_above_benign_thresh(self):
        scores, labels = _synthetic_scores()
        path_t, _, ben_t, _, _ = derive_thresholds(scores, labels)
        assert path_t > ben_t, "Pathogenic threshold must be higher than benign"

    def test_overlap_case_is_repaired_to_strict_ordering(self):
        # Construct a difficult distribution where naive first-hit selection from
        # two PR curves can overlap or cross in threshold space.
        scores = np.array(
            [0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.48, 0.47, 0.55, 0.56],
            dtype=np.float32,
        )
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
        path_t, _, ben_t, _, _ = derive_thresholds(scores, labels, target=0.60)
        assert path_t > ben_t

    def test_frac_ambiguous_between_zero_and_one(self):
        scores, labels = _synthetic_scores()
        _, _, _, _, frac_amb = derive_thresholds(scores, labels)
        assert 0.0 <= frac_amb <= 1.0

    def test_higher_target_gives_stricter_path_threshold(self):
        scores, labels = _synthetic_scores()
        path_t_90, *_ = derive_thresholds(scores, labels, target=0.90)
        path_t_95, *_ = derive_thresholds(scores, labels, target=0.95)
        # Higher precision target → need a higher score cutoff
        assert path_t_95 >= path_t_90

    def test_perfect_separator_achieves_100pct_precision(self):
        """With a perfect model, any target precision should be achievable."""
        scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
        labels = np.array([0, 0, 1, 1], dtype=np.int32)
        path_t, path_prec, ben_t, ben_prec, _ = derive_thresholds(scores, labels, 0.99)
        assert path_prec >= 0.99
        assert ben_prec >= 0.99


# ── ThresholdBundle ────────────────────────────────────────────────────────

class TestThresholdBundle:
    def _make_bundle(self, **overrides):
        defaults = dict(
            path_thresh=0.62,
            benign_thresh=0.28,
            path_precision=0.91,
            benign_precision=0.92,
            frac_ambiguous=0.18,
            auroc=0.85,
            cal_c1=1.2,
            cal_c0=-0.1,
            n_variants=500,
            n_pathogenic=250,
            n_benign=250,
            precision_target=0.90,
        )
        defaults.update(overrides)
        return ThresholdBundle(**defaults)

    def test_compare_to_am_populates_deltas(self):
        b = self._make_bundle(path_thresh=0.62, benign_thresh=0.28)
        b.compare_to_am()
        assert b.am_path_thresh == AM_PATH_THRESH
        assert b.am_ben_thresh  == AM_BEN_THRESH
        assert abs(b.path_thresh_delta  - (0.62 - AM_PATH_THRESH))  < 1e-6
        assert abs(b.ben_thresh_delta   - (0.28 - AM_BEN_THRESH))   < 1e-6

    def test_save_and_load_roundtrip(self, tmp_path):
        b = self._make_bundle()
        b.compare_to_am()
        out = tmp_path / "bundle.json"
        b.save(out)
        loaded = ThresholdBundle.load(out)
        assert loaded.path_thresh   == b.path_thresh
        assert loaded.benign_thresh == b.benign_thresh
        assert loaded.cal_c1        == b.cal_c1
        assert loaded.cal_c0        == b.cal_c0
        assert loaded.auroc         == b.auroc
        assert loaded.am_path_thresh == AM_PATH_THRESH

    def test_save_creates_parent_dirs(self, tmp_path):
        b = self._make_bundle()
        nested = tmp_path / "deep" / "nested" / "bundle.json"
        b.save(nested)
        assert nested.exists()

    def test_summary_contains_key_fields(self):
        b = self._make_bundle(path_thresh=0.62, benign_thresh=0.28)
        b.compare_to_am()
        s = b.summary()
        assert "0.6200" in s
        assert "0.2800" in s
        assert "auROC"  in s
        # Delta should appear (with sign)
        assert "+" in s or "-" in s

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ThresholdBundle.load(tmp_path / "missing.json")

    def test_sas_threshold_above_am_shows_positive_delta(self):
        """If our SAS path threshold is higher, delta is positive."""
        b = self._make_bundle(path_thresh=AM_PATH_THRESH + 0.05)
        b.compare_to_am()
        assert b.path_thresh_delta > 0

    def test_sas_threshold_below_am_shows_negative_delta(self):
        """If our SAS path threshold is lower, delta is negative."""
        b = self._make_bundle(path_thresh=AM_PATH_THRESH - 0.05)
        b.compare_to_am()
        assert b.path_thresh_delta < 0
