"""
Unit tests for evaluation/bias_report.py.
"""

from __future__ import annotations

import json

import numpy as np

from evaluation.bias_report import BiasReport, _compute_metrics
from evaluation.threshold_calibration import ThresholdBundle, AM_PATH_THRESH, AM_BEN_THRESH


def _bundle():
    return ThresholdBundle(
        path_thresh=0.7,
        benign_thresh=0.3,
        path_precision=0.9,
        benign_precision=0.9,
        frac_ambiguous=0.1,
        auroc=0.85,
        cal_c1=1.2,
        cal_c0=-0.1,
        n_variants=6,
        n_pathogenic=3,
        n_benign=3,
    ).compare_to_am()


def test_compute_metrics_basic_counts():
    scores = np.array([0.8, 0.9, 0.2, 0.1], dtype=np.float32)
    labels = np.array([1, 0, 0, 1], dtype=np.int32)
    m = _compute_metrics(
        scores=scores,
        labels=labels,
        path_thresh=0.75,
        benign_thresh=0.25,
        population="SAS",
        threshold_source="sas_derived",
    )
    assert m.population == "SAS"
    assert m.threshold_source == "sas_derived"
    assert m.n_variants == 4
    assert 0.0 <= m.true_positive_rate <= 1.0
    assert 0.0 <= m.false_positive_rate <= 1.0
    assert 0.0 <= m.precision_pathogenic <= 1.0
    assert 0.0 <= m.precision_benign <= 1.0


def test_bias_report_properties_and_formatting():
    scores = np.array([0.95, 0.85, 0.2, 0.1, 0.6, 0.4], dtype=np.float32)
    labels = np.array([1, 1, 0, 0, 0, 1], dtype=np.int32)
    report = BiasReport(_bundle(), scores, labels)

    assert isinstance(report.precision_gap_pathogenic, float)
    assert isinstance(report.fpr_increase, float)
    assert isinstance(report.ambiguity_change, float)

    txt = report._format_report()
    assert "BIAS QUANTIFICATION REPORT" in txt
    assert "Threshold placement" in txt
    assert "Classification metrics on SAS val set" in txt


def test_bias_report_uses_am_and_sas_thresholds():
    scores = np.array([0.9, 0.8, 0.2, 0.1], dtype=np.float32)
    labels = np.array([1, 0, 0, 1], dtype=np.int32)
    b = _bundle()
    report = BiasReport(b, scores, labels)

    assert report.sas_under_am.path_thresh == round(AM_PATH_THRESH, 4)
    assert report.sas_under_am.benign_thresh == round(AM_BEN_THRESH, 4)
    assert report.sas_under_sas.path_thresh == round(b.path_thresh, 4)
    assert report.sas_under_sas.benign_thresh == round(b.benign_thresh, 4)


def test_bias_report_save_json(tmp_path):
    scores = np.array([0.9, 0.8, 0.2, 0.1], dtype=np.float32)
    labels = np.array([1, 0, 0, 1], dtype=np.int32)
    report = BiasReport(_bundle(), scores, labels)

    out = tmp_path / "reports" / "bias_report.json"
    report.save_json(out)

    assert out.exists()
    payload = json.loads(out.read_text())
    assert "bundle" in payload
    assert "sas_under_am_thresholds" in payload
    assert "sas_under_sas_thresholds" in payload
    assert "summary" in payload
