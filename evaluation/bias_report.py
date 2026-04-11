"""
Generates a bias quantification report comparing SAS-derived classification
thresholds against AM's European-derived values.

This is the core research output of Phase 2: documenting *how* and *how much*
the threshold placement differs between populations, and what clinical
consequence that difference has (false positive rate, misclassification rate).

Usage:
    from evaluation.bias_report import BiasReport
    report = BiasReport(sas_bundle, am_val_df, sas_val_df)
    report.print_report()
    report.save_json("reports/bias_report.json")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from evaluation.threshold_calibration import ThresholdBundle, AM_PATH_THRESH, AM_BEN_THRESH

log = logging.getLogger(__name__)


@dataclass
class PopulationMetrics:
    """Classification metrics for one population at one set of thresholds."""
    population:       str
    threshold_source: str
    path_thresh:      float
    benign_thresh:    float
    true_positive_rate:  float   # sensitivity on pathogenic class
    false_positive_rate: float   # FPR on benign class
    precision_pathogenic: float
    precision_benign:    float
    frac_ambiguous:      float
    n_variants:          int


def _compute_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    path_thresh: float,
    benign_thresh: float,
    population: str,
    threshold_source: str,
) -> PopulationMetrics:
    """Compute classification metrics for a given threshold pair."""
    pathogenic_mask   = labels == 1
    benign_mask       = labels == 0

    pred_path   = scores >= path_thresh
    pred_benign = scores <= benign_thresh

    # True positive rate (sensitivity): of real pathogenic variants,
    # how many did we call likely_pathogenic?
    tp = (pred_path & pathogenic_mask).sum()
    fn = (~pred_path & pathogenic_mask).sum()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # False positive rate: of real benign variants,
    # how many did we incorrectly call likely_pathogenic?
    fp = (pred_path & benign_mask).sum()
    tn = (~pred_path & benign_mask).sum()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Precision on pathogenic calls
    prec_path = tp / pred_path.sum() if pred_path.sum() > 0 else 0.0

    # Precision on benign calls
    true_benign_calls = (pred_benign & benign_mask).sum()
    prec_ben = true_benign_calls / pred_benign.sum() if pred_benign.sum() > 0 else 0.0

    frac_amb = float(np.mean((scores > benign_thresh) & (scores < path_thresh)))

    return PopulationMetrics(
        population=population,
        threshold_source=threshold_source,
        path_thresh=round(path_thresh, 4),
        benign_thresh=round(benign_thresh, 4),
        true_positive_rate=round(float(tpr), 4),
        false_positive_rate=round(float(fpr), 4),
        precision_pathogenic=round(float(prec_path), 4),
        precision_benign=round(float(prec_ben), 4),
        frac_ambiguous=round(frac_amb, 4),
        n_variants=len(scores),
    )


class BiasReport:
    """
    Compares classification performance of:
      (A) AM European thresholds applied to SAS variants
      (B) SAS-derived thresholds applied to SAS variants

    The gap between A and B quantifies the bias in AM's threshold placement
    for the South Asian population.

    Parameters
    ----------
    sas_bundle      : ThresholdBundle derived from your SAS val set
    sas_scores      : calibrated pathogenicity scores for SAS val variants
    sas_labels      : ground-truth labels (0/1) for SAS val variants
    """

    def __init__(
        self,
        sas_bundle: ThresholdBundle,
        sas_scores: np.ndarray,
        sas_labels: np.ndarray,
    ):
        self.bundle     = sas_bundle
        self.scores     = np.asarray(sas_scores, dtype=np.float32)
        self.labels     = np.asarray(sas_labels, dtype=np.int32)

        # Compute metrics under AM thresholds on SAS data
        self.sas_under_am = _compute_metrics(
            self.scores, self.labels,
            AM_PATH_THRESH, AM_BEN_THRESH,
            population="SAS",
            threshold_source="am_european",
        )

        # Compute metrics under SAS-derived thresholds
        self.sas_under_sas = _compute_metrics(
            self.scores, self.labels,
            sas_bundle.path_thresh, sas_bundle.benign_thresh,
            population="SAS",
            threshold_source="sas_derived",
        )

    @property
    def precision_gap_pathogenic(self) -> float:
        """
        How much precision is lost when applying AM thresholds to SAS data,
        compared to SAS-derived thresholds.

        A positive value means AM thresholds give lower precision on SAS —
        i.e. more false pathogenic calls for this population.
        """
        return round(
            self.sas_under_sas.precision_pathogenic
            - self.sas_under_am.precision_pathogenic,
            4,
        )

    @property
    def fpr_increase(self) -> float:
        """
        Increase in false positive rate when using AM vs SAS thresholds.
        Positive = AM thresholds produce more false alarms on SAS variants.
        """
        return round(
            self.sas_under_am.false_positive_rate
            - self.sas_under_sas.false_positive_rate,
            4,
        )

    @property
    def ambiguity_change(self) -> float:
        """
        Change in ambiguous fraction: AM vs SAS thresholds.
        Positive = AM thresholds leave more SAS variants unclassified.
        """
        return round(
            self.sas_under_am.frac_ambiguous
            - self.sas_under_sas.frac_ambiguous,
            4,
        )

    def print_report(self) -> None:
        print(self._format_report())

    def _format_report(self) -> str:
        am  = self.sas_under_am
        sas = self.sas_under_sas
        b   = self.bundle

        lines = [
            "",
            "=" * 62,
            "  BIAS QUANTIFICATION REPORT — SAS Threshold Analysis",
            "=" * 62,
            "",
            f"  Val set: {b.n_variants:,} SAS variants  "
            f"({b.n_pathogenic:,} pathogenic / {b.n_benign:,} benign)",
            f"  Model auROC: {b.auroc:.4f}",
            "",
            "─" * 62,
            "  Threshold placement",
            "─" * 62,
            f"  {'':30s}  {'AM':>8}  {'SAS':>8}  {'Delta':>8}",
            f"  {'Pathogenic threshold':30s}  {AM_PATH_THRESH:>8.4f}  "
            f"{b.path_thresh:>8.4f}  {b.path_thresh_delta:>+8.4f}",
            f"  {'Benign threshold':30s}  {AM_BEN_THRESH:>8.4f}  "
            f"{b.benign_thresh:>8.4f}  {b.ben_thresh_delta:>+8.4f}",
            "",
            "─" * 62,
            "  Classification metrics on SAS val set",
            "─" * 62,
            f"  {'Metric':30s}  {'AM thresh':>10}  {'SAS thresh':>10}  {'Gap':>8}",
            f"  {'Pathogenic precision':30s}  "
            f"{am.precision_pathogenic:>10.4f}  "
            f"{sas.precision_pathogenic:>10.4f}  "
            f"{self.precision_gap_pathogenic:>+8.4f}",
            f"  {'False positive rate':30s}  "
            f"{am.false_positive_rate:>10.4f}  "
            f"{sas.false_positive_rate:>10.4f}  "
            f"{self.fpr_increase:>+8.4f}",
            f"  {'Sensitivity (TPR)':30s}  "
            f"{am.true_positive_rate:>10.4f}  "
            f"{sas.true_positive_rate:>10.4f}  "
            f"{sas.true_positive_rate - am.true_positive_rate:>+8.4f}",
            f"  {'Benign precision':30s}  "
            f"{am.precision_benign:>10.4f}  "
            f"{sas.precision_benign:>10.4f}  "
            f"{sas.precision_benign - am.precision_benign:>+8.4f}",
            f"  {'Ambiguous fraction':30s}  "
            f"{am.frac_ambiguous:>10.4f}  "
            f"{sas.frac_ambiguous:>10.4f}  "
            f"{self.ambiguity_change:>+8.4f}",
            "",
            "─" * 62,
            "  Interpretation",
            "─" * 62,
        ]

        # Auto-generate interpretation based on direction of gaps
        if abs(self.precision_gap_pathogenic) < 0.01:
            lines.append("  Precision gap is small (<1%) — AM thresholds are")
            lines.append("  approximately valid for this SAS population.")
        elif self.precision_gap_pathogenic > 0:
            lines.append(
                f"  AM thresholds achieve {self.precision_gap_pathogenic*100:.1f}pp "
                f"lower pathogenic precision on"
            )
            lines.append(
                "  SAS variants than SAS-derived thresholds. Using AM thresholds"
            )
            lines.append(
                "  would produce more false pathogenic calls for South Asian patients."
            )
        else:
            lines.append(
                f"  AM thresholds are actually stricter for SAS data "
                f"({abs(self.precision_gap_pathogenic)*100:.1f}pp higher precision),"
            )
            lines.append(
                "  but at a cost — check the sensitivity and ambiguous fraction."
            )

        if self.fpr_increase > 0.02:
            lines.append(
                f"  False positive rate increases by {self.fpr_increase*100:.1f}pp "
                f"under AM thresholds — clinically significant."
            )

        lines += ["", "=" * 62, ""]
        return "\n".join(lines)

    def save_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        out = {
            "bundle": asdict(self.bundle),
            "sas_under_am_thresholds": asdict(self.sas_under_am),
            "sas_under_sas_thresholds": asdict(self.sas_under_sas),
            "summary": {
                "precision_gap_pathogenic": self.precision_gap_pathogenic,
                "fpr_increase_under_am": self.fpr_increase,
                "ambiguity_change": self.ambiguity_change,
            },
        }
        Path(path).write_text(json.dumps(out, indent=2))
        log.info("Bias report saved → %s", path)