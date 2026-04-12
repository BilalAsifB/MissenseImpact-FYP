"""
Derives SAS-population-specific classification thresholds from a
validation set, replacing the AM paper's hardcoded 0.564 / 0.340
values (which were calibrated on European-weighted data).

Usage (standalone):
    python -m evaluation.threshold_calibration \
        --checkpoint checkpoints/model0_best.pt \
        --val_csv    data/processed/val.csv \
        --output     checkpoints/sas_thresholds.json

Usage (imported):
    from evaluation.threshold_calibration import ThresholdLearner, ThresholdBundle
    learner = ThresholdLearner(model, pipeline, device)
    bundle  = learner.fit(val_csv)
    bundle.save("checkpoints/sas_thresholds.json")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score

log = logging.getLogger(__name__)

# ── AM paper reference values (European-derived) ───────────────────────────
AM_PATH_THRESH  = 0.564
AM_BEN_THRESH   = 0.340
AM_PRECISION_TARGET = 0.90


# ── Data container ─────────────────────────────────────────────────────────

@dataclass
class ThresholdBundle:
    """
    All threshold-related values derived from a SAS validation set.

    Fields
    ------
    path_thresh      : score >= this → likely_pathogenic
    benign_thresh    : score <= this → likely_benign
    (between the two) → ambiguous

    path_precision   : actual precision achieved at path_thresh
    benign_precision : actual precision achieved at benign_thresh
    frac_ambiguous   : fraction of val variants landing in the gap
    auroc            : val-set auROC (sanity check)
    cal_c1, cal_c0   : logistic calibration coefficients
    n_variants       : how many val variants were used
    n_pathogenic     : number with label=1
    n_benign         : number with label=0
    precision_target : the target precision used during derivation

    Bias fields (populated when compare_to_am() is called)
    ------
    am_path_thresh   : AM's original pathogenic threshold
    am_ben_thresh    : AM's original benign threshold
    path_thresh_delta: our threshold - AM threshold (positive = stricter)
    ben_thresh_delta : our threshold - AM threshold (negative = stricter)
    """
    path_thresh:       float
    benign_thresh:     float
    path_precision:    float
    benign_precision:  float
    frac_ambiguous:    float
    auroc:             float
    cal_c1:            float
    cal_c0:            float
    n_variants:        int
    n_pathogenic:      int
    n_benign:          int
    precision_target:  float = AM_PRECISION_TARGET

    # populated by compare_to_am()
    am_path_thresh:    Optional[float] = None
    am_ben_thresh:     Optional[float] = None
    path_thresh_delta: Optional[float] = None
    ben_thresh_delta:  Optional[float] = None

    def compare_to_am(self) -> "ThresholdBundle":
        """
        Annotate the bundle with a direct comparison to AM's European thresholds.
        Deltas tell you how much our SAS thresholds differ from the paper.

        A positive path_thresh_delta means our model needs a *higher* score
        to reach 90% precision on SAS variants — i.e. it is less confident
        for this population at the same precision level.

        A negative ben_thresh_delta means our model needs a *lower* score
        to reach 90% benign precision — same interpretation.
        """
        self.am_path_thresh    = AM_PATH_THRESH
        self.am_ben_thresh     = AM_BEN_THRESH
        self.path_thresh_delta = round(self.path_thresh  - AM_PATH_THRESH,  4)
        self.ben_thresh_delta  = round(self.benign_thresh - AM_BEN_THRESH,   4)
        return self

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))
        log.info("ThresholdBundle saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "ThresholdBundle":
        data = json.loads(Path(path).read_text())
        return cls(**data)

    def summary(self) -> str:
        lines = [
            f"{'=' * 56}",
            f"  SAS Threshold Bundle  (n={self.n_variants:,} val variants)",
            f"{'=' * 56}",
            f"  auROC             : {self.auroc:.4f}",
            f"  Calibration       : c1={self.cal_c1:.4f}  c0={self.cal_c0:.4f}",
            f"{'─' * 56}",
            f"  Target precision  : {self.precision_target:.0%}",
            f"  Pathogenic thresh : {self.path_thresh:.4f}"
            f"  (precision={self.path_precision:.3f})",
            f"  Benign thresh     : {self.benign_thresh:.4f}"
            f"  (precision={self.benign_precision:.3f})",
            f"  Ambiguous frac    : {self.frac_ambiguous:.3f}",
        ]
        if self.am_path_thresh is not None:
            lines += [
                f"{'─' * 56}",
                f"  vs AM (European)",
                f"  Path  delta : {self.path_thresh_delta:+.4f}"
                f"  ({self.path_thresh:.4f} vs {self.am_path_thresh:.4f})",
                f"  Benign delta: {self.ben_thresh_delta:+.4f}"
                f"  ({self.benign_thresh:.4f} vs {self.am_ben_thresh:.4f})",
                f"  Interpretation: "
                + ("SAS requires higher score for pathogenic calls"
                   if self.path_thresh_delta > 0.01
                   else "SAS thresholds close to AM" if abs(self.path_thresh_delta) <= 0.01
                   else "SAS requires lower score for pathogenic calls"),
            ]
        lines.append(f"{'=' * 56}")
        return "\n".join(lines)


# ── Threshold derivation logic ─────────────────────────────────────────────

def derive_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    target: float = AM_PRECISION_TARGET,
) -> tuple[float, float, float, float, float]:
    """
    Derive pathogenic and benign thresholds that achieve `target` precision.

    Returns
    -------
    path_thresh, path_precision, benign_thresh, benign_precision, frac_ambiguous
    """
    # --- Pathogenic threshold candidates ---
    # Walk the precision-recall curve: collect thresholds with precision >= target
    # for the pathogenic class.
    prec, _, thresh = precision_recall_curve(labels, scores)
    path_idx = np.where(prec[:-1] >= target)[0]
    path_candidates = [(float(thresh[i]), float(prec[i])) for i in path_idx]
    if not path_candidates:
        path_thresh = float(scores.max())
        path_precision = float(prec[-2]) if len(prec) > 1 else 0.0
        log.warning(
            "Could not achieve %.0f%% pathogenic precision — "
            "falling back to max score %.4f (precision=%.4f). "
            "Consider a lower target or more pathogenic training data.",
            target * 100, path_thresh, path_precision,
        )
        path_candidates = [(path_thresh, path_precision)]

    # --- Benign threshold candidates ---
    # Mirror: invert labels and negate scores so precision_recall_curve treats
    # 'benign' as the positive class.
    prec_b, _, thresh_b = precision_recall_curve(1 - labels, -scores)
    ben_idx = np.where(prec_b[:-1] >= target)[0]
    benign_candidates = [(float(-thresh_b[i]), float(prec_b[i])) for i in ben_idx]
    if not benign_candidates:
        benign_thresh = float(scores.min())
        benign_precision = float(prec_b[-2]) if len(prec_b) > 1 else 0.0
        log.warning(
            "Could not achieve %.0f%% benign precision — "
            "falling back to min score %.4f (precision=%.4f).",
            target * 100, benign_thresh, benign_precision,
        )
        benign_candidates = [(benign_thresh, benign_precision)]

    # Keep historical behaviour as first choice, then repair overlap if needed.
    path_thresh, path_precision = path_candidates[0]
    benign_thresh, benign_precision = benign_candidates[0]

    if path_thresh <= benign_thresh:
        feasible = [
            (pt, pp, bt, bp)
            for pt, pp in path_candidates
            for bt, bp in benign_candidates
            if pt > bt
        ]
        if feasible:
            # Prefer the tightest non-overlapping pair to minimize ambiguous gap.
            path_thresh, path_precision, benign_thresh, benign_precision = min(
                feasible,
                key=lambda t: (t[0] - t[2], -t[1], -t[3]),
            )
        else:
            # As a last resort, enforce strict ordering numerically.
            # This preserves the intended class regions even in degenerate PR cases.
            if path_thresh == benign_thresh:
                benign_thresh = float(np.nextafter(benign_thresh, -np.inf))
            else:
                # Keep highest-safe pathogenic and lowest-safe benign candidates.
                path_thresh, path_precision = max(path_candidates, key=lambda t: t[0])
                benign_thresh, benign_precision = min(benign_candidates, key=lambda t: t[0])
                if path_thresh <= benign_thresh:
                    benign_thresh = float(np.nextafter(path_thresh, -np.inf))

    # Fraction of variants that fall in the ambiguous gap
    frac_ambiguous = float(
        np.mean((scores > benign_thresh) & (scores < path_thresh))
    )
    return path_thresh, path_precision, benign_thresh, benign_precision, frac_ambiguous


# ── Main learner class ─────────────────────────────────────────────────────

class ThresholdLearner:
    """
    Runs inference on a validation CSV and derives SAS-specific thresholds.

    Parameters
    ----------
    model      : trained ESMMissense (or any model with a forward() → {"logit": ...})
    pipeline   : DataPipeline instance
    device     : "cuda" or "cpu"
    batch_size : inference batch size
    """

    def __init__(self, model, pipeline, device: str = "cuda", batch_size: int = 32):
        self.model      = model
        self.pipeline   = pipeline
        self.device     = device
        self.batch_size = batch_size

    def fit(
        self,
        val_csv: str,
        precision_target: float = AM_PRECISION_TARGET,
        compare_to_am: bool = True,
    ) -> ThresholdBundle:
        """
        Full pipeline:
          1. Run inference on val_csv
          2. Fit logistic calibration
          3. Derive thresholds at `precision_target`
          4. Optionally annotate with AM deltas
        """
        import pandas as pd
        from evaluation.benchmark import run_inference
        from evaluation.metrics import fit_calibration, apply_calibration

        val_df = pd.read_csv(val_csv)
        required = {"protein_id", "sequence", "position",
                    "reference_aa", "alternate_aa", "label"}
        missing = required - set(val_df.columns)
        if missing:
            raise ValueError(f"val_csv missing columns: {missing}")

        log.info("Running inference on %d val variants...", len(val_df))
        logits = run_inference(
            self.model, val_df, self.pipeline,
            self.device, self.batch_size,
        )

        valid_mask = ~np.isnan(logits)
        if valid_mask.sum() < len(logits):
            log.warning(
                "%d variants skipped during inference (sequence validation errors)",
                (~valid_mask).sum(),
            )

        logits_v = logits[valid_mask]
        labels_v = val_df["label"].values[valid_mask].astype(int)
        n_path   = int(labels_v.sum())
        n_benign = int((labels_v == 0).sum())

        if len(np.unique(labels_v)) < 2:
            raise ValueError(
                "Val set must contain both pathogenic (label=1) and benign "
                "(label=0) variants to derive thresholds. "
                f"Found: {n_path} pathogenic, {n_benign} benign."
            )

        log.info(
            "Val set: %d pathogenic, %d benign (%d total valid)",
            n_path, n_benign, len(logits_v),
        )

        # Step 2: Calibration — maps raw logits to well-calibrated probabilities
        log.info("Fitting logistic calibration...")
        cal_c1, cal_c0 = fit_calibration(logits_v, labels_v)
        scores = apply_calibration(logits_v, cal_c1, cal_c0)
        log.info("Calibration: c1=%.4f  c0=%.4f", cal_c1, cal_c0)

        # Step 3: auROC on calibrated scores
        auroc = float(roc_auc_score(labels_v, scores))
        log.info("Val auROC: %.4f", auroc)

        # Step 4: Derive thresholds
        log.info("Deriving thresholds at %.0f%% precision target...", precision_target * 100)
        path_thresh, path_prec, ben_thresh, ben_prec, frac_amb = derive_thresholds(
            scores, labels_v, precision_target,
        )
        log.info(
            "Pathogenic threshold: %.4f (precision=%.3f)",
            path_thresh, path_prec,
        )
        log.info(
            "Benign threshold:     %.4f (precision=%.3f)",
            ben_thresh, ben_prec,
        )
        log.info("Ambiguous fraction:   %.3f", frac_amb)

        bundle = ThresholdBundle(
            path_thresh      = round(path_thresh,    4),
            benign_thresh    = round(ben_thresh,     4),
            path_precision   = round(path_prec,      4),
            benign_precision = round(ben_prec,       4),
            frac_ambiguous   = round(frac_amb,       4),
            auroc            = round(auroc,          4),
            cal_c1           = round(cal_c1,         6),
            cal_c0           = round(cal_c0,         6),
            n_variants       = int(valid_mask.sum()),
            n_pathogenic     = n_path,
            n_benign         = n_benign,
            precision_target = precision_target,
        )

        if compare_to_am:
            bundle.compare_to_am()

        log.info("\n%s", bundle.summary())
        return bundle


# ── Standalone entry point ─────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import torch
    from model.esm_missense import ESMMissense
    from data.pipeline import DataPipeline
    from training.trainer import EMAModel

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s  %(message)s",
    )

    p = argparse.ArgumentParser(
        description="Derive SAS-specific classification thresholds from a val set"
    )
    p.add_argument("--checkpoint",        required=True, help="Path to model checkpoint")
    p.add_argument("--val_csv",           required=True, help="Validation CSV with label column")
    p.add_argument("--output",            default="checkpoints/sas_thresholds.json")
    p.add_argument("--device",            default="cuda")
    p.add_argument("--batch_size",        type=int,   default=32)
    p.add_argument("--precision_target",  type=float, default=AM_PRECISION_TARGET,
                   help="Target precision for threshold derivation (default=0.90)")
    p.add_argument("--no_am_compare",     action="store_true",
                   help="Skip comparison to AM European thresholds")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    kw   = ckpt.get("model_kwargs", {})
    model = ESMMissense(**kw).to(args.device)
    if "ema_state" in ckpt:
        ema = EMAModel(model)
        ema.load_state_dict(ckpt["ema_state"])
        ema.apply_to(model)
    else:
        model.load_state_dict(ckpt.get("model_state", ckpt))
    model.eval()

    pipeline = DataPipeline()
    learner  = ThresholdLearner(model, pipeline, args.device, args.batch_size)
    bundle   = learner.fit(
        args.val_csv,
        precision_target=args.precision_target,
        compare_to_am=not args.no_am_compare,
    )
    bundle.save(args.output)
    print(f"\nSaved: {args.output}")
