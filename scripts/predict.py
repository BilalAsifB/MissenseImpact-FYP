#!/usr/bin/env python3
"""
Score new variants with a trained checkpoint.

SAS-specific thresholds replace the AM paper's hardcoded European values.
Thresholds are loaded from a ThresholdBundle JSON produced by:

    python -m evaluation.threshold_calibration \
        --checkpoint <ckpt> --val_csv <val> --output sas_thresholds.json

If --threshold_bundle is not provided, thresholds are derived on-the-fly
from --val_csv (requires label column).  If neither is available, AM's
original European thresholds are used with a warning.

Input CSV must have:  protein_id, sequence, position, ref_aa, alt_aa
Output CSV adds:      pathogenicity, logit, classification, threshold_source
"""
from __future__ import annotations

import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import SASVariantDataset, collate_variants
from evaluation.metrics import apply_calibration, fit_calibration
from evaluation.threshold_calibration import (
    ThresholdBundle, ThresholdLearner,
    AM_PATH_THRESH, AM_BEN_THRESH,
)
from model.esm_missense import ESMMissense
from training.trainer import EMAModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)


@torch.no_grad()
def predict(
    model,
    csv_path: str,
    device: str,
    batch_size: int = 32,
    cal_c1: float | None = None,
    cal_c0: float | None = None,
    path_thresh: float = AM_PATH_THRESH,
    benign_thresh: float = AM_BEN_THRESH,
    threshold_source: str = "am_european",
) -> pd.DataFrame:
    """
    Run inference and classify variants.

    threshold_source is written into the output CSV so downstream
    analyses know which thresholds were applied.
    """
    df = pd.read_csv(csv_path)
    has_label = "label" in df.columns
    if not has_label:
        df["label"] = 0

    tmp = "/tmp/_predict_input.csv"
    df.to_csv(tmp, index=False)

    ds = SASVariantDataset(tmp)
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_variants, num_workers=2,
    )

    model.eval()
    all_logits = []
    for batch in dl:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        all_logits.append(model(batch)["logit"].cpu().numpy())

    logits = np.concatenate(all_logits)

    if cal_c1 is not None and cal_c0 is not None:
        probs = apply_calibration(logits, cal_c1, cal_c0)
    else:
        probs = 1 / (1 + np.exp(-logits))

    if not has_label:
        df = df.drop(columns="label")

    df["logit"]       = logits
    df["pathogenicity"] = probs

    # SAS-specific (or AM fallback) classification
    df["classification"]    = "ambiguous"
    df.loc[df["pathogenicity"] >= path_thresh,   "classification"] = "likely_pathogenic"
    df.loc[df["pathogenicity"] <= benign_thresh,  "classification"] = "likely_benign"
    df["threshold_source"]  = threshold_source

    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",        required=True)
    p.add_argument("--input_csv",         required=True)
    p.add_argument("--output_csv",        required=True)

    # --- Threshold options (mutually exclusive, in priority order) ---
    thresh_group = p.add_mutually_exclusive_group()
    thresh_group.add_argument(
        "--threshold_bundle",
        default=None,
        help="Path to SAS ThresholdBundle JSON. "
             "Produced by evaluation/threshold_calibration.py. "
             "Highest priority — use this if you have it.",
    )
    thresh_group.add_argument(
        "--val_csv",
        default=None,
        help="If provided and no --threshold_bundle, derive thresholds "
             "on-the-fly from this val CSV (must have label column). "
             "Slower but convenient during experiments.",
    )

    p.add_argument(
        "--precision_target", type=float, default=0.90,
        help="Target precision for on-the-fly threshold derivation (default=0.90)",
    )
    p.add_argument("--device",     default="cuda")
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()

    # ── Load model ──────────────────────────────────────────────────────────
    ckpt  = torch.load(args.checkpoint, map_location=args.device)
    kw    = ckpt.get("model_kwargs", {})
    model = ESMMissense(**kw).to(args.device)
    if "ema_state" in ckpt:
        ema = EMAModel(model)
        ema.load_state_dict(ckpt["ema_state"])
        ema.apply_to(model)
    else:
        model.load_state_dict(ckpt.get("model_state", ckpt))
    model.eval()

    # ── Resolve thresholds ──────────────────────────────────────────────────
    cal_c1 = cal_c0 = None
    path_thresh    = AM_PATH_THRESH
    benign_thresh  = AM_BEN_THRESH
    threshold_source = "am_european"

    if args.threshold_bundle:
        # Option 1: pre-computed SAS bundle
        bundle = ThresholdBundle.load(args.threshold_bundle)
        cal_c1, cal_c0    = bundle.cal_c1, bundle.cal_c0
        path_thresh        = bundle.path_thresh
        benign_thresh      = bundle.benign_thresh
        threshold_source   = "sas_derived"
        log.info("Loaded SAS thresholds from %s", args.threshold_bundle)
        log.info(
            "  Pathogenic: %.4f  Benign: %.4f  "
            "(AM was %.3f / %.3f)",
            path_thresh, benign_thresh,
            AM_PATH_THRESH, AM_BEN_THRESH,
        )

    elif args.val_csv:
        # Option 2: derive on-the-fly from val CSV
        log.info(
            "No threshold_bundle provided — deriving thresholds "
            "from val CSV: %s", args.val_csv
        )
        from data.pipeline import DataPipeline
        pipeline = DataPipeline()
        learner  = ThresholdLearner(model, pipeline, args.device, args.batch_size)
        bundle   = learner.fit(
            args.val_csv,
            precision_target=args.precision_target,
        )
        cal_c1, cal_c0   = bundle.cal_c1, bundle.cal_c0
        path_thresh       = bundle.path_thresh
        benign_thresh     = bundle.benign_thresh
        threshold_source  = "sas_derived_onthefly"

        # Auto-save alongside the output for reproducibility
        auto_save = Path(args.output_csv).with_suffix("") / "sas_thresholds.json"
        auto_save.parent.mkdir(parents=True, exist_ok=True)
        bundle.save(auto_save)
        log.info("Auto-saved threshold bundle → %s", auto_save)

    else:
        # Option 3: fall back to AM thresholds with a clear warning
        log.warning(
            "=" * 60
        )
        log.warning(
            "WARNING: Using AM European thresholds (%.3f / %.3f). "
            "These were not calibrated on South Asian data and may "
            "produce incorrect precision for your population. "
            "Provide --threshold_bundle or --val_csv for SAS-specific thresholds.",
            AM_PATH_THRESH, AM_BEN_THRESH,
        )
        log.warning("=" * 60)

    # ── Run prediction ──────────────────────────────────────────────────────
    result = predict(
        model, args.input_csv, args.device,
        args.batch_size,
        cal_c1=cal_c1, cal_c0=cal_c0,
        path_thresh=path_thresh,
        benign_thresh=benign_thresh,
        threshold_source=threshold_source,
    )

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)

    counts = result["classification"].value_counts()
    print(f"\nSaved {len(result):,} predictions → {args.output_csv}")
    print(f"Threshold source: {threshold_source}")
    if threshold_source.startswith("sas"):
        print(f"  Pathogenic >= {path_thresh:.4f}  |  Benign <= {benign_thresh:.4f}")
    else:
        print(f"  AM fallback: Pathogenic >= {AM_PATH_THRESH}  |  Benign <= {AM_BEN_THRESH}")
    print(counts.to_string())


if __name__ == "__main__":
    main()