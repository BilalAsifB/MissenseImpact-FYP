#!/usr/bin/env python3
"""
Score new variants with a trained checkpoint.

Input CSV must have: protein_id, sequence, position, reference_aa, alternate_aa
Output CSV adds:     pathogenicity, logit, classification
"""

import sys, argparse, logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset       import SASVariantDataset, collate_variants
from model.esm_missense import ESMMissense
from training.trainer   import EMAModel
from evaluation.metrics import apply_calibration, fit_calibration

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s  %(message)s")


@torch.no_grad()
def predict(model, csv_path, device, batch_size=32,
            cal_c1=None, cal_c0=None):
    """Returns DataFrame with pathogenicity scores added."""
    # Add dummy label column if absent (needed by Dataset)
    df = pd.read_csv(csv_path)
    has_label = "label" in df.columns
    if not has_label:
        df["label"] = 0

    tmp = "/tmp/_predict_input.csv"
    df.to_csv(tmp, index=False)

    ds = SASVariantDataset(tmp)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    collate_fn=collate_variants, num_workers=2)

    model.eval()
    all_logits = []
    for batch in dl:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        all_logits.append(model(batch)["logit"].cpu().numpy())

    logits = np.concatenate(all_logits)

    if cal_c1 is not None:
        probs = apply_calibration(logits, cal_c1, cal_c0)
    else:
        probs = 1 / (1 + np.exp(-logits))

    if not has_label:
        df = df.drop(columns="label")

    df["logit"]          = logits
    df["pathogenicity"]  = probs

    # Apply AM-style classification thresholds
    # Defaults from paper: likely_pathogenic >= 0.564, likely_benign <= 0.34
    df["classification"] = "ambiguous"
    df.loc[df["pathogenicity"] >= 0.564, "classification"] = "likely_pathogenic"
    df.loc[df["pathogenicity"] <= 0.340, "classification"] = "likely_benign"

    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--input_csv",   required=True)
    p.add_argument("--output_csv",  required=True)
    p.add_argument("--val_csv",     default=None,
                   help="If provided, fit calibration on this set first")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--batch_size",  type=int, default=32)
    args = p.parse_args()

    ckpt  = torch.load(args.checkpoint, map_location=args.device)
    kw    = ckpt.get("model_kwargs", {})
    model = ESMMissense(**kw).to(args.device)
    if "ema_state" in ckpt:
        ema = EMAModel(model)
        ema.load_state_dict(ckpt["ema_state"])
        ema.apply_to(model)
    else:
        model.load_state_dict(ckpt.get("model_state", ckpt))

    cal_c1 = cal_c0 = None
    if args.val_csv:
        from data.pipeline      import DataPipeline
        from evaluation.benchmark import run_inference
        pipeline = DataPipeline()
        val_df   = pd.read_csv(args.val_csv)
        val_logits = run_inference(model, val_df, pipeline,
                                   args.device, args.batch_size)
        valid = ~np.isnan(val_logits)
        cal_c1, cal_c0 = fit_calibration(val_logits[valid],
                                          val_df["label"].values[valid])
        logging.getLogger(__name__).info(
            "Calibration: c1=%.4f  c0=%.4f", cal_c1, cal_c0)

    result = predict(model, args.input_csv, args.device,
                     args.batch_size, cal_c1, cal_c0)
    result.to_csv(args.output_csv, index=False)

    counts = result["classification"].value_counts()
    print(f"\nSaved {len(result):,} predictions → {args.output_csv}")
    print(counts.to_string())

if __name__ == "__main__":
    main()
