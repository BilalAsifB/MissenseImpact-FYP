"""
Runs all AM-paper benchmarks.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.pipeline      import DataPipeline, ProteinVariant
from data.dataset       import collate_variants
from evaluation.metrics import (
    VariantPredictions, EvalResult, evaluate,
    fit_calibration, apply_calibration,
)

log = logging.getLogger(__name__)


@torch.no_grad()
def run_inference(model, df: pd.DataFrame, pipeline: DataPipeline,
                  device: str, batch_size: int = 32) -> np.ndarray:
    model.eval()
    logits = []
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start+batch_size]
        samples = []
        for _, row in chunk.iterrows():
            try:
                v = ProteinVariant(
                    protein_id=str(row["protein_id"]),
                    sequence=str(row["sequence"]),
                    position=int(row["position"]),
                    reference_aa=str(row["reference_aa"]),
                    alternate_aa=str(row["alternate_aa"]),
                    label=int(row["label"]),
                )
                samples.append(pipeline.process(v))
            except Exception as e:
                log.debug("Skipping variant: %s", e)
                samples.append(None)

        valid = [(i, s) for i, s in enumerate(samples) if s is not None]
        if not valid:
            logits.extend([np.nan] * len(chunk))
            continue

        idxs, samps = zip(*valid)
        batch = collate_variants(list(samps))
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        out = model(batch)["logit"].cpu().numpy()

        chunk_logits = np.full(len(chunk), np.nan)
        for local_i, logit in zip(idxs, out):
            chunk_logits[local_i] = logit
        logits.extend(chunk_logits)

    return np.array(logits)


class BenchmarkSuite:
    """
    Runs all AM-paper benchmarks and returns a summary DataFrame.

    Benchmarks:
        clinvar_test     — main ClinVar held-out set
        clinvar_balanced — per-gene balanced subset (no gene-label bias)
        cancer_hotspot   — inferred cancer hotspot mutations
        de_novo          — DDD rare disease de novo variants
        sas_test         — your SAS-filtered holdout

    Each benchmark CSV must have columns:
        protein_id, sequence, position, reference_aa, alternate_aa, label
    """

    def __init__(self, data_dir: str, n_bootstrap: int = 999):
        self.data_dir    = Path(data_dir)
        self.n_bootstrap = n_bootstrap
        self._cal_c1: Optional[float] = None
        self._cal_c0: Optional[float] = None

    def calibrate(self, model, pipeline, val_csv: str,
                  device="cuda", batch_size=32):
        """Fit AM's logistic calibration on the validation set."""
        val_df = pd.read_csv(val_csv)
        logits = run_inference(model, val_df, pipeline, device, batch_size)
        valid  = ~np.isnan(logits)
        self._cal_c1, self._cal_c0 = fit_calibration(
            logits[valid], val_df["label"].values[valid])
        log.info("Calibration: c1=%.4f  c0=%.4f", self._cal_c1, self._cal_c0)

    def run_one(self, model, pipeline, name: str,
                device="cuda", batch_size=32) -> Optional[EvalResult]:
        path = self.data_dir / f"{name}.csv"
        if not path.exists():
            log.warning("Benchmark not found: %s", path)
            return None

        df = pd.read_csv(path)
        logits = run_inference(model, df, pipeline, device, batch_size)
        valid  = ~np.isnan(logits)
        df_v   = df[valid].reset_index(drop=True)
        scores = logits[valid]

        if self._cal_c1 is not None:
            scores = apply_calibration(scores, self._cal_c1, self._cal_c0)

        preds = VariantPredictions(
            scores=scores, labels=df_v["label"].values,
            gene_ids=df_v.get("gene_id", df_v["protein_id"]).values,
            positions=df_v["position"].values, source=name)

        result = evaluate(preds, n_bootstrap=self.n_bootstrap)
        print(result.summary())
        return result

    def run_all(self, model, pipeline, device="cuda", batch_size=32,
                benchmarks=None) -> pd.DataFrame:
        if benchmarks is None:
            benchmarks = ["clinvar_test", "clinvar_balanced",
                          "cancer_hotspot", "de_novo", "sas_test"]
        rows = []
        for name in benchmarks:
            res = self.run_one(model, pipeline, name, device, batch_size)
            if res is None: continue
            rows.append({
                "benchmark":      name,
                "auroc":          res.auroc,
                "auroc_ci_lo":    res.auroc_ci[0],
                "auroc_ci_hi":    res.auroc_ci[1],
                "auprc":          res.auprc,
                "gene_bias_auroc":res.gene_bias_auroc,
                "debiased_auroc": res.debiased_auroc,
                "gene_auroc_mean":res.gene_auroc_mean,
                "brier":          res.brier,
                "ece":            res.ece,
                "frac_ambiguous": res.frac_ambiguous,
                "n_variants":     res.n_variants,
                "n_genes":        res.n_genes,
            })
        return pd.DataFrame(rows)
