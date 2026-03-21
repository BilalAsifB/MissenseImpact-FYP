"""
All metrics from the AlphaMissense paper.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    brier_score_loss, log_loss, precision_recall_curve,
)
from sklearn.calibration import calibration_curve


@dataclass
class VariantPredictions:
    scores:   np.ndarray   # (N,) — higher = more pathogenic
    labels:   np.ndarray   # (N,) int {0,1}
    gene_ids: np.ndarray   # (N,) str
    positions:np.ndarray   # (N,) int 1-based
    weights:  Optional[np.ndarray] = None
    source:   str = "unknown"

    def __post_init__(self):
        self.scores    = np.asarray(self.scores,    dtype=np.float32)
        self.labels    = np.asarray(self.labels,    dtype=np.int32)
        self.gene_ids  = np.asarray(self.gene_ids)
        self.positions = np.asarray(self.positions, dtype=np.int32)


@dataclass
class EvalResult:
    auroc:           float = 0.0
    auroc_ci:        tuple = (0.0, 0.0)
    auprc:           float = 0.0
    auprc_ci:        tuple = (0.0, 0.0)
    brier:           float = 0.0
    log_loss_val:    float = 0.0
    ece:             float = 0.0
    gene_bias_auroc: float = 0.0   # naive per-gene fraction predictor (AM Fig S9)
    debiased_auroc:  float = 0.0   # balanced per-gene subset
    gene_auroc_mean: float = 0.0
    path_thresh_90p: float = 0.0   # score threshold for 90% positive precision
    benign_thresh_90p: float = 0.0
    frac_ambiguous:  float = 0.0
    spearman_r:      Optional[float] = None
    spearman_p:      Optional[float] = None
    n_variants:      int = 0
    n_genes:         int = 0
    n_bootstrap:     int = 999
    per_gene:        Optional[pd.DataFrame] = None

    def summary(self) -> str:
        lines = [
            f"{'='*52}",
            f"  {self.n_variants:,} variants  {self.n_genes:,} genes",
            f"{'='*52}",
            f"  auROC          : {self.auroc:.4f}  "
            f"CI [{self.auroc_ci[0]:.4f}, {self.auroc_ci[1]:.4f}]",
            f"  auPRC          : {self.auprc:.4f}  "
            f"CI [{self.auprc_ci[0]:.4f}, {self.auprc_ci[1]:.4f}]",
            f"{'─'*52}",
            f"  Gene-bias auROC: {self.gene_bias_auroc:.4f}  (naive baseline)",
            f"  Debiased auROC : {self.debiased_auroc:.4f}",
            f"  Per-gene mean  : {self.gene_auroc_mean:.4f}",
            f"{'─'*52}",
            f"  Brier score    : {self.brier:.4f}",
            f"  ECE            : {self.ece:.4f}",
            f"  Frac ambiguous : {self.frac_ambiguous:.3f}",
        ]
        if self.spearman_r is not None:
            lines.append(f"  Spearman r     : {self.spearman_r:.4f}  "
                         f"p={self.spearman_p:.2e}")
        lines.append(f"{'='*52}")
        return "\n".join(lines)


# ── Bootstrap ──────────────────────────────────────────────────────────────

def _bootstrap(fn, *arrays, B=999, rng=None):
    if rng is None: rng = np.random.default_rng(42)
    N, vals = len(arrays[0]), []
    for _ in range(B):
        idx = rng.integers(0, N, N)
        try:
            v = fn(*[a[idx] for a in arrays])
            if np.isfinite(v): vals.append(v)
        except: pass
    if not vals: return (0.0, 0.0)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ── Gene-label bias (AM Supplementary Note + Fig S9) ──────────────────────

def gene_bias_auroc(preds: VariantPredictions) -> float:
    """auROC of per-gene pathogenic-fraction predictor.
    AM paper shows this naive baseline scores 0.914 on ClinVar.
    Your model must substantially exceed this."""
    df = pd.DataFrame({"gene": preds.gene_ids, "label": preds.labels})
    frac = df.groupby("gene")["label"].mean().rename("frac")
    df = df.join(frac, on="gene")
    try: return float(roc_auc_score(df["label"], df["frac"]))
    except: return float("nan")


def balance_per_gene(preds: VariantPredictions,
                     rng=None) -> VariantPredictions:
    """Keep min(n_pos, n_neg) per gene to remove gene-label bias."""
    if rng is None: rng = np.random.default_rng(42)
    df = pd.DataFrame({"gene": preds.gene_ids, "label": preds.labels,
                       "idx": np.arange(len(preds.labels))})
    keep = []
    for _, g in df.groupby("gene"):
        pos = g[g.label==1]["idx"].values
        neg = g[g.label==0]["idx"].values
        n = min(len(pos), len(neg))
        if n == 0: continue
        keep.extend(rng.choice(pos, n, replace=False))
        keep.extend(rng.choice(neg, n, replace=False))
    k = np.array(keep)
    return VariantPredictions(
        scores=preds.scores[k], labels=preds.labels[k],
        gene_ids=preds.gene_ids[k], positions=preds.positions[k],
        source=preds.source+"_balanced")


def per_gene_auroc(preds: VariantPredictions,
                   min_pos=5, min_neg=5) -> pd.DataFrame:
    """Per-gene auROC table (mirrors AM Fig S3)."""
    rows = []
    df = pd.DataFrame({"gene": preds.gene_ids, "score": preds.scores,
                       "label": preds.labels})
    for gene, g in df.groupby("gene"):
        if (g.label==1).sum() < min_pos or (g.label==0).sum() < min_neg:
            continue
        try:    auc = roc_auc_score(g["label"], g["score"])
        except: auc = float("nan")
        rows.append({"gene": gene, "auroc": auc,
                     "n_pos": int((g.label==1).sum()),
                     "n_neg": int((g.label==0).sum())})
    return pd.DataFrame(rows).sort_values("auroc", ascending=False)


# ── Calibration ────────────────────────────────────────────────────────────

def ece(scores: np.ndarray, labels: np.ndarray, n_bins=10) -> float:
    probs = 1/(1+np.exp(-scores)) if scores.min()<0 else scores
    edges = np.linspace(0, 1, n_bins+1)
    err = 0.0
    for i in range(n_bins):
        m = (probs >= edges[i]) & (probs < edges[i+1])
        if m.sum() == 0: continue
        err += m.mean() * abs(labels[m].mean() - probs[m].mean())
    return float(err)


def fit_calibration(logits: np.ndarray,
                    labels: np.ndarray) -> tuple[float, float]:
    """Fit AM's logistic rescaling s̃ = σ(c1·s + c0) on validation set."""
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(logits.reshape(-1,1), labels)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def apply_calibration(logits, c1, c0):
    return 1/(1+np.exp(-(c1*logits+c0)))


# ── Classification thresholds (AM Fig S4A) ─────────────────────────────────

def derive_thresholds(scores, labels, target=0.90):
    prec, rec, thresh = precision_recall_curve(labels, scores)
    v = np.where(prec[:-1] >= target)[0]
    path_t = float(thresh[v[0]]) if len(v) else float(scores.max())

    prec_n, _, thresh_n = precision_recall_curve(1-labels, -scores)
    v2 = np.where(prec_n[:-1] >= target)[0]
    ben_t = float(-thresh_n[v2[0]]) if len(v2) else float(scores.min())

    ambig = float(np.mean((scores > ben_t) & (scores < path_t)))
    return path_t, ben_t, ambig


# ── MAVE Spearman (AM Methods) ─────────────────────────────────────────────

def mave_spearman(scores, continuous_labels) -> tuple[float, float]:
    r, p = stats.spearmanr(scores, continuous_labels)
    return float(abs(r)), float(p)


# ── Full evaluation runner ─────────────────────────────────────────────────

def evaluate(preds: VariantPredictions,
             continuous_labels=None,
             n_bootstrap=999, seed=42) -> EvalResult:
    rng = np.random.default_rng(seed)
    res = EvalResult(n_variants=len(preds.scores),
                     n_genes=len(np.unique(preds.gene_ids)),
                     n_bootstrap=n_bootstrap)

    res.auroc    = float(roc_auc_score(preds.labels, preds.scores))
    res.auroc_ci = _bootstrap(roc_auc_score, preds.labels, preds.scores,
                               B=n_bootstrap, rng=rng)
    res.auprc    = float(average_precision_score(preds.labels, preds.scores))
    res.auprc_ci = _bootstrap(average_precision_score, preds.labels, preds.scores,
                               B=n_bootstrap, rng=rng)

    probs = 1/(1+np.exp(-preds.scores)) if preds.scores.min()<0 else preds.scores
    res.brier        = float(brier_score_loss(preds.labels, probs))
    res.log_loss_val = float(log_loss(preds.labels, probs))
    res.ece          = ece(preds.scores, preds.labels)

    res.gene_bias_auroc = gene_bias_auroc(preds)
    bal = balance_per_gene(preds, rng)
    if len(np.unique(bal.labels)) == 2:
        res.debiased_auroc = float(roc_auc_score(bal.labels, bal.scores))
    pg = per_gene_auroc(preds)
    res.per_gene = pg
    if len(pg): res.gene_auroc_mean = float(pg["auroc"].mean())

    res.path_thresh_90p, res.benign_thresh_90p, res.frac_ambiguous = \
        derive_thresholds(preds.scores, preds.labels)

    if continuous_labels is not None:
        res.spearman_r, res.spearman_p = mave_spearman(
            preds.scores, continuous_labels)

    return res
