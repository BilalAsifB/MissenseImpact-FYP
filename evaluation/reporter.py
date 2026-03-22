"""
Generate AM-paper figures.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve
from evaluation.metrics import EvalResult, VariantPredictions

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


class Reporter:
    """Saves all AM-paper evaluation figures to output_dir."""

    COLORS = {"model": "#1f77b4", "bias": "#ff7f0e",
              "debiased": "#2ca02c", "path": "#d62728", "benign": "#1f77b4"}

    def __init__(self, output_dir: str = "reports"):
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    def generate_all(self, result: EvalResult, preds: VariantPredictions,
                     model_name: str = "ESM-Missense",
                     calibrated_probs: Optional[np.ndarray] = None):
        self._save_json(result, model_name)
        if not HAS_MPL:
            return
        self.plot_roc(result, preds, model_name)
        self.plot_pr(result, preds, model_name)
        self.plot_calibration(preds, calibrated_probs, model_name)
        self.plot_score_dist(result, preds, model_name)
        self.plot_gene_bias(result, preds, model_name)
        if result.per_gene is not None and len(result.per_gene):
            self.plot_per_gene(result.per_gene, model_name)

    def plot_roc(self, res, preds, name):
        fig, ax = plt.subplots(figsize=(5, 5))
        fpr, tpr, _ = roc_curve(preds.labels, preds.scores)
        ax.plot(fpr, tpr, lw=2, color=self.COLORS["model"],
                label=f"{name}  auROC={res.auroc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.axhline(res.gene_bias_auroc - 0.5, color=self.COLORS["bias"],
                   ls="--", alpha=0.6, label=f"gene-bias={res.gene_bias_auroc:.3f}")
        ax.set(xlabel="FPR", ylabel="TPR",
               title=f"ROC — {preds.source}\n"
               f"CI [{res.auroc_ci[0]:.4f},{res.auroc_ci[1]:.4f}]")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(self.out / f"roc_{preds.source}.png", dpi=150)
        plt.close(fig)

    def plot_pr(self, res, preds, name):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        prec, rec, thr = precision_recall_curve(preds.labels, preds.scores)
        ax1.plot(thr, prec[:-1], color=self.COLORS["path"], lw=2, label="Precision")
        ax1.plot(thr, rec[:-1], color=self.COLORS["path"], ls="--", lw=2, label="Recall")
        ax1.axvline(res.path_thresh_90p, color="gray", ls=":", lw=1.5,
                    label=f"90% prec @ {res.path_thresh_90p:.3f}")
        ax1.axhline(0.9, color="gray", ls=":", alpha=0.4)
        ax1.set(xlabel="Score", title="Pathogenic class")
        ax1.legend(fontsize=8)

        prec_n, rec_n, thr_n = precision_recall_curve(1 - preds.labels, -preds.scores)
        ax2.plot(-thr_n, prec_n[:-1], color=self.COLORS["benign"], lw=2)
        ax2.plot(-thr_n, rec_n[:-1], color=self.COLORS["benign"], ls="--", lw=2)
        ax2.axvline(res.benign_thresh_90p, color="gray", ls=":", lw=1.5,
                    label=f"90% prec @ {res.benign_thresh_90p:.3f}")
        ax2.axhline(0.9, color="gray", ls=":", alpha=0.4)
        ax2.set(xlabel="Score", title="Benign class")
        ax2.legend(fontsize=8)

        fig.suptitle(f"{name} — {preds.source}  "
                     f"({res.frac_ambiguous:.1%} ambiguous)", fontsize=10)
        fig.tight_layout()
        fig.savefig(self.out / f"pr_{preds.source}.png", dpi=150)
        plt.close(fig)

    def plot_calibration(self, preds, cal_probs, name):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")
        probs = 1 / (1 + np.exp(-preds.scores)) if preds.scores.min() < 0 else preds.scores
        fp, mp = calibration_curve(preds.labels, probs, n_bins=10)
        ax.plot(mp, fp, "s-", color=self.COLORS["model"], lw=2, label=f"{name} raw")
        if cal_probs is not None:
            fp2, mp2 = calibration_curve(preds.labels, cal_probs, n_bins=10)
            ax.plot(mp2, fp2, "^-", color=self.COLORS["debiased"],
                    lw=2, label=f"{name} calibrated")
        ax.set(xlabel="Mean predicted prob", ylabel="Fraction positives",
               title=f"Calibration — {preds.source}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(self.out / f"calibration_{preds.source}.png", dpi=150)
        plt.close(fig)

    def plot_score_dist(self, res, preds, name):
        probs = 1 / (1 + np.exp(-preds.scores)) if preds.scores.min() < 0 else preds.scores
        fig, ax = plt.subplots(figsize=(8, 4))
        bins = np.linspace(0, 1, 50)
        ax.hist(probs[preds.labels == 1], bins=bins, alpha=0.6,
                color=self.COLORS["path"], label="Pathogenic", density=True)
        ax.hist(probs[preds.labels == 0], bins=bins, alpha=0.6,
                color=self.COLORS["benign"], label="Benign", density=True)
        lo = min(res.benign_thresh_90p, res.path_thresh_90p)
        hi = max(res.benign_thresh_90p, res.path_thresh_90p)
        ax.axvspan(lo, hi, alpha=0.12, color="gray",
                   label=f"Ambiguous {res.frac_ambiguous:.1%}")
        ax.axvline(res.path_thresh_90p, color="darkred", ls="--", lw=1.5)
        ax.axvline(res.benign_thresh_90p, color="darkblue", ls="--", lw=1.5)
        ax.set(xlabel=f"{name} score", ylabel="Density",
               title=f"Score distribution — {preds.source}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(self.out / f"score_dist_{preds.source}.png", dpi=150)
        plt.close(fig)

    def plot_gene_bias(self, res, preds, name):
        df = pd.DataFrame({"gene": preds.gene_ids, "label": preds.labels})
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        fracs = df.groupby("gene")["label"].mean()
        axes[0].hist(fracs, bins=20, color=self.COLORS["model"], ec="white")
        axes[0].set(xlabel="% pathogenic per gene", title="Gene-label bias (Fig S9B)")

        metrics = {"Gene-fraction\nbaseline": res.gene_bias_auroc,
                   f"{name}\n(raw)": res.auroc,
                   f"{name}\n(debiased)": res.debiased_auroc}
        bars = axes[1].bar(list(metrics.keys()), list(metrics.values()),
                           color=[self.COLORS["bias"], self.COLORS["model"],
                                  self.COLORS["debiased"]], width=0.4)
        for b, v in zip(bars, metrics.values()):
            axes[1].text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                         ha="center", va="bottom", fontsize=9)
        axes[1].set(ylim=[0.5, 1.0], ylabel="auROC", title="auROC vs gene-bias")

        gene_counts = df.groupby("gene").size().sort_values()
        cum = gene_counts.cumsum() / gene_counts.sum()
        axes[2].plot(np.arange(1, len(gene_counts) + 1) / len(gene_counts) * 100,
                     cum * 100, color=self.COLORS["model"], lw=2)
        axes[2].set(xlabel="% genes", ylabel="% variants",
                    title="Gene coverage bias (Fig S9D)")

        fig.suptitle(f"Gene-label bias — {preds.source}", fontsize=11)
        fig.tight_layout()
        fig.savefig(self.out / f"gene_bias_{preds.source}.png", dpi=150)
        plt.close(fig)

    def plot_per_gene(self, pg: pd.DataFrame, name: str, top_n=40):
        df = pg.head(top_n)
        fig, ax = plt.subplots(figsize=(13, 4))
        sc = ax.scatter(range(len(df)), df["auroc"],
                        c=df["n_pos"] + df["n_neg"], cmap="viridis", s=55, zorder=3)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df["gene"], rotation=90, fontsize=7)
        ax.axhline(0.9, color="gray", ls="--", alpha=0.4)
        ax.set(ylabel="auROC", ylim=[0.5, 1.02],
               title=f"Per-gene auROC top {top_n} — {name}")
        ax.grid(axis="y", alpha=0.3)
        plt.colorbar(sc, ax=ax, label="n variants")
        fig.tight_layout()
        fig.savefig(self.out / "per_gene_auroc.png", dpi=150)
        plt.close(fig)

    def _save_json(self, res: EvalResult, name: str):
        d = {"model": name, "auroc": res.auroc,
             "auroc_ci": list(res.auroc_ci), "auprc": res.auprc,
             "gene_bias_auroc": res.gene_bias_auroc,
             "debiased_auroc": res.debiased_auroc,
             "brier": res.brier, "ece": res.ece,
             "frac_ambiguous": res.frac_ambiguous,
             "path_thresh": res.path_thresh_90p,
             "benign_thresh": res.benign_thresh_90p,
             "n_variants": res.n_variants, "n_genes": res.n_genes}
        (self.out / "eval_summary.json").write_text(json.dumps(d, indent=2))
