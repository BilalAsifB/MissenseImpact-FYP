# MissenseImpact: Population-Aware Genomic Models for Missense Variant Effect Prediction

## 📌 Overview

**MissenseImpact** is a Final Year Project (FYP) focused on developing **population-aware variant effect predictors (VEPs)** for genomic data, with special emphasis on **South Asian populations**.

Missense variants — single amino acid substitutions in proteins — are among the most common genetic alterations in humans. They are central to understanding genetic diseases, cancer susceptibility, and precision medicine. However, most existing computational predictors (PolyPhen-2, SIFT, REVEL, EVE, AlphaMissense) are trained primarily on **European ancestry datasets**, which causes systematic misclassification of variants from underrepresented groups such as South Asians.

Our project addresses this bias by adapting state-of-the-art **protein language models (PLMs)** to South Asian genomic data, enabling more accurate and fair variant interpretation.

---

## 🎯 Objectives

* **Bias Quantification**
  Evaluate existing predictors on South Asian datasets and document performance gaps across ancestries.

* **Model Adaptation**
  Apply allele frequency–based recalibration, logistic and Bayesian methods, and fine-tuning of PLMs (ESM-1b, ProtBERT, AlphaMissense) to South Asian cohorts.

* **Evaluation Framework**
  Develop a benchmarking suite that measures accuracy, calibration, subgroup fairness, and clinical utility.

* **Prototype Clinical Pipeline**
  Build a modular, research-ready pipeline that integrates adapted predictors with automated bias audits and ACMG/AMP guideline alignment.

---

## 📦 Deliverables

* A curated **South Asian genomic dataset** prepared from open sources (gnomAD-SAS, GenomeAsia100K, IndiGen, ClinVar).
* Adapted **population-aware VEPs** fine-tuned for South Asian data.
* A **bias-aware evaluation suite** with reproducible analyses, visualizations, and fairness audits.
* A **prototype annotation pipeline** demonstrating clinical readiness.
* Final **research dissertation and documentation** with results and case studies.

---

## Quick start

```bash
pip install -e .

# 1. Build training data from annotated VCFs
python scripts/build_training_data.py \
    --gnomad_dir   /path/gnomad_pure_sas_annotated/ \
    --sg10k_dir    /path/sg10k_annotated/ \
    --indigen_dir  /path/indigen_annotated/ \
    --thousandg_dir /path/1k_annotated/ \
    --output_dir   data/processed/

# 2. Train model 0 of 3
python scripts/train.py \
    --train_csv data/processed/train.csv \
    --val_csv   data/processed/val.csv \
    --model_id  0 \
    --save_dir  checkpoints/

# 3. Evaluate
python scripts/evaluate.py \
    --checkpoint checkpoints/model0_best.pt \
    --benchmark_dir benchmarks/ \
    --val_csv data/processed/val.csv
```

---

## Repo layout

| Path | Purpose |
|---|---|
| `data/pipeline.py` | `ProteinVariant` dataclass + ESM-1b tokenisation |
| `data/dataset.py` | `SASVariantDataset` + `collate_variants` |
| `data/post_vep.py` | VCF → training CSV (all 4 sources) |
| `data/splits.py` | Position-aware train/val/test splitting |
| `model/backbone.py` | ESM-1b wrapper with configurable freezing |
| `model/fusion.py` | Ref/alt projection + difference |
| `model/head.py` | Pathogenicity MLP |
| `model/esm_missense.py` | Top-level model |
| `training/loss.py` | AM clipped sigmoid cross-entropy |
| `training/trainer.py` | EMA + training loop + checkpointing |
| `evaluation/` | auROC, calibration, gene-bias, MAVE metrics |
| `tuning/` | 3-phase Optuna hyperparameter search |
| `scripts/` | Runnable entry points |
| `configs/` | YAML hyperparameter configs |

---

## 👨‍💻 Team

* **Bilal Asif Burney (Lead)**
* **Sami Ur Rehman**
* **Mustafa Waqar**

---

## ✅ Impact

This project will reduce ancestry bias in genomic variant interpretation, leading to fairer and more reliable predictions for South Asian patients. It contributes to **equitable genomic medicine** while advancing methodological research in AI-driven bioinformatics.
