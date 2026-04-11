# MissenseImpact: Population-Aware Genomic Models for Missense Variant Effect Prediction

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## 📌 Overview

**MissenseImpact** is a Final Year Project (FYP) focused on developing **population-aware variant effect predictors (VEPs)** for genomic data, with special emphasis on **South Asian populations**.

Missense variants — single amino acid substitutions in proteins — are among the most common genetic alterations in humans. They are central to understanding genetic diseases, cancer susceptibility, and precision medicine. However, most existing computational predictors (PolyPhen-2, SIFT, REVEL, EVE, AlphaMissense) are trained primarily on **European ancestry datasets**, which causes systematic misclassification of variants from underrepresented groups such as South Asians.

Our project addresses this bias by adapting state-of-the-art **protein language models (PLMs)** (specifically ESM-1b) to South Asian genomic data, enabling more accurate and fair variant interpretation.

---

## 🎯 Objectives

* **Bias Quantification**: Evaluate existing predictors on South Asian datasets and document performance gaps across ancestries.
* **Model Adaptation**: Apply allele frequency–based recalibration, logistic and Bayesian methods, and fine-tuning of PLMs (ESM-1b) to South Asian cohorts.
* **Evaluation Framework**: Develop a benchmarking suite that measures accuracy, calibration, subgroup fairness, and clinical utility.
* **Prototype Clinical Pipeline**: Build a modular, research-ready pipeline that integrates adapted predictors with automated bias audits and ACMG/AMP guideline alignment.

---

## 🏗️ System Architecture

![System Architecture](SystemArch-MissenseImpact.png)

The system is a fully modular machine learning and bioinformatics pipeline consisting of four major subsystems:
1. **Data Pipeline**: Ingests VCFs from multiple cohorts (gnomAD, SG10K, IndiGen, 1kG), filters for MANE-select transcripts, fetches Uniprot sequences, and applies position-aware train/val/test splitting to prevent data leakage.
2. **Model Architecture**: Utilizes an ESM-1b Protein Language Model backbone to extract representations for reference and alternate sequences. A custom fusion mechanism projects and differences these embeddings before pathogenic classification.
3. **Training & Optimization**: Employs an AlphaMissense-derived clipped sigmoid cross-entropy loss, step-wise layer unfreezing, exponential moving average (EMA) smoothing, and Optuna-based hyperparameter tuning.
4. **Evaluation Suite**: Generates South Asian-specific precision-recall thresholds, produces robust bias quantification reports vs. European baselines, and benchmarks on curated sets (ClinVar, Cancer Hotspots, De Novo).

---

## 🚀 Quick Start

### 1. Installation

```bash
pip install -e .
```

### 2. General Workflow

**Step 1: Build training data from annotated VCFs**
Process population VCFs, extract Allele Frequencies, and structure ML-ready datasets:
```bash
python scripts/build_training_data.py \
    --gnomad_dir   /path/gnomad_pure_sas_annotated/ \
    --sg10k_dir    /path/sg10k_annotated/ \
    --indigen_dir  /path/indigen_annotated/ \
    --thousandg_dir /path/1k_annotated/ \
    --output_dir   data/processed/
```

**Step 2: Train the ESM-Missense Model**
Train a population-aware model with position-aware splits:
```bash
python scripts/train.py \
    --train_csv data/processed/train.csv \
    --val_csv   data/processed/val.csv \
    --model_id  0 \
    --save_dir  checkpoints/
```

**Step 3: Evaluate & Benchmark**
Calibrate and evaluate against established holdouts (ClinVar, de novo, hotspot):
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/model0_best.pt \
    --benchmark_dir benchmarks/ \
    --val_csv data/processed/val.csv
```

---

## 📂 Extensive Repository Layout

| Directory / File | Description |
|------------------|-------------|
| **`configs/`** | YAML configurations for defaults to be shared across tuning and training. (`base.yaml`) |
| **`data/`** | Data ingestion, formulation, and representation mapping. |
| ├── `post_vep.py` | VCF processing, MANE selectivity filtering, and Uniprot symbol mapping. |
| ├── `pipeline.py` | Implementation of the `ProteinVariant` dataclass and ESM-1b sub-sequence tokenization. |
| ├── `dataset.py` | PyTorch map-style `Dataset` with collate function for sequence handling. |
| └── `splits.py` | Strict position-aware dataset splitting to guarantee uncontaminated cross-validation. |
| **`model/`** | Deep Learning formulation for variant effect modeling. |
| ├── `backbone.py` | Pre-trained `ESMBackbone` wrapper with dynamic layer-freezing mechanics. |
| ├── `fusion.py` | Handles distinct projection and symmetric differences between variant and reference representations. |
| ├── `head.py` | Final pathogenic classification MLP. |
| └── `esm_missense.py` | Full combined ESM-Missense pipeline container module. |
| **`training/`** | Custom logic for convergence and robustness. |
| ├── `loss.py` | AM clipped sigmoid cross-entropy, placing loss ceilings on noisy labels. |
| └── `trainer.py` | Main event loop with early stopping, warmup progression, EMA, and regular logging. |
| **`evaluation/`** | Comprehensive algorithmic auditing and validation. |
| ├── `threshold_calibration.py` | Recalibrates Benign/Pathogenic bounds locally for South Asian populations. |
| ├── `bias_report.py` | Emits comparative statistics illustrating European threshold misalignments on SAS cohorts. |
| └── `benchmark.py` | Systematic execution pipeline assessing datasets (e.g. ClinVar, Cancer Hotspots). |
| **`tuning/`** | Optuna script for sequential hyper-parameter sweep. (`optuna_tuner.py`) |
| **`scripts/`** | Accessible CLI entrypoints mapping to pipeline actions. |

---

## 👨‍💻 Team

* **Bilal Asif Burney (Lead)**
* **Sami Ur Rehman**
* **Mustafa Waqar**

---

## 📄 License

This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for more details.

---

## ✅ Impact

This project will reduce ancestry bias in genomic variant interpretation, leading to fairer and more reliable predictions for South Asian patients. It contributes to **equitable genomic medicine** while advancing methodological research in AI-driven bioinformatics.
