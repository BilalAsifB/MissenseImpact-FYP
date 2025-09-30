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

## 👨‍💻 Team

* **Bilal Asif Burney (Lead)**
* **Sami Ur Rehman**
* **Mustafa Waqar**

---

## ✅ Impact

This project will reduce ancestry bias in genomic variant interpretation, leading to fairer and more reliable predictions for South Asian patients. It contributes to **equitable genomic medicine** while advancing methodological research in AI-driven bioinformatics.
