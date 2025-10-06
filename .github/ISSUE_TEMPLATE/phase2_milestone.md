---
name: Phase 2 – Bias Quantification & Baseline Evaluation
about: Milestone tracker for Phase 2 of MissenseImpact FYP
title: "Phase 2: Bias Quantification & Baseline Evaluation"
labels: ["Phase 2 - Bias Analysis", "high-priority"]
assignees: ["BilalAsifB", "MusW02", "Wander-1ife"]
---

## 🎯 Phase Objective
Quantify population and ancestry bias in existing variant effect predictors by benchmarking their performance on South Asian datasets.

📅 **Timeline:** November – December 2025  
🧩 **Leads:** Bilal (Modeling), Sami (Recalibration Design), Mustafa (Data Validation)

---

## ✅ Deliverables
- [ ] Baseline performance metrics (AUROC, sensitivity, specificity) for AlphaMissense, EVE, and REVEL on South Asian vs. European subsets  
- [ ] ClinVar misclassification analysis (Sharo et al., 2023 methodology)  
- [ ] Population-mismatched filtering test (Lee et al., 2024)  
- [ ] Statistical bias quantification (DeLong, McNemar tests)  
- [ ] Summary report + fairness visualizations (calibration plots, subgroup gaps)  
- [ ] Documentation in `02_bias_quantification.ipynb`

---

## 🧩 Task Breakdown

| Task | Description | Assignee | Due |
|------|--------------|----------|-----|
| Baseline predictor setup | Download AlphaMissense, REVEL, EVE predictor outputs for test set | Bilal | Nov 15 |
| Benchmark dataset creation | Stratify South Asian vs. European variants for fair comparison | Mustafa | Nov 15 |
| Bias quantification | Compute ancestry-specific metrics and AUROC gap analysis | Bilal | Nov 25 |
| ClinVar misclassification audit | Identify reclassification discrepancies by ancestry | Sami | Nov 30 |
| Population-mismatched test | Evaluate false positive rates using non-matched controls | Mustafa | Dec 2 |
| Statistical testing | Apply DeLong and McNemar significance tests | Sami | Dec 7 |
| Visualization & reporting | Generate calibration plots, subgroup disparity figures | Bilal | Dec 10 |
| Documentation | Write and format Phase 2 report and summary notebook | Sami (Lead) / Mustafa | Dec 15 |

---

## 🧠 Dependencies
- Phase 1 curated dataset (VEP-processed South Asian variants)  
- Access to ClinVar & gnomAD API  
- Python/R setup for AUROC comparison and statistical tests

---

## 📊 Evaluation Metrics
- [ ] AUROC/Sensitivity gap ≤ 25% between South Asian and European subsets  
- [ ] Statistically significant fairness improvement (p < 0.05)  
- [ ] Complete documentation in `02_bias_quantification.ipynb`  
- [ ] All plots and metrics reproducible via notebook run

---

## 📎 Notes / Comments
_Add weekly updates, metrics results, or blockers below._

---

**Linked Milestone:** `Phase 2 – Bias Quantification`  
**Related Project Board:** `MissenseImpact-FYP → Project #1`
