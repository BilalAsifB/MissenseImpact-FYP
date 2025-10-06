---
name: Phase 1 – Dataset Collection & Preprocessing
about: Milestone tracker for Phase 1 of MissenseImpact FYP
title: "Phase 1: Dataset Collection, Curation & Preprocessing"
labels: ["phase-1", "data-preprocessing", "high-priority"]
assignees: ["BilalAsifB", "MusW02", "Wander-1ife"]
---

## 🎯 Phase Objective
Build a high-quality, South Asian–focused missense variant dataset ready for bias quantification and model adaptation.

📅 **Timeline:** October – November 2025  
🧩 **Leads:** Mustafa (Data), Bilal (Pipeline), Sami (Documentation)

---

## ✅ Deliverables
- [ ] Curated South Asian Variant Database (gnomAD-SAS + ClinVar + GenomeAsia100K + IndiGen)
- [ ] Preprocessing scripts under `src/preprocessing/`
- [ ] Ensembl VEP annotation reports & QC logs
- [ ] DVC-tracked dataset with storage backend (Google Drive / B2)
- [ ] Notebook: `01_data_preprocessing.ipynb`
- [ ] Wiki section documenting dataset sources & schema

---

## 🧩 Task Breakdown

| Task | Description | Assignee | Due |
|------|--------------|----------|-----|
| Literature & dataset review | Review genomic dataset formats & allele frequency fields | Mustafa (Lead) / Sami | Oct 10 |
| Data acquisition | Download gnomAD-SAS, GenomeAsia100K, IndiGen, ClinVar | Mustafa | Oct 15 |
| Data normalization | Run Ensembl VEP; filter multi-allelic & low-quality variants | Mustafa (Lead) / Sami | Oct 25 |
| South Asian AF extraction | Extract AF_SAS from all datasets | Mustafa | Oct 25 |
| Variant filtering pipeline | Implement population-matched filtering (Lee et al. 2024) | Bilal | Nov 5 |
| QC & schema unification | Align columns (variant_id, gene, AF_SAS, consequence) | Bilal | Nov 10 |
| Documentation | Write preprocessing logs & dataset wiki page | Sami | Nov 15 |
| Verification & versioning | Validate dataset & commit via DVC pipeline | Bilal | Nov 20 |

---

## 🧠 Dependencies
- Ensembl VEP installed locally
- Access to gnomAD, GenomeAsia, and ClinVar APIs
- DVC linked to cloud storage (Backblaze B2)

---

## 📊 Evaluation Metrics
- [ ] ≥ 90% of target variants successfully annotated  
- [ ] Unified schema validated across all datasets  
- [ ] DVC pipeline reproducible via notebook run  
- [ ] Documentation reviewed and merged to `main`

---

## 📎 Notes / Comments
_Add weekly updates, blockers, and validation results below._

---

**Linked Milestone:** `Phase 1 – Dataset Collection & Preprocessing`  
**Related Project Board:** `MissenseImpact-FYP → Phase 1`
