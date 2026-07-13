# Thesis Writing Plan — DILI Prediction (AIUB Template, Spring 2025-26)

Maps every template section to existing repo assets. Status: ✅ have it · ✍️ write from existing material · 🔨 needs new work.

## Front matter (quick fill-ins)
| Section | Status | What to do |
|---|---|---|
| Title page | ✍️ | Title + 4 member names/IDs. Suggested title: *"Honest Evaluation of Drug-Induced Liver Injury Prediction: Descriptor-Fingerprint Gradient Boosting Matches Deep Learning, and the TDC Benchmark Overstates Real-World Difficulty."* |
| Declaration | ✅ | Copy verbatim from template, add names/IDs/signatures. |
| Approval / Acknowledgement | ✅ | Boilerplate; fill supervisor name + thanks. |
| List of Figures / Tables / Abbreviations | 🔨 | Auto-generate after figures/tables are placed (DILI, AUROC, MCC, GNN, TDC, XGBoost, SMILES...). |
| Abstract + Keywords | ✍️ | 200–250 words. Keywords: DILI, hepatotoxicity, graph neural networks, molecular fingerprints, gradient boosting, benchmark validity, external validation. |

## Chapter 1 — Introduction  (✍️ substance exists, just write)
| Section | Source material |
|---|---|
| 1.2 Problem Statement | DILI = leading cause of drug withdrawals; need structure-based prediction. |
| 1.3 Problem Background | TDC-DILI benchmark, DILIrank/FDA labels. |
| 1.4 Research Objectives | Build + honestly evaluate structure-only DILI models; test generalization. |
| 1.5 Research Questions | RQ1 Can structure-only models match SOTA under leak-free eval? RQ2 Does complexity help on small data? RQ3 Does benchmark performance transfer to unseen chemistry? |
| 1.6 Motivations / 1.8 Significance | Patient safety + benchmark-validity critique. |
| 1.9 Research Contribution | (a) leakage-clean TDC benchmark of 6 GNNs + feature-union GBM (0.920±0.014); (b) complexity-doesn't-help finding; (c) GNNExplainer toxicophore interpretability; (d) **external validation: benchmark ceiling 0.93 vs real ~0.71**. |

## Chapter 2 — Literature Review  (🔨 the one chapter needing real outside effort)
- 2.2 Review: AttentiveFP (0.886), MapLight+GNN (0.917), AttrMasking (0.919), MolFormer-XL, ChemBERTa, MiniMol (0.956, leakage-flagged), the **Feb-2026 bioRxiv leakage audit**. Structural-alert / toxicophore literature for DILI.
- 2.3 Problem Analysis: small-data regime (n≈475), scaffold split, benchmark curation bias.
- **TODO:** collect ~20–30 APA references. This is the main reading task.

## Chapter 3 — Proposed Model  (✍️ fully have it — see README + code)
| Section | Source |
|---|---|
| 3.4 Research Methodology | `README.md` Methods; scaffold split, feature union, GBM + GNNs. |
| 3.5 System Design / Architecture | `src/improved/models.py` (GCN/GAT/GraphSAGE/GIN/MPNN + descriptor fusion); feature-union diagram (desc+Morgan+Avalon+ErG+MACCS+MolFormer). **Draw 1–2 architecture figures.** |
| 3.6 Implementation & Simulation | `tdc_official.py`, `tdc_official_v2.py`, bagged XGBoost, Optuna. |
| 3.7 Tools & Technologies | Python, PyTorch, PyG, RDKit, PyTDC, XGBoost/LightGBM/CatBoost, Optuna, transformers. |

## Chapter 4 — Implementation and Testing  (✅ STRONGEST chapter — all results exist)
| Section | Source |
|---|---|
| 4.2 System Setup | venv, Python 3.14, CPU; RDKit 2026.03, XGBoost 3.2. |
| 4.3 Implementation | feature builders + bagged XGB + GNN training. |
| 4.4 Evaluation & Testing | AUROC/ACC/F1/MCC; DeLong/Wilcoxon/McNemar + bootstrap CIs (`results/significance_*.csv`, `stats_utils.py`). |
| 4.5 **Results & Discussion** | (1) Official TDC 0.920±0.014 table; (2) 6-GNN benchmark (`final_metrics.csv`); (3) complexity ablation; (4) toxicophore interpretability (`results/explain/`); (5) **external validation** (`results/external_validation_summary.md`) — the headline discussion. Figures: `data/*_chart.png`, toxicophore PNGs, + new external-validation bar chart. |

## Chapter 5 — Standards, Constraints, Milestones  (✍️/🔨 structured writing)
- 5.1 Standards: IEEE/reproducibility; TDC benchmark protocol.
- 5.2 Sustainability: CPU-only, lightweight GBM vs heavy GNN energy cost.
- 5.3 Societal Impact: safer drug screening vs risk of over-reliance/false negatives.
- 5.4 Ethics: bias/fairness (label curation bias — ties to your external-validation finding), responsible use.
- 5.5 Security/Privacy: public data, no patient PII.
- 5.6–5.8 Constraints: small dataset (n≈475), class imbalance, CPU compute, no budget.
- 5.9–5.11 Timeline / Milestones / **Gantt chart** — 🔨 make a simple Gantt of your semester phases.

## Chapter 6 — Conclusion  (✍️ have it)
- 6.1 Summary · 6.2 Key Contributions (4 above)
- 6.3 **Limitations**: benchmark ceiling, small n, structure-only (no dose/clinical features) → `external-validation-finding`.
- 6.4 **Future Work**: MiniMol/graph-SSL embeddings, mechanism-informed proxy features, larger external sets → `dili-improvement-levers`.

## References + Appendix
- APA style. Appendix: extra tables, hyperparameters (`configs/`), code repo link.

---
## Recommended writing order (fastest path to a full draft)
1. **Front matter fill-ins** (30 min).
2. **Ch4 Results** first — you have every number; it's your strongest, easiest chapter.
3. **Ch3 Proposed Model** — draw 2 architecture figures, write around the code.
4. **Ch1 Introduction** — objectives/RQs/contributions.
5. **Ch6 Conclusion** — limitations + future work (memory notes cover these).
6. **Ch2 Literature Review** — the reading task; do in parallel from day 1.
7. **Ch5 Constraints/Ethics** — structured writing.
8. **Figures, References (APA), TOC auto-update, final formatting.**

## Figures still to make
- [ ] External-validation bar chart (TDC 0.93 vs external 0.68 vs ceiling 0.71, CI whiskers) — highest impact.
- [ ] Feature-union / model architecture diagram.
- [ ] Gantt chart (Ch5).
- [x] AUROC/ACC/F1/MCC charts (`data/`), toxicophore maps (`results/explain/`).
