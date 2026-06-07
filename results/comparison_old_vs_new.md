# DILI GNN Improvement Comparison

**Date:** 2026-05-18
**Run:** `retrain_only --seeds 5 --final-epochs 100`

## What changed (architecture)

1. **Atom features**: 53 → 54 dimensions
   - Added per-atom **Gasteiger partial charge** (chemically meaningful for DILI: polar atoms drive reactive metabolite formation, mitochondrial uncoupling)

2. **Global molecular descriptors** (NEW): 10-dimensional per-molecule vector attached to each graph
   - MolLogP (lipophilicity)
   - MolWt
   - TPSA (topological polar surface area)
   - NumHDonors, NumHAcceptors
   - NumRotatableBonds
   - NumAromaticRings, NumAliphaticRings
   - FractionCSP3
   - HeavyAtomCount

3. **Descriptor fusion**: a small MLP lifts the 10-d descriptor into the model's hidden space; the result is concatenated with the pooled graph embedding *before* the FC classifier head. All 5 GNNs (GCN, GAT, GraphSAGE, GIN, MPNN) now have end-to-end access to global physicochemical context — the same features that make classical RF competitive on this dataset.

## Results — 5-seed mean ± std (sorted by AUROC)

| Model | AUROC (old) | AUROC (new) | ΔAUROC | MCC (old) | MCC (new) | ΔMCC |
|---|---:|---:|---:|---:|---:|---:|
| **GraphSAGE** | 0.6883 ± 0.0194 | **0.6926 ± 0.0134** | **+0.0043** | 0.2651 | 0.2296 | −0.0355 |
| **GIN** | 0.6875 ± 0.0128 | **0.6913 ± 0.0115** | **+0.0038** | 0.2165 | **0.2600** | **+0.0435** |
| **MPNN** | 0.6566 ± 0.0285 | **0.6785 ± 0.0310** | **+0.0219** | 0.1928 | **0.2474** | **+0.0546** |
| **GAT** | 0.6637 ± 0.0103 | **0.6769 ± 0.0099** | **+0.0132** | 0.2039 | 0.2269 | +0.0230 |
| **GCN** | 0.6754 ± 0.0167 | 0.6739 ± 0.0199 | −0.0015 | 0.2561 | 0.2279 | −0.0282 |

**Summary:** 4/5 models improved on AUROC. Largest gain on MPNN (+0.022) — the model that previously lost most by ignoring global context. AUROC variance shrank for 3/5 models (more stable across seeds).

## Threshold-tuned (val-MCC optimum) — best operating point per model

| Model | AUROC (old) | AUROC (new) | MCC (old) | MCC (new) |
|---|---:|---:|---:|---:|
| GraphSAGE | 0.7102 | **0.7117** | **0.3326** | 0.2302 |
| GIN | 0.7103 | 0.7087 | 0.2659 | 0.2504 |
| MPNN | 0.6649 | **0.6784** | 0.2157 | **0.1632** *(worse)* |
| GAT | 0.6756 | **0.6878** | 0.2224 | 0.2391 |
| GCN | 0.6928 | **0.6936** | 0.2536 | 0.2740 |
| **GNN-Ensemble** | 0.7037 | **0.7105** | 0.2409 | **0.2768** |

**Ensemble result is the headline:** AUROC 0.7105, MCC 0.2768 — both better than old (+0.007 AUROC, +0.036 MCC).

## Comparison to classical baseline (the bar)

| Model | AUROC |
|---|---:|
| Random Forest (FP+Mordred, random oversample) | 0.7259 |
| **GNN-Ensemble (new, threshold-tuned)** | **0.7105** |
| GraphSAGE (new, threshold-tuned) | 0.7117 |

The GNN ensemble now sits ~1.5 percentage points below classical RF — closer than before but classical still leads. The 10-fold scaffold-CV results (running next) will give a more defensible comparison, since the current 60/20/20 single scaffold split may favor whichever model happens to fit that one split better.

## What's still pending

- [ ] 10-fold scaffold-grouped CV (running) → `results/cv_summary.csv`
- [ ] GNNExplainer toxophore heatmaps (running) → `results/explain/toxophore_*.png`
- [ ] Update `results/summary.txt` with comparison once all jobs finish
