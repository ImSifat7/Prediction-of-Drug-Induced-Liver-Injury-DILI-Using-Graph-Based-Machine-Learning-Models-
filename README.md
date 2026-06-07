# Prediction of Drug-Induced Liver Injury (DILI) Using Graph-Based Machine Learning Models

A machine-learning pipeline for predicting **Drug-Induced Liver Injury (DILI)** directly from molecular structure (SMILES). The project benchmarks five Graph Neural Networks — **GCN, GAT, GraphSAGE, GIN, MPNN** (plus AttentiveFP) — and a molecular-descriptor + fingerprint **gradient-boosting** model, evaluated under the **official Therapeutics Data Commons (TDC) DILI benchmark protocol** for leaderboard-comparable, leakage-free results.

## Highlights

- **Official TDC-DILI benchmark** (`admet_group`): fixed held-out test set, **5 seeds**, mean ± std AUROC — directly comparable to the public leaderboard.
- **Leakage-clean evaluation** — every model/feature/threshold decision is made on the validation split; the test set is touched only to report the final number.
- **Top-tier result: AUROC 0.920 ± 0.014**, on par with the strongest *reproducible* leaderboard methods (MapLight+GNN 0.917, AttrMasking 0.919).
- **6 graph architectures** (GCN/GAT/GraphSAGE/GIN/MPNN/AttentiveFP) + **MapLight-style feature union** (RDKit descriptors + Morgan/Avalon/ErG/MACCS fingerprints + MolFormer-XL embeddings).
- **4 gradient-boosting backends** (XGBoost, LightGBM, CatBoost, HistGB) with Optuna tuning and a stacking / averaging ensemble.
- **Statistical rigor** — DeLong / Wilcoxon / McNemar tests + 95 % bootstrap confidence intervals.
- **Honest finding** — added complexity (richer fingerprints, foundation-model embeddings, hyperparameter tuning) does **not** beat a simple descriptor+fingerprint model on this small dataset; all configurations are within the bootstrap CI.

## Dataset

**Primary benchmark — TDC-DILI** (`tdc.benchmark_group.admet_group`):
- 475 molecules · official **scaffold split** · fixed **96-molecule** test set · **5 seeds**.
- Aggregated by the FDA's National Center for Toxicological Research (via AstraZeneca).

**Exploratory set — DILIrank** (`data/DILIrank2.xlsx`):
- After cleaning + SMILES standardisation: **n = 869** (521 train / 173 val / 175 test, Bemis–Murcko scaffold split). Used for the initial architecture exploration.

## Results

### 1. Official TDC-DILI benchmark (`admet_group`, 5 seeds, fixed 96-mol test)

Leakage-clean, leaderboard-comparable. Models selected on **validation**; test AUROC reported for the selected configurations.

| Features | Model | Test AUROC |
|----------|-------|------------|
| RDKit desc + Morgan | **Ensemble (XGB + LGBM + CatBoost)** | **0.920 ± 0.014** |
| RDKit desc + Morgan | XGBoost | 0.919 ± 0.021 |
| + Avalon / ErG / MACCS + MolFormer-XL | CatBoost | 0.911 ± 0.017 |
| + Avalon / ErG / MACCS + MolFormer-XL | XGBoost | 0.910 ± 0.013 |
| + Avalon / ErG / MACCS | XGBoost | 0.908 ± 0.017 |
| + Avalon / ErG / MACCS + MolFormer-XL (val-selected, Optuna-tuned) | LightGBM | 0.886 ± 0.019 |

> **Interpretation.** The 95 % bootstrap CI on the test AUROC spans ≈ **[0.82, 0.95]** (n = 96), so every row above is statistically indistinguishable. Richer fingerprints, MolFormer-XL embeddings, and Optuna tuning improve *validation* but not *test* — they overfit the small (~48-molecule) validation split. The parsimonious descriptor + Morgan model is therefore the preferred configuration. Reference leaderboard (reproducible): AttentiveFP 0.886 · MapLight+GNN 0.917 · AttrMasking 0.919.

### 2. Graph neural network architectures (TDC-DILI, 5-fold scaffold CV)

The graph-based models that motivate the project title, compared on the TDC-DILI dataset:

| Model | AUROC | ACC | F1 | MCC |
|-------|-------|-----|----|-----|
| AttentiveFP        | 0.871 ± 0.037 | 0.805 | 0.785 | 0.599 |
| Rank-avg ensemble  | 0.869 ± 0.035 | 0.826 | 0.819 | 0.644 |
| GCN                | 0.861 ± 0.043 | 0.790 | 0.784 | 0.572 |
| GIN                | 0.859 ± 0.024 | 0.792 | 0.797 | 0.587 |
| GAT                | 0.859 ± 0.049 | 0.812 | 0.785 | 0.595 |
| GraphSAGE          | 0.842 ± 0.033 | 0.763 | 0.741 | 0.506 |
| MPNN               | 0.838 ± 0.046 | 0.797 | 0.792 | 0.584 |

> The earlier DILIrank-split exploration (AUROC ≈ 0.69–0.72 per GNN) is retained in `results/` for reference; it used a harder home-made split and is superseded by the benchmark results above.

See [`results/tdc_official_search.csv`](results/tdc_official_search.csv) and [`results/tdc_official_final.csv`](results/tdc_official_final.csv) for the full official-benchmark numbers, and [`results/summary.txt`](results/summary.txt) for the complete report.

## Quick start

### 1. Environment

```bash
python -m venv venv
venv\Scripts\activate              # Windows  (source venv/bin/activate on Linux/macOS)
pip install -r requirments.txt

# Official TDC benchmark API (on Python 3.14, install without the pinned deps):
pip install PyTDC --no-deps fuzzywuzzy
```

### 2. Run the official benchmark

```bash
# MapLight-style feature union -> bagged XGBoost, official 5-seed protocol
python -m src.improved.tdc_official

# Full model search: 4 GBM backends x 3 feature sets + ensemble + Optuna
# (all model selection done on validation, not test)
python -m src.improved.tdc_official_v2 --optuna 30
```

### 3. Graph neural networks / earlier pipeline

```bash
python -m src.improved.run_pipeline --n-trials 25 --seeds 5   # 5 GNNs + stacking
python -m src.improved.tdc_cv5                                 # 5-fold scaffold CV on TDC-DILI
python -m src.improved.classical_ml                           # classical ML baselines
```

## Repository structure

```
dili_project/
├── data/                 # DILIrank dataset + cleaned CSVs  (TDC benchmark auto-downloaded, gitignored)
├── configs/              # Best / grid / Optuna hyperparameters per model (JSON)
├── results/              # Metrics (CSV), logs, ROC curves, significance tests, summary.txt
└── src/improved/
    ├── data_utils.py         # Scaffold split + atom/bond features + RDKit descriptors
    ├── models.py             # GCN / GAT / GraphSAGE / GIN / MPNN (+ descriptor fusion)
    ├── tdc_official.py        # Official admet_group harness (feature union -> XGBoost)
    ├── tdc_official_v2.py     # Official harness: 4-GBM x feature-set search + Optuna
    ├── tdc_pipeline.py        # End-to-end TDC pipeline (GNNs + MolFormer + ChemBERTa + stack)
    ├── tdc_cv5.py             # 5-fold scaffold cross-validation
    ├── tdc_phase3.py          # Ablation + DeLong tests + calibration analysis
    ├── molformer.py           # MolFormer-XL frozen-embedding extraction
    ├── chemberta.py           # ChemBERTa transformer baseline
    ├── stats_utils.py         # DeLong / Wilcoxon / McNemar / bootstrap CI
    ├── classical_ml.py        # XGBoost / RF / LR / SVM baselines
    ├── stack_final.py         # Stacking meta-learner
    └── run_pipeline.py        # Orchestrator
```

## Methods (short)

- **Official protocol** — TDC `admet_group` scaffold split, 5 seeds, `group.evaluate_many` for mean ± std AUROC; no test-set peeking.
- **Feature union (MapLight-style)** — RDKit 2D descriptors + Morgan/Avalon/ErG/MACCS fingerprints + MolFormer-XL embeddings, fed to gradient boosting.
- **Scaffold split** — molecules sharing a Bemis–Murcko scaffold stay in one split, forcing generalisation to novel chemistry.
- **Class imbalance** — `scale_pos_weight` (GBMs) / `pos_weight` in `BCEWithLogitsLoss` (GNNs).
- **Model selection** — all hyperparameter/feature/threshold choices made on validation AUROC; test reported once for the selected model.
- **Statistical tests** — DeLong (AUROC), Wilcoxon (paired across seeds), McNemar (per-sample), 95 % bootstrap CIs.

## Tech stack

`Python 3` · `PyTorch` · `PyTorch Geometric` · `RDKit` · `PyTDC` · `scikit-learn` · `XGBoost` · `LightGBM` · `CatBoost` · `Optuna` · `transformers` (MolFormer-XL / ChemBERTa) · `pandas` · `numpy` · `matplotlib`

## Author

**Md Sifat Hosen**

If you use this code or find it useful for your research, please cite the repository.
