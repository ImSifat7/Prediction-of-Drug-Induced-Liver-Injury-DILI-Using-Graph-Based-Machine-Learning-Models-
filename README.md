# Prediction of Drug-Induced Liver Injury (DILI) Using Graph-Based Machine Learning Models

A machine-learning pipeline for predicting **Drug-Induced Liver Injury (DILI)** directly from molecular structure (SMILES). The project benchmarks five Graph Neural Networks — **GCN, GAT, GraphSAGE, GIN, MPNN** (plus AttentiveFP) — and a molecular-descriptor + fingerprint **gradient-boosting** model, evaluated under the **official Therapeutics Data Commons (TDC) DILI benchmark protocol** for leaderboard-comparable, leakage-free results.

## Highlights

- **Official TDC-DILI benchmark** (`admet_group`): fixed held-out test set, **5 seeds**, mean ± std AUROC — directly comparable to the public leaderboard.
- **Leakage-clean evaluation** — every model/feature/threshold decision is made on the validation split; the test set is touched only to report the final number.
- **Leakage-free headline: test AUROC 0.886 ± 0.020** (the **validation-selected** model under the official 5-seed protocol) — ties the *AttentiveFP* leaderboard entry (0.886). Some configurations reach **~0.92 on the test set**, but they are *not* the ones selected on validation, so headlining them would be test-set peeking; the wide 95 % CI (≈ [0.82, 0.95], n = 96) makes every config statistically tied.
- **6 graph architectures** (GCN/GAT/GraphSAGE/GIN/MPNN/AttentiveFP) + **MapLight-style feature union** (RDKit descriptors + Morgan/Avalon/ErG/MACCS fingerprints + MolFormer-XL embeddings).
- **4 gradient-boosting backends** (XGBoost, LightGBM, CatBoost, HistGB) with Optuna tuning and a stacking / averaging ensemble.
- **Statistical rigor** — DeLong / Wilcoxon / McNemar tests + 95 % bootstrap confidence intervals.
- **Honest finding** — added complexity (richer fingerprints, foundation-model embeddings, hyperparameter tuning) does **not** beat a simple descriptor+fingerprint model on this small dataset; all configurations are within the bootstrap CI.
- **Mechanistic interpretability** — known reactive-metabolite toxicophores (sulfonamide, aromatic amine, nitroaromatic, furan, N-oxide) are **significantly enriched** among DILI-positive drugs (Fisher exact *p* < 0.05), linking model behaviour to established hepatotoxicity mechanisms — even though, being already encoded by fingerprints, they add no significant AUROC.
- **Multi-dataset generalization** — evaluated on **3 independent DILI datasets** (TDC-DILI, DILIrank, DILIst, InChIKey-deduplicated). The model scores 0.89 on TDC-DILI but only **0.66–0.74** on the larger sets, and merging 1,294 extra DILI molecules does **not** help — evidence that TDC-DILI is an easy, saturated slice and the bottleneck is the benchmark, not the model.

## Dataset

**Primary benchmark — TDC-DILI** (`tdc.benchmark_group.admet_group`):
- 475 molecules · official **scaffold split** · fixed **96-molecule** test set · **5 seeds**.
- Aggregated by the FDA's National Center for Toxicological Research (via AstraZeneca).

**Exploratory set — DILIrank** (`data/DILIrank2.xlsx`):
- After cleaning + SMILES standardisation: **n = 869** (521 train / 173 val / 175 test, Bemis–Murcko scaffold split). Used for the initial architecture exploration.

## Results

### 1. Official TDC-DILI benchmark (`admet_group`, 5 seeds, fixed 96-mol test)

Leakage-clean and leaderboard-comparable. The protocol requires choosing the model on the **validation** split and scoring the held-out **test** set only once. The table is therefore sorted by **validation** AUROC — the score we are *allowed* to select on — **not** by test AUROC.

| Features | Model | Val AUROC | Test AUROC |
|----------|-------|-----------|------------|
| + Avalon / ErG / MACCS (+ MolFormer-XL) | **LightGBM** | **0.869** | 0.892 ± 0.020 |
| + Avalon / ErG / MACCS | Ensemble (XGB+LGBM+CatBoost) | 0.865 | 0.909 ± 0.017 |
| + Avalon / ErG / MACCS + MolFormer-XL | XGBoost | 0.864 | 0.910 ± 0.013 |
| + Avalon / ErG / MACCS | XGBoost | 0.862 | 0.908 ± 0.017 |
| RDKit desc + Morgan | Ensemble (XGB+LGBM+CatBoost) | 0.843 | 0.920 ± 0.014 |
| RDKit desc + Morgan | XGBoost | 0.841 | 0.919 ± 0.021 |

**→ Final reported model** (validation-selected, Optuna-tuned, bagged over 5 seeds): **`rich_mf / LightGBM` — test AUROC 0.886 ± 0.020**, 95 % CI [0.82, 0.95].

> **Interpretation — why the headline is 0.886, not 0.920.** The `RDKit desc + Morgan` ensemble has the highest *test* AUROC (0.920) but ranks only **#7 on validation**. Choosing it would mean using the test set to make a modelling decision — *data leakage* / test-set peeking. Under honest validation-based selection the chosen model is `rich_mf / LightGBM`, whose test AUROC is **0.886** (ties AttentiveFP on the leaderboard). The 95 % bootstrap CI on the test AUROC spans ≈ **[0.82, 0.95]** (n = 96), so 0.886 and 0.920 are statistically indistinguishable in any case. Richer fingerprints, MolFormer-XL embeddings and Optuna tuning raise *validation* but not *test* — they overfit the tiny (~48-molecule) validation split. Reference leaderboard (reproducible): AttentiveFP 0.886 · MapLight+GNN 0.917 · AttrMasking 0.919.

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

### 3. Mechanism-informed structural alerts — ablation & interpretability

We added 31 **mechanism-informed structural-alert features** — known reactive-metabolite / hepatotoxicity toxicophores (nitroaromatic, aromatic amine, sulfonamide, furan, Michael acceptor, quinone, …) plus PAINS / Brenk / NIH filter catalogs — and re-ran the official val-selected protocol ([`src/improved/tdc_alerts.py`](src/improved/tdc_alerts.py), [`tdc_alerts_rigor.py`](src/improved/tdc_alerts_rigor.py)).

- **Predictively, they add nothing significant**: mean ΔAUROC **+0.002**, paired Wilcoxon **p = 0.12**, DeLong **p > 0.37**. The Morgan fingerprint already encodes these substructures, so explicit alerts are redundant (0.4 % of model importance). This is a *third independent confirmation* that ~0.89–0.92 is the dataset ceiling.
- **But they are mechanistically informative**: several toxicophores are **significantly enriched** among DILI-positive drugs (Fisher exact, train/val) vs the 49 % base rate — sulfonamide 96 %, nitroaromatic 91 %, N-oxide 91 %, furan 100 %, aromatic amine 79 %, aromatic halide 67 % (all *p* < 0.05). See [`results/tdc_alerts_interpret.csv`](results/tdc_alerts_interpret.csv).

> **Takeaway:** the value of structural alerts here is **interpretability, not accuracy** — the simple validation-selected model (AUROC 0.886) is still the one to report.

### 4. Generalization across multiple DILI datasets

We evaluated the selected model on **three independent DILI benchmarks**, standardized to canonical SMILES + binary label and **deduplicated by InChIKey** to prevent cross-dataset leakage ([`src/improved/multi_dataset.py`](src/improved/multi_dataset.py)): **TDC-DILI** (475), **DILIrank** (881), **DILIst** (1096, [sourced](data/external/)).

Cross-dataset AUROC (train row → test column; diagonal = 5-fold CV):

| train ↓ / test → | TDC-DILI | DILIrank | DILIst |
|---|---|---|---|
| **TDC-DILI** | **0.892** | 0.664 | 0.635 |
| **DILIrank** | 0.819 | **0.739** | 0.698 |
| **DILIst** | 0.669 | 0.705 | **0.664** |

- **TDC-DILI is an easy slice**: the model scores 0.89 within TDC but only **0.66–0.74** within the larger DILIrank/DILIst — and TDC-trained models **do not transfer** (drop to ~0.64). The headline benchmark number overstates real-world DILI difficulty.
- **More data does not help**: merging DILIrank + DILIst (1,294 extra molecules, leakage-filtered) into training changed TDC test AUROC by **−0.02** (DeLong *p* = 0.63, not significant). The benchmark is saturated — see [`results/multi_dataset_crossval.csv`](results/multi_dataset_crossval.csv), [`multi_dataset_merged.csv`](results/multi_dataset_merged.csv).

### 5. Hyperparameter optimization (Optuna) and model selection

- **Optimizer** — the GNNs train with **Adam**; the gradient-boosting models are tuned with **Optuna** (TPE, [`src/improved/optimizer_demo.py`](src/improved/optimizer_demo.py)). Optuna raised *validation* AUROC 0.841 → 0.860 (top drivers: `n_estimators`, `colsample`, `learning_rate`), but the gain does **not** reach test — small-validation-set overfitting. See [`results/optuna_history.png`](results/optuna_history.png).
- **Selected model** — **RDKit descriptors + Morgan → gradient boosting (XGBoost/LightGBM)**, chosen on validation AUROC + parsimony + robustness; **AttentiveFP** is the best graph model. Reported leakage-free headline: **AUROC 0.886 ± 0.020**. Across every experiment the **dataset, not the model, is the bottleneck**.

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
