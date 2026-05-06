# Prediction of Drug-Induced Liver Injury (DILI) Using Graph-Based Machine Learning Models

A graph-based machine learning pipeline for predicting **Drug-Induced Liver Injury (DILI)** directly from molecular structure (SMILES). The project trains and compares five Graph Neural Networks — **GCN, GAT, GraphSAGE, GIN, and MPNN** — and combines them with classical ML baselines (XGBoost / RF / LR / SVM on Morgan fingerprints + RDKit descriptors) through a stacking meta-learner for maximum accuracy.

## Highlights

- **5 GNN architectures** trained and benchmarked on the same scaffold-split dataset.
- **Bemis–Murcko scaffold split** to prevent data leakage between train / val / test.
- **Hyperparameter optimization** with both Grid Search and Optuna (TPE sampler).
- **Class imbalance** handled via `pos_weight` in `BCEWithLogitsLoss`.
- **Statistical significance testing** between models (DeLong, Wilcoxon, McNemar).
- **Stacked ensemble** (GNNs + classical ML) with a logistic-regression meta-learner.
- Full reproducibility — multi-seed runs with mean ± std and 95 % bootstrap CIs.

## Repository structure

```
dili_project/
├── data/                # DILIrank dataset, cleaned CSVs, charts
├── configs/             # Best/grid/optuna hyperparams per model (JSON)
├── results/             # Metrics, ROC curves, significance tests, summary.txt
├── src/
│   ├── step1_check_dataset.py … step10_train_mpnn.py   # Step-by-step pipeline
│   ├── plot_results.py
│   ├── final_results_table.py
│   └── improved/        # Upgraded pipeline (tuning, stats, stacking)
│       ├── data_utils.py        # Scaffold split + atom features
│       ├── models.py            # GCN / GAT / GraphSAGE / GIN / MPNN
│       ├── tune.py              # Grid + Optuna search
│       ├── train_utils.py       # Class-imbalance-aware training loop
│       ├── cv_utils.py          # Parity tables across splits
│       ├── stats_utils.py       # DeLong, Wilcoxon, McNemar, bootstrap CI
│       ├── classical_ml.py      # XGBoost / RF / LR / SVM baselines
│       ├── chemberta.py         # ChemBERTa transformer baseline
│       ├── stack.py             # Stacking meta-learner
│       └── run_pipeline.py      # End-to-end orchestrator
├── requirments.txt
└── setup_env.bat
```

## Dataset

- Source: **DILIrank** (FDA, `data/DILIrank2.xlsx`).
- After cleaning and SMILES standardisation: **n = 869 molecules** (pos = 528).
- Split: **521 train / 173 val / 175 test** (Bemis–Murcko scaffold split).

## Quick start

### 1. Set up the environment

On Windows:

```bat
setup_env.bat
```

Or manually:

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS
pip install -r requirments.txt
```

### 2. Run the pipeline

```bash
# Smoke test (~5–10 min on CPU): validates the full pipeline end-to-end
python -m src.improved.run_pipeline --quick

# Full run (recommended for reporting)
python -m src.improved.run_pipeline --n-trials 25 --seeds 5

# Classical ML baselines (XGB / RF / LR / SVM on Morgan FP + RDKit descriptors)
python -m src.improved.classical_ml

# Stacking meta-learner (run after the two above)
python -m src.improved.stack
```

## Results

Final test metrics (mean ± std over multiple seeds, sorted by AUROC):

| Model      | AUROC           | ACC    | F1     | MCC    |
|------------|-----------------|--------|--------|--------|
| GIN        | 0.688 ± 0.000   | 0.669  | 0.743  | 0.282  |
| GraphSAGE  | 0.680 ± 0.025   | 0.657  | 0.740  | 0.248  |
| GCN        | 0.678 ± 0.022   | 0.646  | 0.716  | 0.246  |
| GAT        | 0.673 ± 0.018   | 0.634  | 0.674  | 0.273  |
| MPNN       | 0.653 ± 0.039   | 0.646  | 0.763  | 0.177  |

Threshold-tuned (validation-MCC) metrics on the held-out test set:

| Model         | Threshold | AUROC | ACC   | F1    | MCC   |
|---------------|-----------|-------|-------|-------|-------|
| GIN           | 0.53      | 0.716 | 0.691 | 0.748 | 0.351 |
| GraphSAGE     | 0.54      | 0.715 | 0.657 | 0.720 | 0.279 |
| GNN-Ensemble  | 0.50      | 0.713 | 0.640 | 0.725 | 0.212 |
| GCN           | 0.55      | 0.697 | 0.640 | 0.644 | 0.347 |
| GAT           | 0.49      | 0.693 | 0.657 | 0.703 | 0.306 |
| MPNN          | 0.57      | 0.683 | 0.674 | 0.740 | 0.305 |

See [`results/summary.txt`](results/summary.txt) for the full numerical report and `results/roc_curves.png` for the ROC-curve comparison plot.

## Methods (short)

- **Scaffold split** — molecules with the same Bemis–Murcko scaffold stay in the same split, so the model is forced to generalise to truly novel chemistry.
- **Atom features (53-dim)** — atomic number, degree, formal charge, hybridisation, aromaticity, H-count, chirality, radical electrons, ring-size membership (3- to 7-membered).
- **Class imbalance** — handled at loss level with `pos_weight` in `BCEWithLogitsLoss` (SMOTE is not well-defined on graphs).
- **Hyperparameter search** — Grid Search over optimizer / lr / weight-decay; Optuna TPE over architecture (hidden_dim, num_layers, dropout, …).
- **Statistical tests** — DeLong (AUROC curves), Wilcoxon (paired across seeds), McNemar (per-sample disagreement).
- **Stacking** — logistic-regression meta-learner trained on validation-set predictions of the GNN ensemble + classical models.

## Tech stack

`Python 3` · `PyTorch` · `PyTorch Geometric` · `RDKit` · `scikit-learn` · `XGBoost` · `Optuna` · `pandas` · `numpy` · `matplotlib`

## Author

**Md Sifat Hosen**

If you use this code or find it useful for your research, please cite the repository.
