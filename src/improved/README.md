# Improved DILI prediction pipeline

This directory contains the upgraded version of the DILI graph-ML pipeline that
addresses the instructor's checklist:

| Instructor requirement | Where it's handled |
|---|---|
| Hyperparameter optimization (Grid + Optuna) | [tune.py](tune.py) — `run_grid_search`, `run_optuna`, `pick_best` |
| Class imbalance handling | [train_utils.py](train_utils.py) — `compute_pos_weight` + `BCEWithLogitsLoss(pos_weight=...)` |
| Statistical parity check | [cv_utils.py](cv_utils.py) — `class_distribution`, `parity_table`; reported in `results/parity_table.csv` |
| Statistical significance between models | [stats_utils.py](stats_utils.py) — DeLong, Wilcoxon, McNemar; bootstrap AUROC CI |
| Specific layer changes per model | [models.py](models.py) — see header docstring for what changed in each of GCN/GAT/GraphSAGE/GIN/MPNN |
| No data leakage | [data_utils.py](data_utils.py) — Bemis-Murcko `scaffold_split` + held-out test set never seen during tuning |

## How to run

From the project root (`c:/Users/sifat/Desktop/dili_project`):

```bash
# Smoke test (~5-10 min on CPU): validates the whole pipeline runs end-to-end
python -m src.improved.run_pipeline --quick

# Full run (recommended for the thesis report)
python -m src.improved.run_pipeline --n-trials 25 --seeds 5

# Classical ML baselines (XGBoost / RF / LR / SVM on Morgan FP + RDKit descriptors)
python -m src.improved.classical_ml

# Stacking meta-learner over GNN + classical predictions (run after the two above)
python -m src.improved.stack
```

## Accuracy-boosting layer (added on top of the GNN comparison)

The base pipeline scores 5 GNNs against each other. To push absolute accuracy
higher (the thesis's secondary goal), three additional components were added:

1. **Richer atom features** ([data_utils.py](data_utils.py)) — chirality tag,
   radical-electron count, ring-size membership (3- to 7-membered).
   Feature dim grew from 42 → 53. Models pick this up automatically via `in_dim`.
2. **GNN ensemble + val-tuned threshold** ([run_pipeline.py](run_pipeline.py)
   step 8b) — averages the 5 GNNs and tunes the decision threshold on the val
   set for MCC. Reported in `results/tuned_metrics.csv` and `summary.txt`.
3. **Classical-ML + stacking** ([classical_ml.py](classical_ml.py),
   [stack.py](stack.py)) — XGBoost / RandomForest / LogReg / SVM on Morgan
   fingerprints + RDKit descriptors, combined with the GNN predictions through
   a logistic-regression meta-learner trained on the val set. Outputs:
   `results/classical_metrics.csv`, `results/stack_metrics.csv`.

The full recipe to maximize held-out accuracy is therefore:

```bash
python -m src.improved.run_pipeline --n-trials 50 --seeds 5
python -m src.improved.classical_ml
python -m src.improved.stack
```

Each step writes `*_test_probs.npz` / `*_val_probs.npz`, so `stack.py` can
combine them without re-training anything.

## Outputs

After a full run you get:

```
configs/
  GCN_optuna.json, GCN_grid.json, GCN_best.json    (and same for each model)
results/
  parity_table.csv                 # class distribution per split (parity check)
  final_metrics.csv                # AUROC/ACC/F1/MCC mean +/- std + 95% CI
  per_seed_metrics.csv             # raw per-seed numbers
  tuned_metrics.csv                # threshold-tuned + GNN-ensemble metrics
  significance_delong.csv          # pairwise DeLong p-values
  significance_wilcoxon.csv        # paired Wilcoxon over seeds
  significance_mcnemar.csv         # McNemar's test on predictions
  roc_curves.png                   # ROC plot of all 5 models on the held-out test
  gnn_test_probs.npz               # per-model + ensemble test probabilities
  gnn_val_probs.npz                # per-model + ensemble val probabilities
  classical_metrics.csv            # XGB/RF/LR/SVM standalone metrics
  classical_test_probs.npz         # classical model test probabilities
  classical_val_probs.npz          # classical model val probabilities
  stack_metrics.csv                # stacked-ensemble metrics (mean + logreg)
  stack_final_probs.npz            # final stacked predictions
  summary.txt                      # human-readable final summary
```

## Defending the design in the thesis

- **Why scaffold split?** Random splits leak structurally similar molecules between
  train and test. Bemis-Murcko scaffold split forces the model to generalize to
  truly novel chemical scaffolds — the standard practice in drug-discovery ML.
- **Why pos_weight (not SMOTE)?** SMOTE on graphs is ill-defined (you can't
  interpolate molecular graphs). `pos_weight` in `BCEWithLogitsLoss` is the
  standard, mathematically equivalent loss-level fix for class imbalance.
- **Why both Optuna and Grid?** Grid covers the optimizer/lr/weight-decay axes
  exhaustively (defensible in a thesis: "I tried every combination"). Optuna's
  TPE sampler then explores architecture knobs (hidden_dim, num_layers, dropout,
  etc.) that would be too expensive to grid.
- **Why DeLong + Wilcoxon + McNemar?** They test different things: DeLong
  compares AUROC curves on the same test set; Wilcoxon is a non-parametric
  alternative across seeds; McNemar tests whether two classifiers disagree
  systematically on the same samples. Reporting all three is the gold-standard
  for binary-classifier comparison.
