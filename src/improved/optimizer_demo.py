"""Hyperparameter OPTIMIZER demonstration (Optuna / TPE) for the selected model.

Runs a Bayesian (Tree-structured Parzen Estimator) hyperparameter search for the
selected desc+Morgan -> XGBoost model, maximising mean VALIDATION AUROC over the
official TDC 5-seed protocol (no test peeking). Saves:
  - results/optuna_trials.csv        (every trial's params + val-AUROC)
  - results/optuna_history.png       (optimisation convergence)
  - results/optuna_importance.png    (which hyperparameters matter)

Run:  python -m src.improved.optimizer_demo --trials 60
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.improved.tdc_official_v2 import precompute_blocks, eval_config, SEEDS, RES, DATA_PATH
else:
    from .tdc_official_v2 import precompute_blocks, eval_config, SEEDS, RES, DATA_PATH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    args = ap.parse_args()

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from tdc.benchmark_group import admet_group

    group = admet_group(path=str(DATA_PATH))
    b = group.get("DILI")
    name = b["name"]
    test_df = b["test"]; trv_df = b["train_val"]
    master = sorted(set(trv_df["Drug"]).union(set(test_df["Drug"])))
    smi2i = {s: i for i, s in enumerate(master)}
    blocks = precompute_blocks(master, ["desc", "morgan"])
    X_all_idx = {"smi2i": smi2i, "test": [smi2i[s] for s in test_df["Drug"]]}
    y_test = test_df["Y"].values.astype(int)
    kinds = ["desc", "morgan"]
    print(f"[opt] TDC DILI  train_val={len(trv_df)} test={len(test_df)}  "
          f"optimizing desc+Morgan XGBoost over {args.trials} TPE trials", flush=True)

    def objective(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "lr": trial.suggest_float("lr", 0.01, 0.1, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 8.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample": trial.suggest_float("colsample", 0.4, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 8.0),
        }
        val, _, _ = eval_config(group, name, X_all_idx, blocks, kinds, "xgb", y_test, params=p, n_bag=1)
        return val

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    df = study.trials_dataframe()
    df.to_csv(RES / "optuna_trials.csv", index=False)
    vals = np.array([t.value for t in study.trials], float)
    running_best = np.maximum.accumulate(vals)
    print(f"[opt] best val-AUROC = {study.best_value:.4f}", flush=True)
    print(f"[opt] best params    = {study.best_params}", flush=True)
    # default-config baseline for reference
    base_val, _, _ = eval_config(group, name, X_all_idx, blocks, kinds, "xgb", y_test, n_bag=1)
    print(f"[opt] default-config val-AUROC = {base_val:.4f}  "
          f"(gain from tuning = {study.best_value - base_val:+.4f})", flush=True)

    # ---- convergence plot ----
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(range(1, len(vals) + 1), vals, "o", ms=4, alpha=0.4, color="#3A6EA5", label="trial val-AUROC")
    ax.plot(range(1, len(vals) + 1), running_best, "-", lw=2, color="#D1495B", label="best so far")
    ax.axhline(base_val, ls="--", lw=1, color="grey", label=f"default config ({base_val:.3f})")
    ax.set_xlabel("Optuna trial"); ax.set_ylabel("Mean validation AUROC (5 seeds)")
    ax.set_title("Hyperparameter optimization (TPE) — desc+Morgan XGBoost")
    ax.legend(loc="lower right", fontsize=9); plt.tight_layout()
    fig.savefig(RES / "optuna_history.png", dpi=130); plt.close(fig)

    # ---- importance plot ----
    try:
        imp = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ks = list(imp.keys())[::-1]; vs = [imp[k] for k in ks]
        ax.barh(ks, vs, color="#4C9F70")
        ax.set_xlabel("Relative importance (fANOVA)")
        ax.set_title("Which hyperparameters matter most")
        plt.tight_layout(); fig.savefig(RES / "optuna_importance.png", dpi=130); plt.close(fig)
        print(f"[opt] param importances = {dict((k, round(v,3)) for k,v in imp.items())}", flush=True)
    except Exception as e:
        print(f"[opt] importance plot skipped: {e}", flush=True)

    print("[opt] saved -> results/optuna_trials.csv, optuna_history.png, optuna_importance.png", flush=True)


if __name__ == "__main__":
    main()
