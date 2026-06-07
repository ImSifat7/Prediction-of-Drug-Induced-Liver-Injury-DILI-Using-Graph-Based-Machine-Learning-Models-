"""10-fold scaffold-grouped cross-validation for each GNN (supervisor's ask).

Uses scaffold_kfold() from cv_utils so scaffolds never split across folds and class
ratio stays balanced. For each fold:
  - 8 folds -> train, 1 fold -> val (early stop), 1 fold -> test
  - Train each model with its previously-tuned best params from configs/<MODEL>_best.json
  - Report AUROC/ACC/F1/MCC per fold + mean +/- std across folds.

Run:
    python -m src.improved.cv_run --models GIN GAT          # subset
    python -m src.improved.cv_run                            # all 5 models
    python -m src.improved.cv_run --n-splits 5               # 5-fold instead

Outputs:
  results/cv_per_fold.csv      — every (model, fold) metric row
  results/cv_summary.csv       — mean +/- std per model
  results/cv_metrics.csv       — alias matching the meeting-prep doc
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.improved.data_utils import load_dataset
    from src.improved.cv_utils import scaffold_kfold, class_distribution
    from src.improved.models import build_model
    from src.improved.train_utils import (
        compute_pos_weight, evaluate, make_loader, set_seed, train_model,
    )
else:
    from .data_utils import load_dataset
    from .cv_utils import scaffold_kfold, class_distribution
    from .models import build_model
    from .train_utils import (
        compute_pos_weight, evaluate, make_loader, set_seed, train_model,
    )


MODELS = ["GCN", "GAT", "GraphSAGE", "GIN", "MPNN"]


def carve_val(train_idx: List[int], seed: int, val_frac: float = 0.111):
    """Carve a 1/9 inner-val from training fold for early stopping (no leak into test)."""
    import random
    rng = random.Random(seed)
    pool = list(train_idx)
    rng.shuffle(pool)
    n_val = max(int(len(pool) * val_frac), 8)
    return pool[n_val:], pool[:n_val]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base = Path(__file__).resolve().parents[2]
    configs_dir = base / "configs"
    results_dir = base / "results"
    results_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cv] device={device}  n_splits={args.n_splits}  models={args.models}", flush=True)

    graphs = load_dataset(base / "data" / "dili_clean.csv")
    in_dim = graphs[0].x.shape[1]
    edge_dim = graphs[0].edge_attr.shape[1]
    desc_dim = graphs[0].u.shape[1] if hasattr(graphs[0], "u") and graphs[0].u is not None else 0

    print("[cv] building scaffold-grouped k-fold splits…", flush=True)
    splits = scaffold_kfold(graphs, n_splits=args.n_splits, seed=args.seed)
    for fi, (tr, te) in enumerate(splits):
        d_tr = class_distribution(graphs, tr)
        d_te = class_distribution(graphs, te)
        print(f"  fold {fi}  train n={d_tr['n']} pos={d_tr['pos']}  test n={d_te['n']} pos={d_te['pos']}", flush=True)

    # Load best params per model from existing configs.
    best_params: Dict[str, dict] = {}
    for name in args.models:
        cfg = configs_dir / f"{name}_best.json"
        if not cfg.exists():
            print(f"[cv] WARN: {cfg} missing — using sane defaults for {name}", flush=True)
            best_params[name] = {
                "lr": 1e-3, "weight_decay": 1e-4, "hidden_dim": 64, "num_layers": 3,
                "dropout": 0.3, "optimizer_name": "adamw", "batch_size": 32, "heads": 4,
                "aggr": "max", "num_steps": 3,
            }
        else:
            best_params[name] = json.loads(cfg.read_text())["params"]

    per_fold_rows = []
    t_total = time.time()
    for name in args.models:
        params = best_params[name]
        print(f"\n[cv] === {name} ===", flush=True)
        for fi, (train_idx, test_idx) in enumerate(splits):
            t0 = time.time()
            inner_train, inner_val = carve_val(train_idx, seed=args.seed + fi)
            set_seed(args.seed)
            model = build_model(name, in_dim, edge_dim, params, desc_dim=desc_dim)
            tr_loader = make_loader(graphs, inner_train, batch_size=params["batch_size"], shuffle=True)
            va_loader = make_loader(graphs, inner_val, batch_size=params["batch_size"], shuffle=False)
            te_loader = make_loader(graphs, test_idx, batch_size=64, shuffle=False)
            pos_w = compute_pos_weight(graphs, inner_train)
            model, best_val_auc = train_model(
                model, tr_loader, va_loader,
                lr=params["lr"], weight_decay=params["weight_decay"],
                optimizer_name=params["optimizer_name"], pos_weight=pos_w,
                epochs=args.epochs, patience=args.patience, device=device,
            )
            r = evaluate(model, te_loader, device)
            per_fold_rows.append({
                "model": name, "fold": fi,
                "AUROC": r["AUROC"], "ACC": r["ACC"], "F1": r["F1"], "MCC": r["MCC"],
                "best_val_auc": best_val_auc,
                "n_test": len(test_idx),
            })
            print(f"  fold {fi}  best_val_auc={best_val_auc:.4f}  test_auc={r['AUROC']:.4f}  "
                  f"acc={r['ACC']:.4f}  mcc={r['MCC']:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(per_fold_rows)
    df.to_csv(results_dir / "cv_per_fold.csv", index=False)

    # Aggregate per model.
    agg_rows = []
    for name in args.models:
        sub = df[df["model"] == name]
        agg_rows.append({
            "model": name,
            "AUROC_mean": float(sub["AUROC"].mean()),
            "AUROC_std":  float(sub["AUROC"].std(ddof=1)),
            "ACC_mean":   float(sub["ACC"].mean()),
            "ACC_std":    float(sub["ACC"].std(ddof=1)),
            "F1_mean":    float(sub["F1"].mean()),
            "F1_std":     float(sub["F1"].std(ddof=1)),
            "MCC_mean":   float(sub["MCC"].mean()),
            "MCC_std":    float(sub["MCC"].std(ddof=1)),
            "n_folds":    len(sub),
        })
    agg_rows.sort(key=lambda r: r["AUROC_mean"], reverse=True)
    pd.DataFrame(agg_rows).to_csv(results_dir / "cv_summary.csv", index=False)
    pd.DataFrame(agg_rows).to_csv(results_dir / "cv_metrics.csv", index=False)  # alias for meeting prep

    print(f"\n[cv] elapsed {(time.time()-t_total)/60:.1f} min", flush=True)
    print("[cv] per-model summary:", flush=True)
    for r in agg_rows:
        print(f"  {r['model']:<10}  AUROC={r['AUROC_mean']:.4f}+/-{r['AUROC_std']:.4f}  "
              f"ACC={r['ACC_mean']:.4f}+/-{r['ACC_std']:.4f}  "
              f"F1={r['F1_mean']:.4f}+/-{r['F1_std']:.4f}  "
              f"MCC={r['MCC_mean']:.4f}+/-{r['MCC_std']:.4f}", flush=True)
    print(f"\n[cv] wrote results to {results_dir}/cv_per_fold.csv, cv_summary.csv, cv_metrics.csv", flush=True)


if __name__ == "__main__":
    main()
