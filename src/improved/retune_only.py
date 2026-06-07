"""Re-tune Optuna only — for the new 14-d global descriptor + descriptor-fusion pipeline.

Why this exists:
  - The existing configs/<MODEL>_best.json files were tuned against the OLD
    53-d atom features + 10-d descriptor pipeline (before we added Brenk/PAINS/NIH
    alerts). Hyperparameters that were optimal then may not be optimal now.
  - This script re-runs Optuna TPE only, updates configs/, and writes a comparison
    of old-vs-new best-val-AUROC per model. It does NOT re-run the final retrain
    — call retrain_only.py separately after to evaluate the new configs.

Run:
    python -m src.improved.retune_only --n-trials 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.improved.data_utils import load_dataset, scaffold_split
    from src.improved.tune import run_optuna
else:
    from .data_utils import load_dataset, scaffold_split
    from .tune import run_optuna


MODELS = ["GCN", "GAT", "GraphSAGE", "GIN", "MPNN"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", nargs="+", default=MODELS)
    args = ap.parse_args()

    base = Path(__file__).resolve().parents[2]
    configs_dir = base / "configs"
    results_dir = base / "results"
    results_dir.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[retune] device={device}  n_trials={args.n_trials}  models={args.models}", flush=True)

    graphs = load_dataset(base / "data" / "dili_clean.csv")
    train_idx, val_idx, _ = scaffold_split(graphs, 0.6, 0.2, 0.2, seed=42)
    print(f"[retune] split tr={len(train_idx)} va={len(val_idx)}", flush=True)

    summary_rows = []
    for name in args.models:
        old_path = configs_dir / f"{name}_best.json"
        old = json.loads(old_path.read_text()) if old_path.exists() else None
        old_val = old["best_value"] if old else float("nan")
        print(f"\n[retune] === {name} (old val_auc={old_val:.4f}) ===", flush=True)
        t0 = time.time()
        new = run_optuna(
            graphs, train_idx, val_idx, name,
            n_trials=args.n_trials, device=device,
            epochs=args.epochs, patience=args.patience, seed=args.seed,
            output_dir=configs_dir,
        )
        elapsed = time.time() - t0
        new_val = new["best_value"]
        better = new_val > (old_val if old else 0)
        decision = "UPDATE" if better else "KEEP-OLD"
        print(f"[retune] {name} new val_auc={new_val:.4f}  delta={new_val - (old_val if old else 0):+.4f}  "
              f"({elapsed:.0f}s)  -> {decision}", flush=True)
        summary_rows.append({
            "model": name, "old_val_auc": old_val, "new_val_auc": new_val,
            "delta": new_val - (old_val if old else 0),
            "elapsed_sec": elapsed, "decision": decision,
        })
        if better:
            (configs_dir / f"{name}_best.json").write_text(json.dumps(new, indent=2))

    import pandas as pd
    pd.DataFrame(summary_rows).to_csv(results_dir / "retune_summary.csv", index=False)
    print(f"\n[retune] wrote results/retune_summary.csv", flush=True)
    for r in summary_rows:
        print(f"  {r['model']:<10} {r['old_val_auc']:.4f} -> {r['new_val_auc']:.4f} "
              f"({r['delta']:+.4f}) {r['decision']}", flush=True)


if __name__ == "__main__":
    main()
