"""Re-run final evaluation with 5+ seeds, reusing the configs already tuned in configs/.

Why: the supervisor's feedback explicitly asks for "mean and standard deviation across
5–10 runs" instead of the 2-seed snapshot from --quick mode. Re-running the full
pipeline (tune + retrain) takes 5+ hours. This script skips re-tuning by loading
configs/<MODEL>_best.json and only doing steps 5-8 (final retrain across seeds +
stats + ROC + summary). Typical wall time: ~60-90 min on CPU for 5 seeds.

Run:
    python -m src.improved.retrain_only --seeds 5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.improved.data_utils import load_dataset, scaffold_split
    from src.improved.cv_utils import class_distribution
    from src.improved.run_pipeline import retrain_evaluate, aggregate, plot_rocs, write_csv
    from src.improved.stats_utils import delong_test, wilcoxon_per_fold, mcnemar_test, auroc_ci
else:
    from .data_utils import load_dataset, scaffold_split
    from .cv_utils import class_distribution
    from .run_pipeline import retrain_evaluate, aggregate, plot_rocs, write_csv
    from .stats_utils import delong_test, wilcoxon_per_fold, mcnemar_test, auroc_ci


MODELS = ["GCN", "GAT", "GraphSAGE", "GIN", "MPNN"]
DEFAULT_SEEDS = [42, 7, 13, 21, 100, 2024, 3, 11, 88, 99]

# NNConv allocates a [E, hidden_dim*hidden_dim] message tensor every step. With
# hidden_dim=128 and batch_size=64 on CPU this hits ~250 MB per allocation and
# OOMs Windows machines mid-training. Cap MPNN's effective batch to keep that
# bounded; results are still aggregated by epoch so this only affects memory.
MPNN_BATCH_CAP = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--final-epochs", type=int, default=120)
    ap.add_argument("--final-patience", type=int, default=18)
    ap.add_argument("--models", nargs="+", default=MODELS)
    args = ap.parse_args()

    base = Path(__file__).resolve().parents[2]
    configs_dir = base / "configs"
    results_dir = base / "results"
    results_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[retrain] device={device} seeds={args.seeds} final_epochs={args.final_epochs}", flush=True)
    t0 = time.time()

    graphs = load_dataset(base / "data" / "dili_clean.csv")
    n = len(graphs)
    train_idx, val_idx, test_idx = scaffold_split(graphs, 0.6, 0.2, 0.2, seed=42)
    print(f"[retrain] split  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}", flush=True)

    parity_rows = [
        {"split": "train", **class_distribution(graphs, train_idx)},
        {"split": "val",   **class_distribution(graphs, val_idx)},
        {"split": "test",  **class_distribution(graphs, test_idx)},
    ]
    write_csv(results_dir / "parity_table.csv", parity_rows)

    # Load tuned configs (must exist from a prior run_pipeline run).
    best_per_model: Dict[str, dict] = {}
    for name in args.models:
        cfg_path = configs_dir / f"{name}_best.json"
        if not cfg_path.exists():
            print(f"[retrain] ERROR: {cfg_path} missing — run run_pipeline.py first to tune.", flush=True)
            sys.exit(1)
        best_per_model[name] = json.loads(cfg_path.read_text())
        print(f"[retrain] {name}: method={best_per_model[name].get('method')} "
              f"val_auc={best_per_model[name].get('best_value'):.4f}", flush=True)

    seeds = DEFAULT_SEEDS[: args.seeds]
    print(f"[retrain] seeds = {seeds}", flush=True)

    final_results: Dict[str, dict] = {}
    summary_rows = []
    per_seed_rows = []

    for name in args.models:
        print(f"\n[retrain] -- {name} --", flush=True)
        params = dict(best_per_model[name]["params"])
        if name == "MPNN" and params.get("batch_size", 0) > MPNN_BATCH_CAP:
            print(f"  capping MPNN batch_size {params['batch_size']} -> {MPNN_BATCH_CAP} (memory)", flush=True)
            params["batch_size"] = MPNN_BATCH_CAP
        res = retrain_evaluate(
            graphs, train_idx, val_idx, test_idx, name, params,
            seeds=seeds, device=device,
            epochs=args.final_epochs, patience=args.final_patience,
        )
        final_results[name] = res
        agg = aggregate(res["per_seed"])
        auroc, ci_lo, ci_hi = auroc_ci(res["y_true"], res["mean_probs"])
        summary_rows.append({
            "model": name, **agg,
            "AUROC_pooled": auroc,
            "AUROC_CI_low": ci_lo, "AUROC_CI_high": ci_hi,
            "best_method": best_per_model[name].get("method", "unknown"),
        })
        for r in res["per_seed"]:
            per_seed_rows.append({"model": name, **r})

    summary_rows.sort(key=lambda r: r["AUROC_mean"], reverse=True)
    write_csv(results_dir / "final_metrics.csv", summary_rows)
    write_csv(results_dir / "per_seed_metrics.csv", per_seed_rows)

    # Significance tests.
    delong_rows, wilcoxon_rows, mcnemar_rows = [], [], []
    names = list(args.models)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ra, rb = final_results[a], final_results[b]
            d = delong_test(ra["y_true"], ra["mean_probs"], rb["mean_probs"])
            delong_rows.append({"model_a": a, "model_b": b, **d})
            aucs_a = [r["AUROC"] for r in ra["per_seed"]]
            aucs_b = [r["AUROC"] for r in rb["per_seed"]]
            w = wilcoxon_per_fold(aucs_a, aucs_b)
            wilcoxon_rows.append({"model_a": a, "model_b": b, **w,
                                  "auc_a_mean": float(np.mean(aucs_a)),
                                  "auc_b_mean": float(np.mean(aucs_b))})
            pred_a = [1 if p >= 0.5 else 0 for p in ra["mean_probs"]]
            pred_b = [1 if p >= 0.5 else 0 for p in rb["mean_probs"]]
            m = mcnemar_test(ra["y_true"], pred_a, pred_b)
            mcnemar_rows.append({"model_a": a, "model_b": b, **m})

    write_csv(results_dir / "significance_delong.csv", delong_rows)
    write_csv(results_dir / "significance_wilcoxon.csv", wilcoxon_rows)
    write_csv(results_dir / "significance_mcnemar.csv", mcnemar_rows)

    plot_rocs(final_results, results_dir / "roc_curves.png")

    # GNN ensemble + threshold tuning (same logic as run_pipeline step 8b).
    print("\n[retrain] GNN ensemble + threshold tuning", flush=True)
    y_true_test = np.asarray(final_results[args.models[0]]["y_true"])
    y_true_val = np.asarray(final_results[args.models[0]]["y_val_true"])
    test_prob_matrix = np.stack([np.asarray(final_results[m]["mean_probs"]) for m in args.models])
    val_prob_matrix = np.stack([np.asarray(final_results[m]["mean_val_probs"]) for m in args.models])
    ensemble_test = test_prob_matrix.mean(axis=0)
    ensemble_val = val_prob_matrix.mean(axis=0)

    from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, roc_auc_score
    def _best_thr(y, p):
        best_t, best_v = 0.5, -1.0
        for t in np.linspace(0.1, 0.9, 81):
            yp = (p >= t).astype(int)
            v = matthews_corrcoef(y, yp)
            if v > best_v:
                best_v, best_t = float(v), float(t)
        return best_t
    def _metrics(y, p, t):
        yp = (p >= t).astype(int)
        return {
            "AUROC": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
            "ACC": float(accuracy_score(y, yp)),
            "F1": float(f1_score(y, yp, zero_division=0)),
            "MCC": float(matthews_corrcoef(y, yp)),
        }

    tuned_rows = []
    for m in args.models:
        p_va = np.asarray(final_results[m]["mean_val_probs"])
        p_te = np.asarray(final_results[m]["mean_probs"])
        t = _best_thr(y_true_val, p_va)
        tuned_rows.append({"model": m, "threshold": t, **_metrics(y_true_test, p_te, t)})
    t_ens = _best_thr(y_true_val, ensemble_val)
    tuned_rows.append({"model": "GNN-Ensemble", "threshold": t_ens, **_metrics(y_true_test, ensemble_test, t_ens)})
    write_csv(results_dir / "tuned_metrics.csv", tuned_rows)

    # Save predictions for stack.py.
    np.savez(results_dir / "gnn_test_probs.npz",
             y_true=y_true_test,
             ensemble=ensemble_test,
             **{m: np.asarray(final_results[m]["mean_probs"]) for m in args.models})
    np.savez(results_dir / "gnn_val_probs.npz",
             y_true=y_true_val,
             ensemble=ensemble_val,
             **{m: np.asarray(final_results[m]["mean_val_probs"]) for m in args.models})

    elapsed = time.time() - t0
    summary_lines = [
        f"DILI retrain-only summary (5+ seeds, no re-tune)  elapsed {elapsed/60:.1f} min",
        f"Dataset: n={n}  pos={sum(int(g.y.item()) for g in graphs)}",
        f"Splits  train/val/test = {len(train_idx)}/{len(val_idx)}/{len(test_idx)}",
        "",
        f"Final results (mean +/- std over {len(seeds)} seeds, sorted by AUROC):",
    ]
    for r in summary_rows:
        summary_lines.append(
            f"  {r['model']:<10} AUROC={r['AUROC_mean']:.4f}+/-{r['AUROC_std']:.4f} "
            f"(CI {r['AUROC_CI_low']:.3f}-{r['AUROC_CI_high']:.3f})  "
            f"ACC={r['ACC_mean']:.4f}  F1={r['F1_mean']:.4f}  MCC={r['MCC_mean']:.4f}"
        )
    summary_lines.append("")
    summary_lines.append("Threshold-tuned (val-MCC) metrics:")
    for r in tuned_rows:
        summary_lines.append(
            f"  {r['model']:<14} thr={r['threshold']:.2f}  AUROC={r['AUROC']:.4f}  "
            f"ACC={r['ACC']:.4f}  F1={r['F1']:.4f}  MCC={r['MCC']:.4f}"
        )
    summary = "\n".join(summary_lines)
    print("\n" + summary, flush=True)
    (results_dir / "summary.txt").write_text(summary)


if __name__ == "__main__":
    main()
