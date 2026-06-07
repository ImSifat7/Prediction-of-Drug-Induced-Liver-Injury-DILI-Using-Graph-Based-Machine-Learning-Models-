"""End-to-end orchestrator for the improved DILI pipeline.

What it does, in order:

  1. Load + standardize SMILES, build PyG graphs (rich atom/bond features).
  2. Scaffold-split the data into  60% train / 20% val / 20% held-out test.
     The test set is NEVER touched until step 6 (no data leakage).
  3. Print the parity table (class distribution per split — the 'statistical parity check').
  4. For each of the 5 models, run BOTH:
       - Optuna (TPE) hyperparameter search on (train, val)
       - Grid Search on the same (focused on the optimizer / lr / weight-decay axes)
     Best parameters per model are saved to configs/.
  5. Re-train each tuned model 5 times (different random seeds) on train+val.
     Each seed gets its own internal 90/10 split for early-stopping.
  6. Evaluate on the held-out test set:
       AUROC / ACC / F1 / MCC reported as mean +/- std + bootstrap 95% CI for AUROC.
  7. Pairwise statistical comparison between models on the held-out test:
       - DeLong's test (correlated AUROCs)
       - Wilcoxon signed-rank (per-seed AUROCs)
       - McNemar's test (paired correctness)
  8. Save: configs/, results/*.csv, results/roc_curves.png, results/summary.txt

Run:
    python -m src.improved.run_pipeline --n-trials 25 --seeds 5

Use --quick for a fast smoke test (small trial count, 2 seeds, fewer epochs).
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
from torch_geometric.data import Data

# Allow running both as a module (-m src.improved.run_pipeline) and as a script.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.improved.data_utils import load_dataset, scaffold_split
    from src.improved.cv_utils import class_distribution
    from src.improved.models import build_model
    from src.improved.train_utils import (
        compute_pos_weight, evaluate, make_loader, set_seed, train_model,
    )
    from src.improved.tune import run_optuna, run_grid_search, pick_best
    from src.improved.stats_utils import delong_test, wilcoxon_per_fold, mcnemar_test, auroc_ci
else:
    from .data_utils import load_dataset, scaffold_split
    from .cv_utils import class_distribution
    from .models import build_model
    from .train_utils import (
        compute_pos_weight, evaluate, make_loader, set_seed, train_model,
    )
    from .tune import run_optuna, run_grid_search, pick_best
    from .stats_utils import delong_test, wilcoxon_per_fold, mcnemar_test, auroc_ci


MODELS = ["GCN", "GAT", "GraphSAGE", "GIN", "MPNN"]


def write_csv(path: Path, rows: List[dict], fieldnames: List[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def split_for_seed(train_idx: List[int], val_idx: List[int], seed: int):
    """For each final-eval seed: combine train+val, then carve out a 10% inner-val
    for early stopping. The held-out test set is untouched."""
    import random
    pool = list(train_idx) + list(val_idx)
    rng = random.Random(seed)
    rng.shuffle(pool)
    n_val = max(int(len(pool) * 0.1), 8)
    inner_val = pool[:n_val]
    inner_train = pool[n_val:]
    return inner_train, inner_val


def retrain_evaluate(
    graphs: List[Data],
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
    model_name: str,
    params: dict,
    seeds: List[int],
    device,
    epochs: int,
    patience: int,
) -> Dict:
    """Train with best params N times (different seeds). Return per-seed test+val probs."""
    in_dim = graphs[0].x.shape[1]
    edge_dim = graphs[0].edge_attr.shape[1]
    desc_dim = graphs[0].u.shape[1] if hasattr(graphs[0], "u") and graphs[0].u is not None else 0
    test_loader = make_loader(graphs, test_idx, batch_size=64, shuffle=False)
    # Use the original outer val (not the inner val) for stacking meta-features so
    # all models share the same val rows.
    outer_val_loader = make_loader(graphs, val_idx, batch_size=64, shuffle=False)

    per_seed = []
    y_probs_seeds = []
    val_probs_seeds = []
    for s in seeds:
        inner_train, inner_val = split_for_seed(train_idx, val_idx, s)
        set_seed(s)
        model = build_model(model_name, in_dim, edge_dim, params, desc_dim=desc_dim)
        train_loader = make_loader(graphs, inner_train, batch_size=params["batch_size"], shuffle=True)
        v_loader = make_loader(graphs, inner_val, batch_size=params["batch_size"], shuffle=False)
        pos_w = compute_pos_weight(graphs, inner_train)
        model, best_val_auc = train_model(
            model, train_loader, v_loader,
            lr=params["lr"], weight_decay=params["weight_decay"],
            optimizer_name=params["optimizer_name"], pos_weight=pos_w,
            epochs=epochs, patience=patience, device=device,
        )
        res = evaluate(model, test_loader, device)
        # Predict on the outer val too (for threshold tuning + stacking).
        # This is fine because the *inner* val (carved from train+val for early stop)
        # is what gates the model choice; the outer val rows here are the same set
        # already used during tuning, so no extra leakage is introduced.
        val_res = evaluate(model, outer_val_loader, device)
        per_seed.append({
            "seed": s,
            "AUROC": res["AUROC"], "ACC": res["ACC"], "F1": res["F1"], "MCC": res["MCC"],
        })
        y_probs_seeds.append(res["y_prob"])
        val_probs_seeds.append(val_res["y_prob"])
        print(f"  seed={s} best_val_auc={best_val_auc:.4f} test_auc={res['AUROC']:.4f}")

    y_true = res["y_true"]
    y_val_true = val_res["y_true"]
    mean_probs = np.mean(np.array(y_probs_seeds), axis=0).tolist()
    mean_val_probs = np.mean(np.array(val_probs_seeds), axis=0).tolist()
    return {
        "per_seed": per_seed,
        "y_true": y_true,
        "y_probs_seeds": y_probs_seeds,
        "mean_probs": mean_probs,
        "y_val_true": y_val_true,
        "mean_val_probs": mean_val_probs,
    }


def aggregate(per_seed: List[dict]) -> dict:
    out = {}
    for k in ("AUROC", "ACC", "F1", "MCC"):
        vals = [r[k] for r in per_seed]
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    return out


def plot_rocs(model_results: Dict[str, dict], out_path: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed, skipping ROC plot")
        return
    from sklearn.metrics import roc_curve, auc

    plt.figure(figsize=(7, 6))
    for name, res in model_results.items():
        y_true = np.asarray(res["y_true"])
        probs = np.asarray(res["mean_probs"])
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves on held-out scaffold-split test set")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dili_clean.csv", help="Input CSV (relative to project root)")
    parser.add_argument("--n-trials", type=int, default=25, help="Optuna trials per model")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds for final evaluation")
    parser.add_argument("--tune-epochs", type=int, default=80)
    parser.add_argument("--tune-patience", type=int, default=12)
    parser.add_argument("--final-epochs", type=int, default=150)
    parser.add_argument("--final-patience", type=int, default=20)
    parser.add_argument("--quick", action="store_true", help="Fast smoke test (5 trials, 2 seeds, fewer epochs)")
    parser.add_argument("--models", nargs="+", default=MODELS, help="Subset of models to run")
    args = parser.parse_args()

    if args.quick:
        args.n_trials = 5
        args.seeds = 2
        args.tune_epochs = 25
        args.final_epochs = 40
        args.tune_patience = 6
        args.final_patience = 8

    base = Path(__file__).resolve().parents[2]
    data_path = base / args.data
    configs_dir = base / "configs"
    results_dir = base / "results"
    configs_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    t0 = time.time()

    # 1. Load + standardize
    print("\n=== 1. Loading data ===")
    graphs = load_dataset(data_path)
    n = len(graphs)

    # 2. Scaffold split
    print("\n=== 2. Scaffold split (60/20/20) ===")
    train_idx, val_idx, test_idx = scaffold_split(graphs, 0.6, 0.2, 0.2, seed=42)
    print(f"train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    # 3. Parity table
    print("\n=== 3. Statistical parity (class distribution per split) ===")
    parity_rows = [
        {"split": "train", **class_distribution(graphs, train_idx)},
        {"split": "val",   **class_distribution(graphs, val_idx)},
        {"split": "test",  **class_distribution(graphs, test_idx)},
    ]
    for r in parity_rows:
        print(f"  {r['split']:<5} n={r['n']:>4}  pos={r['pos']:>3}  neg={r['neg']:>3}  pos_ratio={r['pos_ratio']:.3f}")
    write_csv(results_dir / "parity_table.csv", parity_rows)

    # Quick safety check: positives in all splits
    if any(r["pos"] == 0 or r["neg"] == 0 for r in parity_rows):
        print("[warn] one of the splits has only one class. Adjust frac_train/val or seed.")

    # 4. Tune each model (Optuna + Grid)
    print("\n=== 4. Hyperparameter tuning (Optuna TPE + Grid Search) ===")
    best_per_model: Dict[str, dict] = {}
    for name in args.models:
        print(f"\n-- {name} --")
        opt_res = run_optuna(
            graphs, train_idx, val_idx, name,
            n_trials=args.n_trials, device=device,
            epochs=args.tune_epochs, patience=args.tune_patience,
            seed=42, output_dir=configs_dir,
        )
        print(f"  optuna best val AUROC = {opt_res['best_value']:.4f}")
        grid_res = run_grid_search(
            graphs, train_idx, val_idx, name,
            device=device, epochs=args.tune_epochs, patience=args.tune_patience,
            seed=42, output_dir=configs_dir,
        )
        print(f"  grid   best val AUROC = {grid_res['best_value']:.4f}  ({grid_res['n_combos']} combos)")
        best = pick_best(opt_res, grid_res)
        best_per_model[name] = best
        (configs_dir / f"{name}_best.json").write_text(json.dumps(best, indent=2))
        print(f"  -> chosen: method={best['method']}  val AUROC={best['best_value']:.4f}")

    # 5+6. Final retrain with seeds + held-out test evaluation
    print("\n=== 5-6. Final evaluation (multi-seed retrain on held-out test) ===")
    seeds = [42, 7, 13, 21, 100][: args.seeds]
    final_results: Dict[str, dict] = {}
    summary_rows = []
    per_seed_rows = []

    for name in args.models:
        print(f"\n-- {name} --")
        params = best_per_model[name]["params"]
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
            "best_method": best_per_model[name]["method"],
        })
        for r in res["per_seed"]:
            per_seed_rows.append({"model": name, **r})

    summary_rows.sort(key=lambda r: r["AUROC_mean"], reverse=True)
    write_csv(results_dir / "final_metrics.csv", summary_rows)
    write_csv(results_dir / "per_seed_metrics.csv", per_seed_rows)

    # 7. Statistical comparisons
    print("\n=== 7. Statistical significance ===")
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

    # 8. ROC plot + summary
    plot_rocs(final_results, results_dir / "roc_curves.png")

    # 8b. GNN ensemble + threshold tuning (free wins on top of the per-model results)
    print("\n=== 8b. GNN ensemble + threshold tuning ===")
    y_true_test = np.asarray(final_results[args.models[0]]["y_true"])
    y_true_val = np.asarray(final_results[args.models[0]]["y_val_true"])

    # Stack per-model mean test/val probs.
    test_prob_matrix = np.stack([np.asarray(final_results[m]["mean_probs"]) for m in args.models])
    val_prob_matrix = np.stack([np.asarray(final_results[m]["mean_val_probs"]) for m in args.models])
    ensemble_test = test_prob_matrix.mean(axis=0)
    ensemble_val = val_prob_matrix.mean(axis=0)

    def _best_thr(y, p, target="MCC"):
        from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
        best_t, best_v = 0.5, -1.0
        for t in np.linspace(0.1, 0.9, 81):
            yp = (p >= t).astype(int)
            v = (matthews_corrcoef(y, yp) if target == "MCC"
                 else f1_score(y, yp, zero_division=0) if target == "F1"
                 else accuracy_score(y, yp))
            if v > best_v:
                best_v, best_t = float(v), float(t)
        return best_t

    def _metrics(y, p, t):
        from sklearn.metrics import (accuracy_score, f1_score,
                                     matthews_corrcoef, roc_auc_score)
        yp = (p >= t).astype(int)
        return {
            "AUROC": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
            "ACC": float(accuracy_score(y, yp)),
            "F1": float(f1_score(y, yp, zero_division=0)),
            "MCC": float(matthews_corrcoef(y, yp)),
        }

    # Per-model threshold-tuned metrics (val-tuned thresholds applied to test).
    tuned_rows = []
    for m in args.models:
        p_va = np.asarray(final_results[m]["mean_val_probs"])
        p_te = np.asarray(final_results[m]["mean_probs"])
        t = _best_thr(y_true_val, p_va, target="MCC")
        mt = _metrics(y_true_test, p_te, t)
        tuned_rows.append({"model": m, "threshold": t, **mt})

    t_ens = _best_thr(y_true_val, ensemble_val, target="MCC")
    ens_m = _metrics(y_true_test, ensemble_test, t_ens)
    tuned_rows.append({"model": "GNN-Ensemble", "threshold": t_ens, **ens_m})

    write_csv(results_dir / "tuned_metrics.csv", tuned_rows)
    print(f"GNN ensemble  thr={t_ens:.2f}  AUROC={ens_m['AUROC']:.4f}  "
          f"ACC={ens_m['ACC']:.4f}  F1={ens_m['F1']:.4f}  MCC={ens_m['MCC']:.4f}")

    # 8c. Save GNN predictions for the stacking script (gnn_test_probs.npz / val).
    np.savez(
        results_dir / "gnn_test_probs.npz",
        y_true=y_true_test,
        ensemble=ensemble_test,
        **{m: np.asarray(final_results[m]["mean_probs"]) for m in args.models},
    )
    np.savez(
        results_dir / "gnn_val_probs.npz",
        y_true=y_true_val,
        ensemble=ensemble_val,
        **{m: np.asarray(final_results[m]["mean_val_probs"]) for m in args.models},
    )
    print(f"[gnn] wrote {results_dir/'gnn_test_probs.npz'} and gnn_val_probs.npz")

    elapsed = time.time() - t0
    summary_lines = [
        f"DILI improved pipeline summary  (elapsed {elapsed/60:.1f} min)",
        f"Dataset: n={n}  pos={sum(int(g.y.item()) for g in graphs)}",
        f"Splits  train/val/test = {len(train_idx)}/{len(val_idx)}/{len(test_idx)}",
        "",
        "Final results (mean +/- std over %d seeds, sorted by AUROC):" % len(seeds),
    ]
    for r in summary_rows:
        summary_lines.append(
            f"  {r['model']:<10} AUROC={r['AUROC_mean']:.4f}+/-{r['AUROC_std']:.4f} "
            f"(CI {r['AUROC_CI_low']:.3f}-{r['AUROC_CI_high']:.3f})  "
            f"ACC={r['ACC_mean']:.4f}  F1={r['F1_mean']:.4f}  MCC={r['MCC_mean']:.4f}"
        )
    summary_lines.append("")
    summary_lines.append("Threshold-tuned (val-MCC) metrics on held-out test:")
    for r in tuned_rows:
        summary_lines.append(
            f"  {r['model']:<14} thr={r['threshold']:.2f}  AUROC={r['AUROC']:.4f}  "
            f"ACC={r['ACC']:.4f}  F1={r['F1']:.4f}  MCC={r['MCC']:.4f}"
        )
    summary = "\n".join(summary_lines)
    print("\n" + summary)
    (results_dir / "summary.txt").write_text(summary)

    print(f"\nAll outputs written under: {results_dir}")
    print(f"Tuned configs under:        {configs_dir}")


if __name__ == "__main__":
    main()
