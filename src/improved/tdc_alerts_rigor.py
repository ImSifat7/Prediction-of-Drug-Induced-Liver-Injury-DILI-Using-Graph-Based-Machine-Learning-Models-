"""Rigor + interpretability for the mechanism-alert features.

Answers two questions a reviewer will ask:
  (1) Is the alert improvement REAL or noise?  -> paired per-seed delta across
      every (base feature set x model), Wilcoxon signed-rank, plus a DeLong test
      on the seed-averaged test predictions (alerts vs no-alerts).
  (2) WHICH alerts carry DILI signal?  -> univariate enrichment (Fisher exact,
      DILI+ rate present-vs-absent on TRAIN_VAL only) + XGBoost gain importance.

All AUROC comparisons keep the official val-selected, no-test-peeking protocol.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.improved.tdc_official_v2 import (
        precompute_blocks, eval_config, ensemble_eval, SEEDS, RES, DATA_PATH,
        extract_molformer_embeddings)
    from src.improved.tdc_alerts import compute_alerts_block, FEATURE_NAMES
    from src.improved.stats_utils import delong_test, wilcoxon_per_fold
else:
    from .tdc_official_v2 import (
        precompute_blocks, eval_config, ensemble_eval, SEEDS, RES, DATA_PATH,
        extract_molformer_embeddings)
    from .tdc_alerts import compute_alerts_block, FEATURE_NAMES
    from .stats_utils import delong_test, wilcoxon_per_fold


def per_seed_test(group, name, X_all_idx, blocks, kinds, model, y_test):
    """Return (per_seed_test_aucs, seed_avg_pred) for a single config."""
    if model == "ENS4":
        _, aucs, pred = ensemble_eval(group, name, X_all_idx, blocks, kinds,
                                      ["xgb", "lgbm", "cat"], y_test, n_bag=1)
    else:
        _, aucs, pred = eval_config(group, name, X_all_idx, blocks, kinds,
                                    model, y_test, n_bag=1)
    return np.array(aucs), pred


def main():
    from tdc.benchmark_group import admet_group
    group = admet_group(path=str(DATA_PATH))
    benchmark = group.get("DILI")
    name = benchmark["name"]
    test_df = benchmark["test"]; trv_df = benchmark["train_val"]
    master = sorted(set(trv_df["Drug"]).union(set(test_df["Drug"])))
    smi2i = {s: i for i, s in enumerate(master)}

    base_kinds = ["desc", "morgan", "avalon", "erg", "maccs"]
    blocks = precompute_blocks(master, base_kinds)
    blocks["alerts"] = compute_alerts_block(master)

    import torch
    cache = RES / "tdc_official_molformer_emb.npz"
    d = np.load(cache, allow_pickle=True)
    cmap = {s: e for s, e in zip(d["smiles"], d["emb"])}
    blocks["molformer"] = np.stack([cmap[s] for s in master], 0).astype(np.float32)

    X_all_idx = {"smi2i": smi2i, "test": [smi2i[s] for s in test_df["Drug"]]}
    y_test = test_df["Y"].values.astype(int)

    base_sets = {
        "desc_morgan": ["desc", "morgan"],
        "rich":        ["desc", "morgan", "avalon", "erg", "maccs"],
        "rich_mf":     ["desc", "morgan", "avalon", "erg", "maccs", "molformer"],
    }
    models = ["xgb", "lgbm", "cat", "ENS4"]

    # ---------- (1a) paired per-seed delta across all (base x model) ----------
    print("=== PAIRED per-seed test-AUROC delta: (+alerts) - (base) ===", flush=True)
    rows, all_a, all_b = [], [], []
    winner_pred = {}
    for bset, kinds in base_sets.items():
        for mk in models:
            a_aucs, a_pred = per_seed_test(group, name, X_all_idx, blocks, kinds + ["alerts"], mk, y_test)
            b_aucs, b_pred = per_seed_test(group, name, X_all_idx, blocks, kinds, mk, y_test)
            winner_pred[(bset, mk)] = (a_pred, b_pred)
            d_seed = a_aucs - b_aucs
            all_a.extend(a_aucs); all_b.extend(b_aucs)
            rows.append({"base": bset, "model": mk,
                         "test_base": float(b_aucs.mean()), "test_alerts": float(a_aucs.mean()),
                         "delta": float(d_seed.mean()), "delta_pos_seeds": int((d_seed > 0).sum())})
            print(f"  {bset:11s} {mk:5s}  {b_aucs.mean():.4f} -> {a_aucs.mean():.4f}  "
                  f"delta={d_seed.mean():+.4f}  ({int((d_seed>0).sum())}/5 seeds up)", flush=True)
    pd.DataFrame(rows).to_csv(RES / "tdc_alerts_rigor.csv", index=False)

    all_a, all_b = np.array(all_a), np.array(all_b)
    deltas = all_a - all_b
    w = wilcoxon_per_fold(all_a, all_b)
    print(f"\n  POOLED over {len(deltas)} (base x model x seed) paired points:", flush=True)
    print(f"    mean delta = {deltas.mean():+.4f}   median = {np.median(deltas):+.4f}   "
          f"fraction positive = {(deltas>0).mean():.0%}", flush=True)
    print(f"    Wilcoxon signed-rank p = {w['p_value']:.4g}", flush=True)

    # ---------- (1b) DeLong on seed-averaged predictions, per base set (ENS4) ----------
    print("\n=== DeLong test (seed-averaged ENS4 predictions, alerts vs base) ===", flush=True)
    for bset in base_sets:
        a_pred, b_pred = winner_pred[(bset, "ENS4")]
        dl = delong_test(y_test, a_pred, b_pred)
        print(f"  {bset:11s}: AUROC {dl['auc_b']:.4f} -> {dl['auc_a']:.4f}  "
              f"DeLong z={dl['z']:+.3f}  p={dl['p_value']:.4g}", flush=True)

    # ---------- (2a) univariate alert -> DILI enrichment on TRAIN_VAL ----------
    print("\n=== Which alerts carry DILI signal? (TRAIN_VAL, Fisher exact) ===", flush=True)
    trv_rows = [smi2i[s] for s in trv_df["Drug"]]
    A = blocks["alerts"][trv_rows]
    y = trv_df["Y"].values.astype(int)
    base_rate = y.mean()
    enr = []
    for j, fname in enumerate(FEATURE_NAMES):
        present = A[:, j] > 0
        n_pres = int(present.sum())
        if n_pres < 3 or n_pres > len(y) - 3:
            continue
        rate_pres = y[present].mean()
        rate_abs = y[~present].mean()
        tab = [[int(y[present].sum()), int((~y.astype(bool))[present].sum()) if False else int((1 - y)[present].sum())],
               [int(y[~present].sum()), int((1 - y)[~present].sum())]]
        _, p = stats.fisher_exact(tab)
        enr.append({"alert": fname, "n_present": n_pres,
                    "dili_rate_present": float(rate_pres), "dili_rate_absent": float(rate_abs),
                    "enrichment": float(rate_pres - rate_abs), "fisher_p": float(p)})
    enr_df = pd.DataFrame(enr).sort_values("enrichment", ascending=False)
    enr_df.to_csv(RES / "tdc_alerts_interpret.csv", index=False)
    print(f"  (base DILI rate = {base_rate:.0%})", flush=True)
    print("  Top DILI-enriched alerts (present -> DILI rate):", flush=True)
    for _, r in enr_df.head(8).iterrows():
        flag = " *" if r.fisher_p < 0.05 else ""
        print(f"    {r.alert:18s} n={int(r.n_present):3d}  "
              f"{r.dili_rate_present:.0%} vs {r.dili_rate_absent:.0%} absent  "
              f"(Fisher p={r.fisher_p:.3f}){flag}", flush=True)

    # ---------- (2b) XGBoost gain importance of alert columns ----------
    print("\n=== XGBoost gain importance: alert features in rich_alerts ===", flush=True)
    import xgboost as xgb
    kinds = ["desc", "morgan", "avalon", "erg", "maccs", "alerts"]
    Xcols = np.concatenate([blocks[k] for k in kinds], axis=1)
    Xtr = Xcols[trv_rows]
    n_non_alert = Xcols.shape[1] - len(FEATURE_NAMES)
    spw = float((y == 0).sum()) / max(int((y == 1).sum()), 1)
    clf = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                            subsample=0.85, colsample_bytree=0.7, scale_pos_weight=spw,
                            eval_metric="logloss", n_jobs=4, verbosity=0, random_state=0)
    clf.fit(Xtr, y)
    imp = clf.feature_importances_
    alert_imp = imp[n_non_alert:]
    order = np.argsort(alert_imp)[::-1]
    tot_alert = float(alert_imp.sum()); tot = float(imp.sum())
    print(f"  alert block holds {tot_alert/tot:.1%} of total feature importance "
          f"({len(FEATURE_NAMES)} of {Xcols.shape[1]} features)", flush=True)
    for j in order[:8]:
        if alert_imp[j] > 0:
            print(f"    {FEATURE_NAMES[j]:18s} gain={alert_imp[j]:.4f}", flush=True)

    print("\n[rigor] saved -> results/tdc_alerts_rigor.csv, tdc_alerts_interpret.csv", flush=True)


if __name__ == "__main__":
    main()
