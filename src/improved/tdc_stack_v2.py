"""Stack v2 on TDC DILI: add AttentiveFP + ChemBERTa-TTA base columns,
plus an XGBoost meta-learner over all base columns (replaces logistic).

Run:
    python -m src.improved.tdc_stack_v2
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, roc_auc_score,
)
from xgboost import XGBClassifier


def _load(p: Path) -> Optional[Dict[str, np.ndarray]]:
    if not p.exists():
        return None
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def _best_thr(y, p, target="MCC"):
    best_t, best_v = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        yp = (p >= t).astype(int)
        v = (matthews_corrcoef(y, yp) if target == "MCC"
             else f1_score(y, yp, zero_division=0) if target == "F1"
             else accuracy_score(y, yp))
        if v > best_v:
            best_v, best_t = float(v), float(t)
    return best_t


def _m(y, p, t):
    yp = (p >= t).astype(int)
    return {
        "AUROC": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
        "ACC": float(accuracy_score(y, yp)),
        "F1": float(f1_score(y, yp, zero_division=0)),
        "MCC": float(matthews_corrcoef(y, yp)),
    }


def main():
    base = Path(__file__).resolve().parents[2]
    res = base / "results"

    # ---- Load all available base prediction files ----
    sources = [
        ("tdc_gnn", "_val_probs.npz", "_test_probs.npz", ["y_true"]),
        ("tdc_molformer", "_val_probs.npz", "_test_probs.npz", ["y_true"]),
        ("tdc_chemberta", "_val_probs.npz", "_test_probs.npz", ["y_true", "ChemBERTa"]),
        ("tdc_hybrid", "_val_probs.npz", "_test_probs.npz", ["y_true"]),
        ("tdc_specialists", "_val_probs.npz", "_test_probs.npz", ["y_true"]),
        ("tdc_attentivefp", "_val_probs.npz", "_test_probs.npz", ["y_true", "AttentiveFP"]),
        ("tdc_chemberta_tta", "_val_probs.npz", "_test_probs.npz", ["y_true"]),
    ]
    parts_v, parts_t, cols = [], [], []
    y_val, y_test = None, None
    for prefix, va_suf, te_suf, skip in sources:
        v = _load(res / f"{prefix}{va_suf}")
        t = _load(res / f"{prefix}{te_suf}")
        if v is None or t is None:
            print(f"  [skip] {prefix} not found"); continue
        if y_val is None: y_val = v["y_true"]
        if y_test is None: y_test = t["y_true"]
        for k in v.keys():
            if k in skip: continue
            parts_v.append(v[k]); parts_t.append(t[k]); cols.append(k)
            print(f"  + {k:<25}  val={v[k].shape}  test={t[k].shape}")

    V = np.stack(parts_v, axis=1); T = np.stack(parts_t, axis=1)
    print(f"\n[stack_v2] base columns ({len(cols)}): {cols}\n")

    rows = []
    def add(label, method, p_v, p_t):
        for tgt in ("MCC", "ACC", "F1"):
            t = _best_thr(y_val, p_v, target=tgt)
            m = _m(y_test, p_t, t)
            rows.append({"variant": label, "method": method, "tuned_for": tgt,
                         "threshold": t, **m})

    print("[individual base models]")
    for j, c in enumerate(cols):
        t = _best_thr(y_val, V[:, j])
        m = _m(y_test, T[:, j], t)
        print(f"  {c:<25}  AUROC={m['AUROC']:.4f}  ACC={m['ACC']:.4f}  "
              f"F1={m['F1']:.4f}  MCC={m['MCC']:.4f}")
        rows.append({"variant": "single", "method": c, "tuned_for": "MCC",
                     "threshold": t, **m})

    # All-column ensembles
    print("\n[full-column ensembles]")
    eps = 1e-6
    add("full", "mean", V.mean(axis=1), T.mean(axis=1))
    gV = np.exp(np.log(np.clip(V, eps, 1 - eps)).mean(axis=1))
    gT = np.exp(np.log(np.clip(T, eps, 1 - eps)).mean(axis=1))
    add("full", "geom-mean", gV, gT)
    add("full", "median", np.median(V, axis=1), np.median(T, axis=1))
    def rank(M):
        return np.stack([np.argsort(np.argsort(M[:, j])) / max(len(M) - 1, 1)
                         for j in range(M.shape[1])], axis=1)
    add("full", "rank-avg", rank(V).mean(axis=1), rank(T).mean(axis=1))

    # Reduced ensembles: collapse seed-level cols into one mean per base model family
    def gather(prefixes_or_eq):
        idx = []
        for j, c in enumerate(cols):
            for pref in prefixes_or_eq:
                if c == pref or c.startswith(pref + "_") or c.startswith(pref):
                    idx.append(j); break
        return idx

    gnn_idx = [j for j, c in enumerate(cols) if c in {"GCN", "GAT", "GraphSAGE", "GIN", "MPNN"}]
    afp_idx = [j for j, c in enumerate(cols) if c.startswith("AttentiveFP")]
    cb_idx = [j for j, c in enumerate(cols) if c.startswith("ChemBERTa_")]
    cbtta_idx = [j for j, c in enumerate(cols) if c == "ChemBERTaTTA" or c.startswith("CBTTA_")]
    mf_idx = [j for j, c in enumerate(cols) if c == "molformer"]
    hy_idx = [j for j, c in enumerate(cols) if c == "hybrid"]
    sp_idx = [j for j, c in enumerate(cols) if c == "specialists_soft" or c == "specialists"]

    def mean_block(idx):
        if not idx: return None, None
        return V[:, idx].mean(axis=1), T[:, idx].mean(axis=1)

    gn_v, gn_t = mean_block(gnn_idx)
    afp_v, afp_t = mean_block(afp_idx)
    cb_v, cb_t = mean_block(cb_idx)
    cbtta_v, cbtta_t = mean_block(cbtta_idx)
    mf_v, mf_t = mean_block(mf_idx)
    hy_v, hy_t = mean_block(hy_idx)
    sp_v, sp_t = mean_block(sp_idx)

    # Weighted blends
    print("\n[weighted blends with the new heads]")
    blends = [
        ("eq6", dict(gn=0.15, afp=0.20, cbtta=0.20, mf=0.15, hy=0.15, sp=0.15)),
        ("afp-heavy", dict(gn=0.10, afp=0.30, cbtta=0.20, mf=0.10, hy=0.15, sp=0.15)),
        ("transformer-heavy", dict(gn=0.05, afp=0.25, cbtta=0.30, mf=0.15, hy=0.15, sp=0.10)),
        ("no-gnn5b", dict(gn=0.0, afp=0.30, cbtta=0.25, mf=0.15, hy=0.20, sp=0.10)),
        ("afp-cbtta-hy", dict(gn=0.0, afp=0.40, cbtta=0.30, mf=0.0, hy=0.20, sp=0.10)),
        ("afp+gnn+cbtta", dict(gn=0.20, afp=0.30, cbtta=0.20, mf=0.10, hy=0.10, sp=0.10)),
        ("balanced7", dict(gn=0.15, afp=0.18, cbtta=0.17, mf=0.13, hy=0.15, sp=0.10)),
    ]
    for name, w in blends:
        if any(x is None for x in [gn_v, afp_v, cbtta_v, mf_v, hy_v, sp_v]): continue
        p_v = w["gn"]*gn_v + w["afp"]*afp_v + w["cbtta"]*cbtta_v + w["mf"]*mf_v + w["hy"]*hy_v + w["sp"]*sp_v
        p_t = w["gn"]*gn_t + w["afp"]*afp_t + w["cbtta"]*cbtta_t + w["mf"]*mf_t + w["hy"]*hy_t + w["sp"]*sp_t
        add("blend", name, p_v, p_t)

    # LR + XGBoost meta-learners
    print("\n[meta-learners over all base columns]")
    for C in [0.1, 1.0, 5.0]:
        m_ = LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=42)
        m_.fit(V, y_val)
        add("meta-LR", f"C={C}", m_.predict_proba(V)[:, 1], m_.predict_proba(T)[:, 1])

    pos = (y_val == 1).sum(); neg = (y_val == 0).sum()
    spw = float(neg) / max(int(pos), 1)
    # XGBoost meta with light regularization (val is small ~47 samples)
    for max_depth, n_est, lr in [(2, 100, 0.05), (3, 200, 0.05), (3, 100, 0.1)]:
        bag_v = np.zeros(len(y_val)); bag_t = np.zeros(len(y_test))
        for s in range(5):
            m_ = XGBClassifier(
                n_estimators=n_est, max_depth=max_depth, learning_rate=lr,
                subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.5, scale_pos_weight=spw,
                eval_metric="logloss", random_state=s, n_jobs=4, verbosity=0,
            )
            m_.fit(V, y_val)
            bag_v += m_.predict_proba(V)[:, 1]
            bag_t += m_.predict_proba(T)[:, 1]
        bag_v /= 5; bag_t /= 5
        add("meta-XGB", f"d{max_depth}_n{n_est}_lr{lr}", bag_v, bag_t)

    df = pd.DataFrame(rows)
    df.to_csv(res / "tdc_stack_v2_metrics.csv", index=False)

    print("\n[TOP 15 by AUROC]")
    print(df.sort_values("AUROC", ascending=False).head(15).to_string(index=False))
    print("\n[TOP 10 by ACC]")
    print(df.sort_values("ACC", ascending=False).head(10).to_string(index=False))
    print("\n[TOP 10 by MCC]")
    print(df.sort_values("MCC", ascending=False).head(10).to_string(index=False))
    print("\n[TOP 10 by F1]")
    print(df.sort_values("F1", ascending=False).head(10).to_string(index=False))

    best = df.sort_values("AUROC", ascending=False).iloc[0]
    print(f"\n[HEADLINE] best AUROC={best.AUROC:.4f}  ACC={best.ACC:.4f}  "
          f"F1={best.F1:.4f}  MCC={best.MCC:.4f}  ({best.variant}/{best.method}, tuned-for={best.tuned_for})")


if __name__ == "__main__":
    main()
