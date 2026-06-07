"""Calibration + advanced threshold search on existing stack predictions (Option K).

What this does, on existing prediction NPZs (no retraining needed):
  1. Loads every saved prediction stream (GNN, ChemBERTa, Hybrid, MolFormer,
     Tox21, Specialists).
  2. Builds the curated 6-way and 5-way ensemble probabilities exactly as
     stack_final.py reports them as best.
  3. Calibrates each ensemble probability via isotonic regression fit on val.
  4. Searches thresholds independently for MCC, ACC, and F1 over a finer grid,
     reporting whichever metric the threshold was tuned for.
  5. Also tries Platt scaling and temperature scaling for comparison.

Run:
    python -m src.improved.calibrate
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def _load(path: Path):
    if not path.exists():
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _best_thr_grid(y, p, metric):
    fns = {"MCC": matthews_corrcoef, "ACC": accuracy_score,
           "F1": lambda y, yp: f1_score(y, yp, zero_division=0)}
    fn = fns[metric]
    best_t, best_v = 0.5, -1.0
    for t in np.linspace(0.005, 0.995, 991):
        v = fn(y, (p >= t).astype(int))
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


def _platt(p_v, y_v, p_t):
    lr = LogisticRegression(C=1.0)
    lr.fit(p_v.reshape(-1, 1), y_v)
    return lr.predict_proba(p_v.reshape(-1, 1))[:, 1], lr.predict_proba(p_t.reshape(-1, 1))[:, 1]


def _iso(p_v, y_v, p_t):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
    iso.fit(p_v, y_v)
    return iso.predict(p_v), iso.predict(p_t)


def _temp(p_v, y_v, p_t):
    """Temperature scaling on logits. p must be in (0,1)."""
    eps = 1e-6
    logit_v = np.log(np.clip(p_v, eps, 1 - eps) / np.clip(1 - p_v, eps, 1 - eps))
    logit_t = np.log(np.clip(p_t, eps, 1 - eps) / np.clip(1 - p_t, eps, 1 - eps))
    best_T, best_loss = 1.0, np.inf
    for T in np.linspace(0.5, 3.0, 26):
        z = logit_v / T
        p = 1.0 / (1.0 + np.exp(-z))
        # negative log-likelihood
        nll = -(y_v * np.log(np.clip(p, eps, 1)) + (1 - y_v) * np.log(np.clip(1 - p, eps, 1))).mean()
        if nll < best_loss:
            best_loss, best_T = float(nll), float(T)
    return (1.0 / (1.0 + np.exp(-logit_v / best_T))), (1.0 / (1.0 + np.exp(-logit_t / best_T))), best_T


def main():
    base = Path(__file__).resolve().parents[2]
    res = base / "results"

    cls_v = _load(res / "classical_val_probs.npz")
    cls_t = _load(res / "classical_test_probs.npz")
    gnn_v = _load(res / "gnn_val_probs.npz")
    gnn_t = _load(res / "gnn_test_probs.npz")
    cb_v = _load(res / "chemberta_val_probs.npz")
    cb_t = _load(res / "chemberta_test_probs.npz")
    hy_v = _load(res / "hybrid_val_probs.npz")
    hy_t = _load(res / "hybrid_test_probs.npz")
    mf_v = _load(res / "molformer_val_probs.npz")
    mf_t = _load(res / "molformer_test_probs.npz")
    tx_v = _load(res / "tox21_val_probs.npz")
    tx_t = _load(res / "tox21_test_probs.npz")
    sp_v = _load(res / "specialists_val_probs.npz")
    sp_t = _load(res / "specialists_test_probs.npz")

    y_val = cls_v["y_true"]
    y_test = cls_t["y_true"]

    # Rebuild the curated ensemble streams matching stack_final.
    cb_cols = [k for k in cb_v.keys() if k.startswith("ChemBERTa_")]
    gnn_models = ["GCN", "GAT", "GraphSAGE", "GIN", "MPNN"]
    cb_vm = np.mean([cb_v[c] for c in cb_cols], axis=0)
    cb_tm = np.mean([cb_t[c] for c in cb_cols], axis=0)
    gn_vm = np.mean([gnn_v[c] for c in gnn_models], axis=0)
    gn_tm = np.mean([gnn_t[c] for c in gnn_models], axis=0)
    hy_vm = hy_v["hybrid"]; hy_tm = hy_t["hybrid"]
    mf_vm = mf_v["molformer"]; mf_tm = mf_t["molformer"]
    tx_cols = [k for k in tx_v.keys() if k.startswith("tox21_") and k != "tox21_mean"]
    tx_vm = np.mean([tx_v[c] for c in tx_cols], axis=0)
    tx_tm = np.mean([tx_t[c] for c in tx_cols], axis=0)
    sp_vm = np.mean([sp_v["specialists_hard"], sp_v["specialists_soft"]], axis=0)
    sp_tm = np.mean([sp_t["specialists_hard"], sp_t["specialists_soft"]], axis=0)

    streams = {
        "6way-no-gnn6": (
            0.20*cb_vm + 0.25*hy_vm + 0.25*mf_vm + 0.15*tx_vm + 0.15*sp_vm,
            0.20*cb_tm + 0.25*hy_tm + 0.25*mf_tm + 0.15*tx_tm + 0.15*sp_tm,
        ),
        "5way-no-gnn5-tx-light": (
            0.20*cb_vm + 0.30*hy_vm + 0.30*mf_vm + 0.20*tx_vm,
            0.20*cb_tm + 0.30*hy_tm + 0.30*mf_tm + 0.20*tx_tm,
        ),
        "4way-mf-light": (
            0.10*gn_vm + 0.30*cb_vm + 0.30*hy_vm + 0.30*mf_vm,
            0.10*gn_tm + 0.30*cb_tm + 0.30*hy_tm + 0.30*mf_tm,
        ),
        "3way-no-gnn": (
            0.5*cb_vm + 0.5*hy_vm,
            0.5*cb_tm + 0.5*hy_tm,
        ),
    }

    rows = []
    print("[calibrate] base (no calibration), threshold tuned per-metric:")
    for name, (pv, pt) in streams.items():
        for tgt in ("MCC", "ACC", "F1"):
            thr = _best_thr_grid(y_val, pv, tgt)
            m = _m(y_test, pt, thr)
            rows.append({"stream": name, "calib": "none", "target": tgt, "threshold": thr, **m})
        print(f"  {name}: AUROC={rows[-3]['AUROC']:.4f}  thr-MCC={rows[-3]['threshold']:.3f}  "
              f"MCC={rows[-3]['MCC']:.4f}  ACC={rows[-2]['ACC']:.4f}  F1={rows[-1]['F1']:.4f}")

    print("\n[calibrate] with isotonic calibration on val, then re-tune threshold:")
    for name, (pv, pt) in streams.items():
        pv_iso, pt_iso = _iso(pv, y_val, pt)
        for tgt in ("MCC", "ACC", "F1"):
            thr = _best_thr_grid(y_val, pv_iso, tgt)
            m = _m(y_test, pt_iso, thr)
            rows.append({"stream": name, "calib": "isotonic", "target": tgt, "threshold": thr, **m})
        print(f"  {name}: thr-MCC={rows[-3]['threshold']:.3f}  MCC={rows[-3]['MCC']:.4f}  "
              f"ACC={rows[-2]['ACC']:.4f}  F1={rows[-1]['F1']:.4f}")

    print("\n[calibrate] with Platt scaling:")
    for name, (pv, pt) in streams.items():
        pv_pl, pt_pl = _platt(pv, y_val, pt)
        for tgt in ("MCC", "ACC", "F1"):
            thr = _best_thr_grid(y_val, pv_pl, tgt)
            m = _m(y_test, pt_pl, thr)
            rows.append({"stream": name, "calib": "platt", "target": tgt, "threshold": thr, **m})
        print(f"  {name}: thr-MCC={rows[-3]['threshold']:.3f}  MCC={rows[-3]['MCC']:.4f}  "
              f"ACC={rows[-2]['ACC']:.4f}  F1={rows[-1]['F1']:.4f}")

    print("\n[calibrate] with temperature scaling:")
    for name, (pv, pt) in streams.items():
        pv_ts, pt_ts, T = _temp(pv, y_val, pt)
        for tgt in ("MCC", "ACC", "F1"):
            thr = _best_thr_grid(y_val, pv_ts, tgt)
            m = _m(y_test, pt_ts, thr)
            rows.append({"stream": name, "calib": f"temp(T={T:.2f})", "target": tgt, "threshold": thr, **m})
        print(f"  {name} (T={T:.2f}): thr-MCC={rows[-3]['threshold']:.3f}  MCC={rows[-3]['MCC']:.4f}  "
              f"ACC={rows[-2]['ACC']:.4f}  F1={rows[-1]['F1']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(base / "results" / "calibration_metrics.csv", index=False)

    print("\n[calibrate] Top 10 by MCC (target=MCC):")
    print(df[df["target"] == "MCC"].sort_values("MCC", ascending=False).head(10).to_string(index=False))
    print("\n[calibrate] Top 10 by ACC (target=ACC):")
    print(df[df["target"] == "ACC"].sort_values("ACC", ascending=False).head(10).to_string(index=False))
    print(f"\n[calibrate] wrote results/calibration_metrics.csv")


if __name__ == "__main__":
    main()
