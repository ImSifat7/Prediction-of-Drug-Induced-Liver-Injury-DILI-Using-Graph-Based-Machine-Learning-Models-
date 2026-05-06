"""Stacking meta-learner over GNN + classical-ML predictions.

How it works:
  - Inputs: per-model val + test probabilities saved to results/*.npz by:
        run_pipeline.py        -> gnn_val_probs.npz / gnn_test_probs.npz
        classical_ml.py        -> classical_val_probs.npz / classical_test_probs.npz
  - Meta-learner: logistic regression fit on val-set base-model probabilities,
    then evaluated on test. Threshold tuned on val for MCC.
  - Reports several configurations so you can see exactly what each layer adds:
        1) classical-only stack
        2) GNN-only stack  (same models you already trained)
        3) full stack (classical + GNN)
        4) plain mean (no meta-learner) for comparison

Run (after running both run_pipeline.py and classical_ml.py):
    python -m src.improved.stack
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def _load_npz(path: Path) -> Optional[Dict[str, np.ndarray]]:
    if not path.exists():
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _best_thr(y: np.ndarray, p: np.ndarray, target: str = "MCC") -> float:
    best_t, best_v = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 81):
        yp = (p >= t).astype(int)
        v = (matthews_corrcoef(y, yp) if target == "MCC"
             else f1_score(y, yp, zero_division=0) if target == "F1"
             else accuracy_score(y, yp))
        if v > best_v:
            best_v, best_t = float(v), float(t)
    return best_t


def _metrics(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    yp = (p >= t).astype(int)
    return {
        "AUROC": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
        "ACC": float(accuracy_score(y, yp)),
        "F1": float(f1_score(y, yp, zero_division=0)),
        "MCC": float(matthews_corrcoef(y, yp)),
    }


def _stack(val_mat: np.ndarray, y_val: np.ndarray,
           test_mat: np.ndarray, y_test: np.ndarray,
           label: str) -> Tuple[dict, np.ndarray]:
    """Fit logistic regression meta-learner on val, evaluate on test."""
    meta = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=42)
    meta.fit(val_mat, y_val)
    p_val = meta.predict_proba(val_mat)[:, 1]
    p_test = meta.predict_proba(test_mat)[:, 1]
    t = _best_thr(y_val, p_val, target="MCC")
    m = _metrics(y_test, p_test, t)
    return {"variant": label, "method": "logreg-stack", "threshold": t, **m}, p_test


def _mean(val_mat: np.ndarray, y_val: np.ndarray,
          test_mat: np.ndarray, y_test: np.ndarray,
          label: str) -> Tuple[dict, np.ndarray]:
    p_val = val_mat.mean(axis=1)
    p_test = test_mat.mean(axis=1)
    t = _best_thr(y_val, p_val, target="MCC")
    m = _metrics(y_test, p_test, t)
    return {"variant": label, "method": "mean", "threshold": t, **m}, p_test


def _columns(d: Dict[str, np.ndarray], skip: List[str]) -> List[str]:
    return [k for k in d.keys() if k not in skip]


def main():
    base = Path(__file__).resolve().parents[2]
    results_dir = base / "results"

    cls_val = _load_npz(results_dir / "classical_val_probs.npz")
    cls_test = _load_npz(results_dir / "classical_test_probs.npz")
    gnn_val = _load_npz(results_dir / "gnn_val_probs.npz")
    gnn_test = _load_npz(results_dir / "gnn_test_probs.npz")
    cb_val = _load_npz(results_dir / "chemberta_val_probs.npz")
    cb_test = _load_npz(results_dir / "chemberta_test_probs.npz")

    if cls_val is None or cls_test is None:
        print("[stack] classical_*_probs.npz missing — run `python -m src.improved.classical_ml` first")
        sys.exit(1)

    have_gnn = gnn_val is not None and gnn_test is not None
    have_cb = cb_val is not None and cb_test is not None
    if not have_gnn:
        print("[stack] gnn_*_probs.npz missing — run `python -m src.improved.run_pipeline` to generate them.")
        print("[stack]   Continuing with classical-only stack.")
    if not have_cb:
        print("[stack] chemberta_*_probs.npz missing — run `python -m src.improved.chemberta` to generate them.")
        print("[stack]   Continuing without ChemBERTa.")

    # Use the classical files' y_true as the source of truth for both val and test.
    y_val = cls_val["y_true"]
    y_test = cls_test["y_true"]

    if have_gnn:
        # Sanity check: both pipelines must use the same scaffold split.
        if not (np.array_equal(gnn_val["y_true"], y_val) and np.array_equal(gnn_test["y_true"], y_test)):
            print("[stack] WARNING: GNN and classical y_true mismatch — different splits?")
            print(f"  val sizes: gnn={len(gnn_val['y_true'])} cls={len(y_val)}")
            print(f"  test sizes: gnn={len(gnn_test['y_true'])} cls={len(y_test)}")
            print("  Aborting; ensure both pipelines use scaffold_split(...,seed=42).")
            sys.exit(1)

    if have_cb:
        if not (np.array_equal(cb_val["y_true"], y_val) and np.array_equal(cb_test["y_true"], y_test)):
            print("[stack] WARNING: ChemBERTa y_true mismatch — different splits? Aborting.")
            sys.exit(1)

    rows: List[dict] = []

    # --- 1) Classical-only stack ---
    cls_cols = _columns(cls_val, skip=["y_true", "Classical-Ensemble"])
    cls_val_mat = np.stack([cls_val[c] for c in cls_cols], axis=1)
    cls_test_mat = np.stack([cls_test[c] for c in cls_cols], axis=1)
    print(f"\n[stack] classical-only  base models = {cls_cols}")
    r, _ = _mean(cls_val_mat, y_val, cls_test_mat, y_test, label="classical-only")
    rows.append(r)
    print(f"  mean              thr={r['threshold']:.2f}  AUROC={r['AUROC']:.4f}  ACC={r['ACC']:.4f}  F1={r['F1']:.4f}  MCC={r['MCC']:.4f}")
    r, _ = _stack(cls_val_mat, y_val, cls_test_mat, y_test, label="classical-only")
    rows.append(r)
    print(f"  logreg-stack      thr={r['threshold']:.2f}  AUROC={r['AUROC']:.4f}  ACC={r['ACC']:.4f}  F1={r['F1']:.4f}  MCC={r['MCC']:.4f}")

    if have_gnn:
        # --- 2) GNN-only stack ---
        gnn_cols = _columns(gnn_val, skip=["y_true", "ensemble"])
        gnn_val_mat = np.stack([gnn_val[c] for c in gnn_cols], axis=1)
        gnn_test_mat = np.stack([gnn_test[c] for c in gnn_cols], axis=1)
        print(f"\n[stack] gnn-only  base models = {gnn_cols}")
        r, _ = _mean(gnn_val_mat, y_val, gnn_test_mat, y_test, label="gnn-only")
        rows.append(r)
        print(f"  mean              thr={r['threshold']:.2f}  AUROC={r['AUROC']:.4f}  ACC={r['ACC']:.4f}  F1={r['F1']:.4f}  MCC={r['MCC']:.4f}")
        r, _ = _stack(gnn_val_mat, y_val, gnn_test_mat, y_test, label="gnn-only")
        rows.append(r)
        print(f"  logreg-stack      thr={r['threshold']:.2f}  AUROC={r['AUROC']:.4f}  ACC={r['ACC']:.4f}  F1={r['F1']:.4f}  MCC={r['MCC']:.4f}")

    if have_cb:
        # --- 2b) ChemBERTa standalone ---
        cb_cols = _columns(cb_val, skip=["y_true"])
        cb_val_mat = np.stack([cb_val[c] for c in cb_cols], axis=1)
        cb_test_mat = np.stack([cb_test[c] for c in cb_cols], axis=1)
        print(f"\n[stack] chemberta-only  base models = {cb_cols}")
        # With one base model, mean is just the model itself.
        r, _ = _mean(cb_val_mat, y_val, cb_test_mat, y_test, label="chemberta-only")
        rows.append(r)
        print(f"  standalone        thr={r['threshold']:.2f}  AUROC={r['AUROC']:.4f}  ACC={r['ACC']:.4f}  F1={r['F1']:.4f}  MCC={r['MCC']:.4f}")

    if have_gnn:
        # --- 3) Full stack: classical + GNN (+ ChemBERTa if available) ---
        parts_val = [cls_val_mat, gnn_val_mat]
        parts_test = [cls_test_mat, gnn_test_mat]
        full_cols = cls_cols + gnn_cols
        if have_cb:
            parts_val.append(cb_val_mat)
            parts_test.append(cb_test_mat)
            full_cols = full_cols + cb_cols
        full_val_mat = np.concatenate(parts_val, axis=1)
        full_test_mat = np.concatenate(parts_test, axis=1)
        print(f"\n[stack] full  base models = {full_cols}")
        r, p_full = _mean(full_val_mat, y_val, full_test_mat, y_test, label="full")
        rows.append(r)
        print(f"  mean              thr={r['threshold']:.2f}  AUROC={r['AUROC']:.4f}  ACC={r['ACC']:.4f}  F1={r['F1']:.4f}  MCC={r['MCC']:.4f}")
        r, p_full_lr = _stack(full_val_mat, y_val, full_test_mat, y_test, label="full")
        rows.append(r)
        print(f"  logreg-stack      thr={r['threshold']:.2f}  AUROC={r['AUROC']:.4f}  ACC={r['ACC']:.4f}  F1={r['F1']:.4f}  MCC={r['MCC']:.4f}")

        # Save full-stack final predictions for further analysis.
        np.savez(
            results_dir / "stack_final_probs.npz",
            y_true=y_test,
            full_mean=p_full,
            full_logreg=p_full_lr,
        )

    pd.DataFrame(rows).to_csv(results_dir / "stack_metrics.csv", index=False)
    print(f"\n[stack] wrote {results_dir/'stack_metrics.csv'}")


if __name__ == "__main__":
    main()
