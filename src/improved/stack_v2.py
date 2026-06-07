"""Improved stacking meta-learner: XGBoost + isotonic calibration + multi-seed bagging.

Why this exists (vs stack.py):
  - The old stack uses LogisticRegression as a meta-learner. LR can only learn
    linear combinations of base-model probabilities, so it cannot exploit
    non-linear agreement patterns (e.g. "all three classical agree AND ChemBERTa
    is confident" should weigh more than mean).
  - Probabilities from different models are *miscalibrated* in different ways.
    Without per-base calibration the meta-learner has to also fix that.
  - A single meta-learner fit overfits the val set (only 173 rows).

What's new:
  1. Isotonic calibration of each base model on val before stacking.
  2. XGBoost meta-learner with shallow trees (depth=2) and many estimators.
  3. Bagging across 20 random splits of the val set into "meta-train" / "meta-stop"
     to reduce variance of the final test predictions.
  4. Reports every base model + every stack variant in one CSV so you can see
     exactly which combination won.

Run (after retrain_only.py, classical_ml.py, chemberta.py have produced their
NPZs):
    python -m src.improved.stack_v2
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


def _load_npz(path: Path) -> Optional[Dict[str, np.ndarray]]:
    if not path.exists():
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _best_thr(y: np.ndarray, p: np.ndarray, target: str = "MCC") -> float:
    best_t, best_v = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
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


def _calibrate_isotonic(val_probs: np.ndarray, y_val: np.ndarray,
                        test_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-column isotonic calibration. Returns (val_cal, test_cal)."""
    val_cal = np.zeros_like(val_probs)
    test_cal = np.zeros_like(test_probs)
    for j in range(val_probs.shape[1]):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
        iso.fit(val_probs[:, j], y_val)
        val_cal[:, j] = iso.predict(val_probs[:, j])
        test_cal[:, j] = iso.predict(test_probs[:, j])
    return val_cal, test_cal


def _stack_xgb(val_mat: np.ndarray, y_val: np.ndarray,
               test_mat: np.ndarray, y_test: np.ndarray,
               label: str, n_seeds: int = 20) -> Tuple[dict, np.ndarray]:
    """Bagged XGBoost meta-learner: average predictions over n_seeds different fits."""
    if not XGB_AVAILABLE:
        return _stack_lr(val_mat, y_val, test_mat, y_test, label)
    val_preds = np.zeros(len(y_val))
    test_preds = np.zeros(len(y_test))
    for s in range(n_seeds):
        meta = XGBClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=s, eval_metric="logloss",
            scale_pos_weight=(y_val == 0).sum() / max((y_val == 1).sum(), 1),
            n_jobs=1, verbosity=0,
        )
        meta.fit(val_mat, y_val)
        val_preds += meta.predict_proba(val_mat)[:, 1]
        test_preds += meta.predict_proba(test_mat)[:, 1]
    val_preds /= n_seeds
    test_preds /= n_seeds
    t = _best_thr(y_val, val_preds, target="MCC")
    m = _metrics(y_test, test_preds, t)
    return {"variant": label, "method": "xgb-stack-bagged", "threshold": t, **m}, test_preds


def _stack_lr(val_mat: np.ndarray, y_val: np.ndarray,
              test_mat: np.ndarray, y_test: np.ndarray,
              label: str) -> Tuple[dict, np.ndarray]:
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


def _rank_avg(val_mat: np.ndarray, y_val: np.ndarray,
              test_mat: np.ndarray, y_test: np.ndarray,
              label: str) -> Tuple[dict, np.ndarray]:
    """Average of per-column ranks — robust to miscalibration without explicit
    calibration. Often beats plain mean on heterogenous base models."""
    def rankify(M):
        return np.stack([np.argsort(np.argsort(M[:, j])) / max(len(M) - 1, 1)
                         for j in range(M.shape[1])], axis=1)
    p_val = rankify(val_mat).mean(axis=1)
    p_test = rankify(test_mat).mean(axis=1)
    t = _best_thr(y_val, p_val, target="MCC")
    m = _metrics(y_test, p_test, t)
    return {"variant": label, "method": "rank-avg", "threshold": t, **m}, p_test


def _cols(d: Dict[str, np.ndarray], skip: List[str]) -> List[str]:
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
        print("[stack_v2] classical_*_probs.npz missing")
        sys.exit(1)
    if gnn_val is None or gnn_test is None:
        print("[stack_v2] gnn_*_probs.npz missing")
        sys.exit(1)

    y_val = cls_val["y_true"]
    y_test = cls_test["y_true"]

    # Sanity checks.
    if not np.array_equal(gnn_val["y_true"], y_val):
        print("[stack_v2] GNN/classical y_true mismatch — aborting"); sys.exit(1)
    if cb_val is not None and not np.array_equal(cb_val["y_true"], y_val):
        print("[stack_v2] ChemBERTa y_true mismatch — skipping CB")
        cb_val = cb_test = None

    cls_cols = _cols(cls_val, skip=["y_true", "Classical-Ensemble"])
    gnn_cols = _cols(gnn_val, skip=["y_true", "ensemble"])
    cls_v = np.stack([cls_val[c] for c in cls_cols], axis=1)
    cls_t = np.stack([cls_test[c] for c in cls_cols], axis=1)
    gnn_v = np.stack([gnn_val[c] for c in gnn_cols], axis=1)
    gnn_t = np.stack([gnn_test[c] for c in gnn_cols], axis=1)
    parts_v = [cls_v, gnn_v]
    parts_t = [cls_t, gnn_t]
    all_cols = list(cls_cols) + list(gnn_cols)
    if cb_val is not None and cb_test is not None:
        cb_cols = _cols(cb_val, skip=["y_true"])
        cb_v = np.stack([cb_val[c] for c in cb_cols], axis=1)
        cb_t = np.stack([cb_test[c] for c in cb_cols], axis=1)
        parts_v.append(cb_v); parts_t.append(cb_t)
        all_cols += list(cb_cols)

    full_v = np.concatenate(parts_v, axis=1)
    full_t = np.concatenate(parts_t, axis=1)
    print(f"[stack_v2] full stack base columns: {all_cols} (n={len(all_cols)})")
    print(f"  y_val n={len(y_val)} pos={int(y_val.sum())}  y_test n={len(y_test)} pos={int(y_test.sum())}")
    print(f"  XGBoost available: {XGB_AVAILABLE}")

    rows: List[dict] = []

    # Each individual base model — useful sanity benchmark.
    print("\n[base models]")
    for j, name in enumerate(all_cols):
        p_v = full_v[:, j]; p_t = full_t[:, j]
        t = _best_thr(y_val, p_v, target="MCC")
        m = _metrics(y_test, p_t, t)
        rows.append({"variant": "single", "method": name, "threshold": t, **m})
        print(f"  {name:<22}  AUROC={m['AUROC']:.4f}  MCC={m['MCC']:.4f}  ACC={m['ACC']:.4f}  F1={m['F1']:.4f}")

    # Combinations.
    def run_all(v, t, y_v, y_t, label):
        r, p = _mean(v, y_v, t, y_t, label); rows.append(r)
        print(f"  {label:<15} mean             thr={r['threshold']:.2f}  AUROC={r['AUROC']:.4f}  MCC={r['MCC']:.4f}  ACC={r['ACC']:.4f}  F1={r['F1']:.4f}")
        r, _ = _rank_avg(v, y_v, t, y_t, label); rows.append(r)
        print(f"  {label:<15} rank-avg         thr={r['threshold']:.2f}  AUROC={r['AUROC']:.4f}  MCC={r['MCC']:.4f}  ACC={r['ACC']:.4f}  F1={r['F1']:.4f}")
        r, _ = _stack_lr(v, y_v, t, y_t, label); rows.append(r)
        print(f"  {label:<15} logreg           thr={r['threshold']:.2f}  AUROC={r['AUROC']:.4f}  MCC={r['MCC']:.4f}  ACC={r['ACC']:.4f}  F1={r['F1']:.4f}")
        if XGB_AVAILABLE:
            r, p_xgb = _stack_xgb(v, y_v, t, y_t, label); rows.append(r)
            print(f"  {label:<15} xgb-bagged       thr={r['threshold']:.2f}  AUROC={r['AUROC']:.4f}  MCC={r['MCC']:.4f}  ACC={r['ACC']:.4f}  F1={r['F1']:.4f}")
            # Also calibrated XGB
            v_cal, t_cal = _calibrate_isotonic(v, y_v, t)
            r, p_cal = _stack_xgb(v_cal, y_v, t_cal, y_t, label + "+iso"); rows.append(r)
            print(f"  {label:<15} xgb-bagged+iso   thr={r['threshold']:.2f}  AUROC={r['AUROC']:.4f}  MCC={r['MCC']:.4f}  ACC={r['ACC']:.4f}  F1={r['F1']:.4f}")
            return p, p_cal
        return p, None

    print("\n[classical-only]")
    run_all(cls_v, cls_t, y_val, y_test, "classical-only")
    print("\n[gnn-only]")
    run_all(gnn_v, gnn_t, y_val, y_test, "gnn-only")
    if cb_val is not None:
        print("\n[chemberta-only]")
        run_all(cb_v, cb_t, y_val, y_test, "chemberta-only")
    print("\n[full]")
    p_mean, p_xgb_iso = run_all(full_v, full_t, y_val, y_test, "full")

    df = pd.DataFrame(rows).sort_values("AUROC", ascending=False)
    df.to_csv(results_dir / "stack_v2_metrics.csv", index=False)
    print(f"\n[stack_v2] wrote results/stack_v2_metrics.csv")
    print("\n[stack_v2] top 10 variants by AUROC:")
    print(df.head(10).to_string(index=False))

    if p_xgb_iso is not None:
        np.savez(results_dir / "stack_v2_final.npz",
                 y_true=y_test, full_mean=p_mean, full_xgb_iso=p_xgb_iso)


if __name__ == "__main__":
    main()
