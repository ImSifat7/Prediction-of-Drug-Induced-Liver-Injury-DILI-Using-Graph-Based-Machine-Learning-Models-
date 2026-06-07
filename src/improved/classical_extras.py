"""Supervisor-requested extensions to the classical-ML pipeline.

Three additions on top of classical_ml.py:
  1. SMOTE oversampling on the TRAINING set only (supervisor's explicit ask).
     SMOTE is applied in the fingerprint+descriptor feature space, NOT on graphs
     (which would be ill-defined). Test set is never resampled.
  2. Mordred descriptors (~1500 features) in addition to RDKit descriptors.
  3. Ablation flags so each component (FP / RDKit / Mordred / SMOTE) can be toggled.

Outputs:
  results/classical_metrics_smote.csv     — comparison: no-resample vs SMOTE vs RandomOversample
  results/classical_metrics_mordred.csv   — comparison: RDKit vs Mordred vs both
  results/classical_smote_test_probs.npz  — SMOTE-trained model test probs
  results/classical_smote_val_probs.npz   — SMOTE-trained model val probs

Run:
    python -m src.improved.classical_extras
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler
import xgboost as xgb

warnings.filterwarnings("ignore")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.improved.data_utils import load_dataset, scaffold_split
    from src.improved.classical_ml import (
        morgan_fp, rdkit_descriptors, fit_rf, fit_lr, fit_xgb,
        best_threshold, metrics,
    )
else:
    from .data_utils import load_dataset, scaffold_split
    from .classical_ml import (
        morgan_fp, rdkit_descriptors, fit_rf, fit_lr, fit_xgb,
        best_threshold, metrics,
    )


# ---------------------------------------------------------------------------
# Mordred descriptors  (~1500 features). Computed lazily because slow.
# ---------------------------------------------------------------------------

_MORDRED_CALC = None


def _get_mordred():
    global _MORDRED_CALC
    if _MORDRED_CALC is None:
        from mordred import Calculator, descriptors as md_desc
        # ignore_3D=True -> only 2D descriptors (no need for conformers)
        _MORDRED_CALC = Calculator(md_desc, ignore_3D=True)
    return _MORDRED_CALC


def mordred_features(smiles_list: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Compute Mordred 2D descriptors for a list of SMILES. Returns (X, feature_names)."""
    from rdkit import Chem
    calc = _get_mordred()
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    # nproc=1: Mordred's default multiprocessing re-imports numpy/scipy/pandas in
    # each worker, which on a paging-constrained Windows machine triggers
    # "paging file is too small" DLL load failures. Single-process is slower
    # (~2-3x) but reliable.
    df = calc.pandas(mols, quiet=True, nproc=1)
    # Coerce errors -> NaN -> 0, clip extreme values.
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    arr = np.clip(df.values.astype(np.float32), -1e6, 1e6)
    return arr, list(df.columns)


# ---------------------------------------------------------------------------
# Resampling helpers
# ---------------------------------------------------------------------------

def resample_train(X_tr: np.ndarray, y_tr: np.ndarray, method: str, seed: int = 42):
    if method == "none":
        return X_tr, y_tr
    if method == "smote":
        sm = SMOTE(random_state=seed, k_neighbors=min(5, max((y_tr == 0).sum(), (y_tr == 1).sum()) - 1))
        return sm.fit_resample(X_tr, y_tr)
    if method == "random":
        ros = RandomOverSampler(random_state=seed)
        return ros.fit_resample(X_tr, y_tr)
    raise ValueError(method)


# ---------------------------------------------------------------------------
# A small evaluation helper that fits multiple models and returns rows.
# ---------------------------------------------------------------------------

def fit_and_score_all(
    X_tr, y_tr, X_va, y_va, X_te, y_te,
    label: str,
    resample_method: str = "none",
    seed: int = 42,
) -> List[dict]:
    """Train XGB / RF / LR on the given feature matrix, with optional SMOTE on training set."""
    rows = []
    X_tr_rs, y_tr_rs = resample_train(X_tr, y_tr, resample_method, seed=seed)
    print(f"  [{label}/{resample_method}]  train rows: {len(y_tr)} -> {len(y_tr_rs)} after resample", flush=True)

    # XGB
    clf = fit_xgb(X_tr_rs, y_tr_rs, X_va, y_va)
    p_va = clf.predict_proba(X_va)[:, 1]
    p_te = clf.predict_proba(X_te)[:, 1]
    t = best_threshold(y_va, p_va, target="MCC")
    m = metrics(y_te, p_te, threshold=t)
    rows.append({"feature": label, "resample": resample_method, "model": "XGBoost", "threshold": t, **m,
                 "p_va": p_va, "p_te": p_te})

    # RF
    clf = fit_rf(X_tr_rs, y_tr_rs)
    p_va = clf.predict_proba(X_va)[:, 1]
    p_te = clf.predict_proba(X_te)[:, 1]
    t = best_threshold(y_va, p_va, target="MCC")
    m = metrics(y_te, p_te, threshold=t)
    rows.append({"feature": label, "resample": resample_method, "model": "RandomForest", "threshold": t, **m,
                 "p_va": p_va, "p_te": p_te})

    # LR (with scaling, since LR is sensitive to feature scale)
    sc = StandardScaler(with_mean=False)
    X_tr_s = sc.fit_transform(X_tr_rs)
    X_va_s = sc.transform(X_va)
    X_te_s = sc.transform(X_te)
    lr, _ = fit_lr(X_tr_s, y_tr_rs)
    p_va = lr.predict_proba(X_va_s)[:, 1]
    p_te = lr.predict_proba(X_te_s)[:, 1]
    t = best_threshold(y_va, p_va, target="MCC")
    m = metrics(y_te, p_te, threshold=t)
    rows.append({"feature": label, "resample": resample_method, "model": "LogReg", "threshold": t, **m,
                 "p_va": p_va, "p_te": p_te})

    return rows


def _print_row(r):
    print(f"  {r['feature']:<14} {r['resample']:<7} {r['model']:<13}  "
          f"thr={r['threshold']:.2f}  AUROC={r['AUROC']:.4f}  ACC={r['ACC']:.4f}  "
          f"F1={r['F1']:.4f}  MCC={r['MCC']:.4f}", flush=True)


def main():
    base = Path(__file__).resolve().parents[2]
    data_path = base / "data" / "dili_clean.csv"
    results_dir = base / "results"
    results_dir.mkdir(exist_ok=True)

    print("[extras] loading data", flush=True)
    graphs = load_dataset(data_path)
    smiles = [g.smiles for g in graphs]
    labels = np.array([int(g.y.item()) for g in graphs], dtype=np.int64)
    train_idx, val_idx, test_idx = scaffold_split(graphs, 0.6, 0.2, 0.2, seed=42)
    print(f"[extras] split  tr={len(train_idx)}  va={len(val_idx)}  te={len(test_idx)}", flush=True)

    smiles_tr = [smiles[i] for i in train_idx]
    smiles_va = [smiles[i] for i in val_idx]
    smiles_te = [smiles[i] for i in test_idx]
    y_tr, y_va, y_te = labels[train_idx], labels[val_idx], labels[test_idx]

    print("\n[extras] === featurizing ===", flush=True)
    t0 = time.time()
    fp_all = np.stack([morgan_fp(s) for s in smiles]).astype(np.float32)
    print(f"  Morgan FP: {fp_all.shape}  ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    rdk_all = np.stack([rdkit_descriptors(s) for s in smiles]).astype(np.float32)
    rdk_all = np.clip(rdk_all, -1e6, 1e6)
    print(f"  RDKit desc: {rdk_all.shape}  ({time.time()-t0:.1f}s)", flush=True)

    print("  computing Mordred descriptors (slow, ~3-5 min)…", flush=True)
    t0 = time.time()
    mor_all, mor_names = mordred_features(smiles)
    print(f"  Mordred: {mor_all.shape}  ({time.time()-t0:.1f}s)", flush=True)

    # Variance-filter once on the training rows to drop constant columns.
    def vfilter(X_all, train_rows):
        v = X_all[train_rows].std(axis=0)
        keep = v > 1e-6
        return X_all[:, keep], keep

    rdk_all, _ = vfilter(rdk_all, train_idx)
    mor_all, _ = vfilter(mor_all, train_idx)
    print(f"  after variance filter: RDKit={rdk_all.shape[1]}  Mordred={mor_all.shape[1]}", flush=True)

    # Feature-set combinations (the "ablation" the supervisor asked for).
    feature_sets: Dict[str, np.ndarray] = {
        "FP":              fp_all,
        "RDKit":           rdk_all,
        "Mordred":         mor_all,
        "FP+RDKit":        np.concatenate([fp_all, rdk_all], axis=1),
        "FP+Mordred":      np.concatenate([fp_all, mor_all], axis=1),
        "FP+RDKit+Mord":   np.concatenate([fp_all, rdk_all, mor_all], axis=1),
    }
    resample_methods = ["none", "smote", "random"]

    rows: List[dict] = []
    for feat_label, X_all in feature_sets.items():
        X_tr = X_all[train_idx]
        X_va = X_all[val_idx]
        X_te = X_all[test_idx]
        for rm in resample_methods:
            try:
                got = fit_and_score_all(X_tr, y_tr, X_va, y_va, X_te, y_te,
                                        label=feat_label, resample_method=rm, seed=42)
                for r in got:
                    _print_row(r)
                    rows.append(r)
            except Exception as e:
                print(f"  [{feat_label}/{rm}] failed: {e}", flush=True)

    # Write the ablation results.
    keep_cols = ["feature", "resample", "model", "threshold", "AUROC", "ACC", "F1", "MCC"]
    pd.DataFrame([{k: r[k] for k in keep_cols} for r in rows]).to_csv(
        results_dir / "classical_ablation.csv", index=False)

    # Pick the best SMOTE configuration (highest val-MCC) and save its probs for stacking.
    smote_rows = [r for r in rows if r["resample"] == "smote"]
    if smote_rows:
        best = max(smote_rows, key=lambda r: r["MCC"])
        print(f"\n[extras] best SMOTE config  feature={best['feature']}  model={best['model']}  "
              f"AUROC={best['AUROC']:.4f}  MCC={best['MCC']:.4f}", flush=True)
        np.savez(results_dir / "classical_smote_test_probs.npz",
                 y_true=y_te.astype(np.int64),
                 SMOTE_best=best["p_te"])
        np.savez(results_dir / "classical_smote_val_probs.npz",
                 y_true=y_va.astype(np.int64),
                 SMOTE_best=best["p_va"])

    print(f"\n[extras] wrote {results_dir/'classical_ablation.csv'}", flush=True)


if __name__ == "__main__":
    main()
