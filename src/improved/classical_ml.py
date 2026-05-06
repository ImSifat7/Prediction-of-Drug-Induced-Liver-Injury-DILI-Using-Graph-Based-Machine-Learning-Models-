"""Classical ML baselines on top of Morgan fingerprints + RDKit descriptors.

Why this exists: on small chem datasets (n<2000), tree models on rich molecular
features are *very* competitive with GNNs and often better. They are also a
prerequisite for the stacking ensemble in `stack.py`.

Models trained here (same scaffold split as the GNN pipeline so results are comparable):
  - XGBoost on (Morgan FP || RDKit descriptors)
  - Random Forest on (Morgan FP || RDKit descriptors)
  - Logistic Regression on (Morgan FP || RDKit descriptors)
  - SVM (RBF) on RDKit descriptors only (FP makes it too slow)

Outputs:
  results/classical_metrics.csv       — per-model AUROC/ACC/F1/MCC on test
  results/classical_test_probs.npz    — per-model test probabilities for stacking
  results/classical_val_probs.npz     — per-model val probabilities (for stacking meta-learner)

Run:
    python -m src.improved.classical_ml
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
import xgboost as xgb

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.improved.data_utils import (
        load_dataset,
        scaffold_split,
        standardize_smiles,
    )
else:
    from .data_utils import load_dataset, scaffold_split, standardize_smiles


MORGAN_BITS = 2048
MORGAN_RADIUS = 2


def morgan_fp(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(MORGAN_BITS, dtype=np.uint8)
    bv = GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_BITS)
    arr = np.zeros((MORGAN_BITS,), dtype=np.uint8)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(bv, arr)
    return arr


# Pick a stable subset of RDKit descriptors. Some descriptors fail on weird mols;
# we wrap each in try/except and substitute 0 to keep the pipeline robust.
_DESC_NAMES = [name for name, _ in Descriptors._descList]


def rdkit_descriptors(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(len(_DESC_NAMES), dtype=np.float32)
    out = np.zeros(len(_DESC_NAMES), dtype=np.float32)
    for i, (_, fn) in enumerate(Descriptors._descList):
        try:
            v = fn(mol)
            if v is None or np.isnan(v) or np.isinf(v):
                v = 0.0
            out[i] = float(v)
        except Exception:
            out[i] = 0.0
    return out


def featurize(smiles_list: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Morgan FP, RDKit descriptors, combined) feature matrices."""
    fps = np.stack([morgan_fp(s) for s in smiles_list]).astype(np.float32)
    descs = np.stack([rdkit_descriptors(s) for s in smiles_list]).astype(np.float32)
    # clip absurd descriptor values (some descriptors blow up on weird mols)
    descs = np.clip(descs, -1e6, 1e6)
    combined = np.concatenate([fps, descs], axis=1)
    return fps, descs, combined


def metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "AUROC": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else float("nan"),
        "ACC": float(accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }


def best_threshold(y_true: np.ndarray, y_prob: np.ndarray, target: str = "F1") -> float:
    """Sweep thresholds and pick the one that maximizes the target metric on val."""
    best_t, best_v = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 81):
        y_pred = (y_prob >= t).astype(int)
        if target == "F1":
            v = f1_score(y_true, y_pred, zero_division=0)
        elif target == "MCC":
            v = matthews_corrcoef(y_true, y_pred)
        elif target == "ACC":
            v = accuracy_score(y_true, y_pred)
        else:
            raise ValueError(target)
        if v > best_v:
            best_v, best_t = float(v), float(t)
    return best_t


def fit_xgb(X_tr, y_tr, X_va, y_va) -> xgb.XGBClassifier:
    pos = float((y_tr == 1).sum())
    neg = float((y_tr == 0).sum())
    spw = neg / max(pos, 1.0)
    clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=1.0,
        scale_pos_weight=spw,
        eval_metric="auc",
        early_stopping_rounds=30,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return clf


def fit_rf(X_tr, y_tr) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_tr, y_tr)
    return clf


def fit_lr(X_tr, y_tr) -> Tuple[LogisticRegression, StandardScaler]:
    sc = StandardScaler(with_mean=False)  # FP is sparse-ish; keep with_mean=False safe
    Xs = sc.fit_transform(X_tr)
    clf = LogisticRegression(
        C=0.5,
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )
    clf.fit(Xs, y_tr)
    return clf, sc


def fit_svm(X_tr, y_tr) -> Tuple[SVC, StandardScaler]:
    sc = StandardScaler()
    Xs = sc.fit_transform(X_tr)
    clf = SVC(
        C=1.0,
        kernel="rbf",
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(Xs, y_tr)
    return clf, sc


def main():
    base = Path(__file__).resolve().parents[2]
    data_path = base / "data" / "dili_clean.csv"
    results_dir = base / "results"
    results_dir.mkdir(exist_ok=True)

    print("[classical] loading dataset")
    graphs = load_dataset(data_path)

    # Same scaffold split as the GNN pipeline (seed=42, 60/20/20).
    train_idx, val_idx, test_idx = scaffold_split(graphs, 0.6, 0.2, 0.2, seed=42)
    print(f"[classical] split  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    smiles = [g.smiles for g in graphs]
    labels = np.array([int(g.y.item()) for g in graphs], dtype=np.int64)

    t0 = time.time()
    print("[classical] featurizing (Morgan FP + RDKit descriptors)…")
    fps, descs, combined = featurize(smiles)
    print(f"[classical]   FP shape={fps.shape}  desc shape={descs.shape}  combined={combined.shape}  ({time.time()-t0:.1f}s)")

    # Drop descriptor columns that are constant in train (saves SVM/LR time).
    desc_var = descs[train_idx].std(axis=0)
    keep = desc_var > 1e-6
    descs = descs[:, keep]
    combined = np.concatenate([fps, descs], axis=1)
    print(f"[classical]   after variance filter: desc={descs.shape[1]}  combined={combined.shape[1]}")

    Xtr, Xva, Xte = combined[train_idx], combined[val_idx], combined[test_idx]
    ytr, yva, yte = labels[train_idx], labels[val_idx], labels[test_idx]

    Xtr_d, Xva_d, Xte_d = descs[train_idx], descs[val_idx], descs[test_idx]

    rows = []
    test_probs = {}
    val_probs = {}

    # --- XGBoost (combined features) ---
    print("\n[classical] training XGBoost (FP+desc)…")
    xgb_clf = fit_xgb(Xtr, ytr, Xva, yva)
    p_va = xgb_clf.predict_proba(Xva)[:, 1]
    p_te = xgb_clf.predict_proba(Xte)[:, 1]
    t = best_threshold(yva, p_va, target="MCC")
    m = metrics(yte, p_te, threshold=t)
    print(f"  XGB     thr={t:.2f}  AUROC={m['AUROC']:.4f}  ACC={m['ACC']:.4f}  F1={m['F1']:.4f}  MCC={m['MCC']:.4f}")
    rows.append({"model": "XGBoost", "features": "FP+desc", "threshold": t, **m})
    test_probs["XGBoost"] = p_te
    val_probs["XGBoost"] = p_va

    # --- Random Forest (combined features) ---
    print("[classical] training RandomForest (FP+desc)…")
    rf_clf = fit_rf(Xtr, ytr)
    p_va = rf_clf.predict_proba(Xva)[:, 1]
    p_te = rf_clf.predict_proba(Xte)[:, 1]
    t = best_threshold(yva, p_va, target="MCC")
    m = metrics(yte, p_te, threshold=t)
    print(f"  RF      thr={t:.2f}  AUROC={m['AUROC']:.4f}  ACC={m['ACC']:.4f}  F1={m['F1']:.4f}  MCC={m['MCC']:.4f}")
    rows.append({"model": "RandomForest", "features": "FP+desc", "threshold": t, **m})
    test_probs["RandomForest"] = p_te
    val_probs["RandomForest"] = p_va

    # --- Logistic Regression on FP only (canonical baseline) ---
    print("[classical] training LogReg (Morgan FP)…")
    lr_clf, lr_sc = fit_lr(fps[train_idx], ytr)
    p_va = lr_clf.predict_proba(lr_sc.transform(fps[val_idx]))[:, 1]
    p_te = lr_clf.predict_proba(lr_sc.transform(fps[test_idx]))[:, 1]
    t = best_threshold(yva, p_va, target="MCC")
    m = metrics(yte, p_te, threshold=t)
    print(f"  LR-FP   thr={t:.2f}  AUROC={m['AUROC']:.4f}  ACC={m['ACC']:.4f}  F1={m['F1']:.4f}  MCC={m['MCC']:.4f}")
    rows.append({"model": "LogReg", "features": "FP", "threshold": t, **m})
    test_probs["LogReg"] = p_te
    val_probs["LogReg"] = p_va

    # --- SVM on descriptors only ---
    print("[classical] training SVM (RDKit descriptors)…")
    svm_clf, svm_sc = fit_svm(Xtr_d, ytr)
    p_va = svm_clf.predict_proba(svm_sc.transform(Xva_d))[:, 1]
    p_te = svm_clf.predict_proba(svm_sc.transform(Xte_d))[:, 1]
    t = best_threshold(yva, p_va, target="MCC")
    m = metrics(yte, p_te, threshold=t)
    print(f"  SVM-d   thr={t:.2f}  AUROC={m['AUROC']:.4f}  ACC={m['ACC']:.4f}  F1={m['F1']:.4f}  MCC={m['MCC']:.4f}")
    rows.append({"model": "SVM", "features": "desc", "threshold": t, **m})
    test_probs["SVM"] = p_te
    val_probs["SVM"] = p_va

    # --- Simple average ensemble of classical models ---
    avg_te = np.mean(list(test_probs.values()), axis=0)
    avg_va = np.mean(list(val_probs.values()), axis=0)
    t = best_threshold(yva, avg_va, target="MCC")
    m = metrics(yte, avg_te, threshold=t)
    print(f"\n[classical] ensemble (mean)  thr={t:.2f}  AUROC={m['AUROC']:.4f}  ACC={m['ACC']:.4f}  F1={m['F1']:.4f}  MCC={m['MCC']:.4f}")
    rows.append({"model": "Classical-Ensemble", "features": "mean-of-4", "threshold": t, **m})
    test_probs["Classical-Ensemble"] = avg_te
    val_probs["Classical-Ensemble"] = avg_va

    # --- Save outputs ---
    pd.DataFrame(rows).to_csv(results_dir / "classical_metrics.csv", index=False)
    np.savez(
        results_dir / "classical_test_probs.npz",
        y_true=yte,
        **test_probs,
    )
    np.savez(
        results_dir / "classical_val_probs.npz",
        y_true=yva,
        **val_probs,
    )
    print(f"\n[classical] wrote {results_dir/'classical_metrics.csv'}")
    print(f"[classical] wrote {results_dir/'classical_test_probs.npz'}")
    print(f"[classical] wrote {results_dir/'classical_val_probs.npz'}")
    print(f"[classical] elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
