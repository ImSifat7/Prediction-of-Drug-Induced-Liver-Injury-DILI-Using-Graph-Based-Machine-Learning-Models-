"""Reusable, leakage-free evaluation utilities for the DILI project.

Provides the full metric panel (Accuracy, F1, MCC, Recall/Sensitivity, Specificity,
Precision, NPV, AUROC, PR-AUC, confusion matrix), percentile bootstrap 95% CIs for
every metric, a validation-only Youden's-J threshold selector, SMILES
standardisation, Bemis-Murcko scaffolds, train/external overlap reporting, and
max-Tanimoto chemical-space similarity.

These helpers NEVER look at test or external labels when choosing a threshold or a
hyper-parameter; the caller is responsible for passing only training/validation data
to the selection functions.
"""
from __future__ import annotations
from typing import Sequence

import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             f1_score, matthews_corrcoef, confusion_matrix, roc_curve,
                             brier_score_loss)

from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import rdFingerprintGenerator, inchi
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_LFC = rdMolStandardize.LargestFragmentChooser()


# ---------------------------------------------------------------- structures
def standardize_smiles(smi: str) -> str | None:
    """Canonical SMILES of the largest fragment (salt-stripped). None if invalid."""
    if not isinstance(smi, str) or not smi.strip():
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        m = _LFC.choose(m)
        m = rdMolStandardize.Uncharger().uncharge(m)
        return Chem.MolToSmiles(m)
    except Exception:
        try:
            return Chem.MolToSmiles(m)
        except Exception:
            return None


def inchikey14(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is None:
        return None
    try:
        return inchi.MolToInchiKey(m).split("-")[0]
    except Exception:
        return None


def scaffold(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception:
        return None


def morgan_fp(smi: str):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return _MORGAN.GetFingerprint(m) if m is not None else None


def max_tanimoto_to_train(ext_smiles: Sequence[str], train_smiles: Sequence[str]) -> np.ndarray:
    """Max Tanimoto (Morgan-2048) of each external molecule to any training molecule."""
    train_fps = [fp for fp in (morgan_fp(s) for s in train_smiles) if fp is not None]
    out = np.zeros(len(ext_smiles))
    for i, s in enumerate(ext_smiles):
        fp = morgan_fp(s)
        if fp is None or not train_fps:
            out[i] = np.nan
        else:
            out[i] = max(DataStructs.BulkTanimotoSimilarity(fp, train_fps))
    return out


# ---------------------------------------------------------------- thresholds
def youden_threshold(y_val: Sequence[int], p_val: Sequence[float]) -> float:
    """Threshold maximising Youden's J = sensitivity + specificity - 1, on validation only."""
    fpr, tpr, thr = roc_curve(y_val, p_val)
    j = tpr - fpr
    t = float(thr[int(np.argmax(j))])
    # roc_curve can return inf for the first threshold; clamp into (0,1)
    if not np.isfinite(t):
        t = 0.5
    return float(min(max(t, 1e-4), 1 - 1e-4))


# ---------------------------------------------------------------- metrics
def full_panel(y: Sequence[int], prob: Sequence[float], thr: float) -> dict:
    """Complete metric panel at a fixed threshold. Threshold-free metrics ignore thr."""
    y = np.asarray(y, dtype=int); prob = np.asarray(prob, dtype=float)
    yhat = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    return dict(
        AUROC=roc_auc_score(y, prob) if len(np.unique(y)) > 1 else float("nan"),
        PR_AUC=average_precision_score(y, prob) if len(np.unique(y)) > 1 else float("nan"),
        ACC=accuracy_score(y, yhat), F1=f1_score(y, yhat, zero_division=0),
        MCC=matthews_corrcoef(y, yhat) if len(np.unique(yhat)) > 1 else 0.0,
        Sensitivity=sens, Specificity=spec, Precision=prec, NPV=npv,
        Brier=brier_score_loss(y, prob),
        TP=int(tp), TN=int(tn), FP=int(fp), FN=int(fn), n=int(len(y)), pos=int(y.sum()),
    )


_CI_KEYS = ["AUROC", "PR_AUC", "ACC", "F1", "MCC", "Sensitivity", "Specificity"]


def bootstrap_ci(y: Sequence[int], prob: Sequence[float], thr: float,
                 n_boot: int = 2000, seed: int = 42) -> dict:
    """Percentile 95% bootstrap CI for each metric in _CI_KEYS, at a fixed threshold."""
    y = np.asarray(y, dtype=int); prob = np.asarray(prob, dtype=float)
    rng = np.random.default_rng(seed); n = len(y)
    acc = {k: [] for k in _CI_KEYS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        m = full_panel(y[idx], prob[idx], thr)
        for k in _CI_KEYS:
            acc[k].append(m[k])
    return {k: (round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3))
            if v else (float("nan"), float("nan")) for k, v in acc.items()}


def error_type(y_true: int, y_pred: int) -> str:
    return {(1, 1): "TP", (0, 0): "TN", (0, 1): "FP", (1, 0): "FN"}[(int(y_true), int(y_pred))]
