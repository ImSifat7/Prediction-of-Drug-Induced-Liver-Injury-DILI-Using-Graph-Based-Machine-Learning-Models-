"""Statistical analysis for model comparison.

Functions:
- delong_test:   compares two correlated AUROCs on the same test set.
- wilcoxon_per_fold: paired non-parametric test across CV folds (or seeds).
- mcnemar_test:  compares per-sample correctness of two classifiers.
- auroc_ci:      bootstrap 95% CI for AUROC.

Reference: Sun & Xu (2014), "Fast implementation of DeLong's algorithm".
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from scipy import stats


def _midrank(x: np.ndarray) -> np.ndarray:
    """Midrank as in DeLong (handles ties by averaging ranks)."""
    order = np.argsort(x, kind="mergesort")
    ranked = np.empty(len(x), dtype=float)
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranked[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return ranked


def _delong_components(y_true: np.ndarray, probs: np.ndarray):
    pos = y_true == 1
    neg = ~pos
    m = int(pos.sum())
    n = int(neg.sum())
    x = probs[pos]
    y = probs[neg]
    tx = _midrank(x)
    ty = _midrank(y)
    tz = _midrank(np.concatenate([x, y]))
    auc = (tz[:m].sum() / m - (m + 1) / 2) / n
    v01 = (tz[:m] - tx) / n
    v10 = 1 - (tz[m:] - ty) / m
    return auc, v01, v10, m, n


def delong_test(y_true: Sequence[int], probs_a: Sequence[float], probs_b: Sequence[float]) -> dict:
    """Two-sided DeLong's test for the difference of two correlated AUROCs.

    Returns dict with z, p_value, auc_a, auc_b, var.
    """
    y = np.asarray(y_true, dtype=int)
    pa = np.asarray(probs_a, dtype=float)
    pb = np.asarray(probs_b, dtype=float)
    if not (len(y) == len(pa) == len(pb)):
        raise ValueError("array length mismatch")

    auc_a, v01a, v10a, m, n = _delong_components(y, pa)
    auc_b, v01b, v10b, _, _ = _delong_components(y, pb)
    s01 = np.cov(np.stack([v01a, v01b]))
    s10 = np.cov(np.stack([v10a, v10b]))
    s = s01 / m + s10 / n
    diff = auc_a - auc_b
    var = float(s[0, 0] + s[1, 1] - 2 * s[0, 1])
    if var <= 0:
        return {"z": float("nan"), "p_value": 1.0, "auc_a": float(auc_a), "auc_b": float(auc_b), "var": var}
    z = float(diff / np.sqrt(var))
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return {"z": z, "p_value": p, "auc_a": float(auc_a), "auc_b": float(auc_b), "var": var}


def wilcoxon_per_fold(scores_a: Sequence[float], scores_b: Sequence[float]) -> dict:
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if np.allclose(a, b):
        return {"stat": 0.0, "p_value": 1.0}
    try:
        stat, p = stats.wilcoxon(a, b, alternative="two-sided", zero_method="zsplit")
        return {"stat": float(stat), "p_value": float(p)}
    except ValueError:
        return {"stat": float("nan"), "p_value": 1.0}


def mcnemar_test(y_true: Sequence[int], pred_a: Sequence[int], pred_b: Sequence[int]) -> dict:
    """McNemar's test on paired classifier predictions."""
    y = np.asarray(y_true, dtype=int)
    a = np.asarray(pred_a, dtype=int)
    b = np.asarray(pred_b, dtype=int)
    a_correct = a == y
    b_correct = b == y
    n01 = int(np.sum(a_correct & ~b_correct))
    n10 = int(np.sum(~a_correct & b_correct))
    if n01 + n10 == 0:
        return {"chi2": 0.0, "p_value": 1.0, "n01": n01, "n10": n10}
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p = float(1 - stats.chi2.cdf(chi2, df=1))
    return {"chi2": float(chi2), "p_value": p, "n01": n01, "n10": n10}


def auroc_ci(y_true: Sequence[int], y_prob: Sequence[float], n_bootstrap: int = 1000, seed: int = 42) -> Tuple[float, float, float]:
    """Bootstrap 95% CI for AUROC. Returns (auroc, ci_low, ci_high)."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y_true)
    p = np.asarray(y_prob)
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    if not aucs:
        return float(roc_auc_score(y, p)), float("nan"), float("nan")
    arr = np.asarray(aucs)
    return float(roc_auc_score(y, p)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
