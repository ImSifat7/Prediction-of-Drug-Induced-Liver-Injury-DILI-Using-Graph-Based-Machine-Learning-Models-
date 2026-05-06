"""Quick sanity checks for the statistical-test module (no torch needed).

Run:  python -m src.improved.test_stats
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.improved.stats_utils import delong_test, wilcoxon_per_fold, mcnemar_test, auroc_ci
else:
    from .stats_utils import delong_test, wilcoxon_per_fold, mcnemar_test, auroc_ci

import numpy as np


def test_delong_matches_sklearn_for_single_model():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = rng.random(200) * 0.4 + 0.3 * y
    res = delong_test(y, p, p)
    sk_auc = roc_auc_score(y, p)
    assert abs(res["auc_a"] - sk_auc) < 1e-6, (res["auc_a"], sk_auc)
    assert abs(res["auc_b"] - sk_auc) < 1e-6
    assert res["p_value"] >= 0.99 or res["p_value"] != res["p_value"]  # NaN ok when var<=0
    print(f"  delong AUC matches sklearn: {res['auc_a']:.6f} vs {sk_auc:.6f}")


def test_delong_detects_difference():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    pa = np.clip(rng.random(400) * 0.4 + 0.5 * y, 0, 1)   # informative
    pb = rng.random(400)                                   # random
    res = delong_test(y, pa, pb)
    print(f"  delong informative vs random: auc_a={res['auc_a']:.3f}  auc_b={res['auc_b']:.3f}  p={res['p_value']:.4f}")
    assert res["auc_a"] > res["auc_b"]
    assert res["p_value"] < 0.05


def test_wilcoxon():
    a = [0.70, 0.72, 0.71, 0.73, 0.69]
    b = [0.65, 0.68, 0.67, 0.66, 0.64]
    res = wilcoxon_per_fold(a, b)
    print(f"  wilcoxon: stat={res['stat']:.3f}  p={res['p_value']:.4f}")
    assert res["p_value"] < 0.1


def test_mcnemar():
    y = [0, 1, 1, 0, 1, 1, 0, 0, 1, 1]
    a = [0, 1, 1, 0, 1, 1, 0, 0, 0, 0]   # 2 wrong on positives
    b = [1, 1, 1, 0, 1, 1, 1, 1, 1, 1]   # 3 wrong on negatives
    res = mcnemar_test(y, a, b)
    print(f"  mcnemar: chi2={res['chi2']:.3f}  p={res['p_value']:.4f}  n01={res['n01']}  n10={res['n10']}")


def test_auroc_ci():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    p = np.clip(rng.random(300) * 0.5 + 0.4 * y, 0, 1)
    auc, lo, hi = auroc_ci(y, p, n_bootstrap=200, seed=0)
    print(f"  auroc CI: {auc:.3f}  [{lo:.3f}, {hi:.3f}]")
    assert lo <= auc <= hi


if __name__ == "__main__":
    print("Running stats sanity checks...")
    test_delong_matches_sklearn_for_single_model()
    test_delong_detects_difference()
    test_wilcoxon()
    test_mcnemar()
    test_auroc_ci()
    print("All stats tests passed.")
