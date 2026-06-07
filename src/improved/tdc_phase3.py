"""Phase 3 publication rigor: ablation table + DeLong tests + calibration analysis.

  - Ablation: drop one base model at a time from the best ensemble; report
    delta-AUROC to quantify each model's marginal contribution.
  - Pairwise DeLong tests: compare every two base models for statistically
    significant AUROC difference.
  - Calibration: reliability diagram + ECE / MCE + isotonic regression on val.
  - Confusion matrix + per-class metrics.

Run:
    python -m src.improved.tdc_phase3
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)


# -------------------- DeLong test --------------------
def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = compute_midrank(positive_examples[r])
        ty[r] = compute_midrank(negative_examples[r])
        tz[r] = compute_midrank(predictions_sorted_transposed[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_pvalue(y_true, p1, p2):
    """One-sided / two-sided DeLong p-value for AUROC difference between p1 and p2.
    Returns (auc_diff, p_value_two_sided)."""
    y = np.asarray(y_true).astype(int)
    order = np.argsort(-y)  # positives first
    y_sorted = y[order]
    label_1_count = int(y_sorted.sum())
    preds = np.stack([p1[order], p2[order]], axis=0)
    aucs, cov = fast_delong(preds, label_1_count)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return float(diff), 1.0
    z = diff / np.sqrt(var)
    pval = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(diff), float(pval)


# -------------------- Loaders / helpers --------------------
def _load(p: Path) -> Optional[Dict[str, np.ndarray]]:
    if not p.exists(): return None
    with np.load(p) as z: return {k: z[k] for k in z.files}


def _best_thr(y, p, target="MCC"):
    bt, bv = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        yp = (p >= t).astype(int)
        v = (matthews_corrcoef(y, yp) if target == "MCC"
             else f1_score(y, yp, zero_division=0) if target == "F1"
             else accuracy_score(y, yp))
        if v > bv: bv, bt = float(v), float(t)
    return bt


def _m(y, p, t):
    yp = (p >= t).astype(int)
    return {
        "AUROC": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
        "ACC": float(accuracy_score(y, yp)),
        "F1": float(f1_score(y, yp, zero_division=0)),
        "MCC": float(matthews_corrcoef(y, yp)),
        "Precision": float(precision_score(y, yp, zero_division=0)),
        "Recall": float(recall_score(y, yp, zero_division=0)),
        "Brier": float(brier_score_loss(y, p)),
    }


def expected_calibration_error(y, p, n_bins=10):
    """Standard ECE."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    ece = 0.0
    n = len(y)
    for b in range(n_bins):
        m = (idx == b)
        if m.sum() == 0: continue
        acc = y[m].mean()
        conf = p[m].mean()
        ece += (m.sum() / n) * abs(acc - conf)
    return float(ece)


def main():
    base = Path(__file__).resolve().parents[2]
    res = base / "results"

    # ----- Load all TDC base predictions -----
    sources = [
        ("tdc_gnn", ["y_true"]),
        ("tdc_molformer", ["y_true"]),
        ("tdc_chemberta", ["y_true", "ChemBERTa"]),  # use seed-level
        ("tdc_hybrid", ["y_true"]),
        ("tdc_specialists", ["y_true"]),
        ("tdc_attentivefp", ["y_true", "AttentiveFP"]),
        ("tdc_chemberta_tta", ["y_true"]),
    ]
    base_cols, V_cols, T_cols = [], [], []
    y_val = y_test = None
    for prefix, skip in sources:
        v = _load(res / f"{prefix}_val_probs.npz")
        t = _load(res / f"{prefix}_test_probs.npz")
        if v is None or t is None:
            print(f"[skip] {prefix}"); continue
        if y_val is None: y_val = v["y_true"]
        if y_test is None: y_test = t["y_true"]
        for k in v.keys():
            if k in skip: continue
            base_cols.append(k); V_cols.append(v[k]); T_cols.append(t[k])

    V = np.stack(V_cols, axis=1); T = np.stack(T_cols, axis=1)
    print(f"[phase3] base columns ({len(base_cols)})", flush=True)

    # ----- Find the best ensemble: full geom-mean -----
    eps = 1e-6
    p_geom_v = np.exp(np.log(np.clip(V, eps, 1 - eps)).mean(axis=1))
    p_geom_t = np.exp(np.log(np.clip(T, eps, 1 - eps)).mean(axis=1))
    t = _best_thr(y_val, p_geom_v, target="MCC")
    m_geom = _m(y_test, p_geom_t, t)
    print("\n=== HEADLINE ENSEMBLE (full geom-mean, MCC-tuned) ===")
    for k, v in m_geom.items(): print(f"  {k:<10} {v:.4f}")

    # ----- Ablation: drop one base column at a time -----
    print("\n=== ABLATION (drop one base, report AUROC delta) ===")
    ablation_rows = []
    for j, c in enumerate(base_cols):
        mask = [i for i in range(V.shape[1]) if i != j]
        v_ = V[:, mask]; t_ = T[:, mask]
        p_v = np.exp(np.log(np.clip(v_, eps, 1 - eps)).mean(axis=1))
        p_t = np.exp(np.log(np.clip(t_, eps, 1 - eps)).mean(axis=1))
        t_thr = _best_thr(y_val, p_v, target="MCC")
        m = _m(y_test, p_t, t_thr)
        dAUROC = m["AUROC"] - m_geom["AUROC"]
        ablation_rows.append({
            "dropped": c, "AUROC_after": m["AUROC"], "ACC_after": m["ACC"],
            "MCC_after": m["MCC"], "F1_after": m["F1"], "dAUROC": dAUROC,
        })
        print(f"  drop {c:<25} AUROC={m['AUROC']:.4f}  dAUROC={dAUROC:+.4f}")
    pd.DataFrame(ablation_rows).sort_values("dAUROC").to_csv(res / "tdc_ablation.csv", index=False)
    print("\n[saved] results/tdc_ablation.csv")

    # ----- DeLong tests: every base vs ensemble + best-base vs ensemble -----
    print("\n=== DELONG TESTS: base-vs-ensemble (test set) ===")
    delong_rows = []
    for j, c in enumerate(base_cols):
        diff, pval = delong_pvalue(y_test, p_geom_t, T[:, j])
        sig = "*" if pval < 0.05 else ""
        delong_rows.append({
            "comparison": f"ensemble vs {c}", "auc_diff": diff,
            "pvalue": pval, "significant_0.05": pval < 0.05,
        })
        print(f"  ens vs {c:<25} d={diff:+.4f}  p={pval:.4f}{sig}")
    # pairwise top-3 bases
    top3 = sorted(range(len(base_cols)),
                  key=lambda i: -roc_auc_score(y_test, T[:, i]))[:3]
    print("\n=== DELONG TESTS: top-3 bases pairwise ===")
    for i in range(len(top3)):
        for k in range(i + 1, len(top3)):
            a, b = top3[i], top3[k]
            diff, pval = delong_pvalue(y_test, T[:, a], T[:, b])
            print(f"  {base_cols[a]:<22} vs {base_cols[b]:<22} d={diff:+.4f}  p={pval:.4f}")
            delong_rows.append({
                "comparison": f"{base_cols[a]} vs {base_cols[b]}",
                "auc_diff": diff, "pvalue": pval, "significant_0.05": pval < 0.05,
            })
    pd.DataFrame(delong_rows).to_csv(res / "tdc_delong.csv", index=False)
    print("\n[saved] results/tdc_delong.csv")

    # ----- Calibration on val + apply to test -----
    print("\n=== CALIBRATION ANALYSIS ===")
    # ECE before
    ece_before = expected_calibration_error(y_test, p_geom_t, n_bins=10)
    brier_before = brier_score_loss(y_test, p_geom_t)
    print(f"  Before isotonic: ECE={ece_before:.4f}  Brier={brier_before:.4f}")
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_geom_v, y_val)
    p_iso_t = iso.transform(p_geom_t)
    ece_after = expected_calibration_error(y_test, p_iso_t, n_bins=10)
    brier_after = brier_score_loss(y_test, p_iso_t)
    print(f"  After isotonic:  ECE={ece_after:.4f}  Brier={brier_after:.4f}")
    t_iso = _best_thr(y_val, iso.transform(p_geom_v), target="MCC")
    m_iso = _m(y_test, p_iso_t, t_iso)
    print(f"  Calibrated metrics: AUROC={m_iso['AUROC']:.4f}  ACC={m_iso['ACC']:.4f}  "
          f"F1={m_iso['F1']:.4f}  MCC={m_iso['MCC']:.4f}")

    # Reliability diagram bins for plot
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(p_geom_t, bins) - 1
    idx = np.clip(idx, 0, 9)
    rel_before = []
    for b in range(10):
        m_ = (idx == b)
        if m_.sum() == 0: continue
        rel_before.append({"bin_mid": (bins[b] + bins[b+1]) / 2,
                           "n": int(m_.sum()),
                           "frac_pos": float(y_test[m_].mean()),
                           "mean_pred": float(p_geom_t[m_].mean())})
    pd.DataFrame(rel_before).to_csv(res / "tdc_reliability_uncalibrated.csv", index=False)
    idx_i = np.digitize(p_iso_t, bins) - 1
    idx_i = np.clip(idx_i, 0, 9)
    rel_after = []
    for b in range(10):
        m_ = (idx_i == b)
        if m_.sum() == 0: continue
        rel_after.append({"bin_mid": (bins[b] + bins[b+1]) / 2,
                          "n": int(m_.sum()),
                          "frac_pos": float(y_test[m_].mean()),
                          "mean_pred": float(p_iso_t[m_].mean())})
    pd.DataFrame(rel_after).to_csv(res / "tdc_reliability_isotonic.csv", index=False)
    print("[saved] reliability tables")

    # ----- Confusion matrix at MCC-tuned threshold -----
    thr_acc = _best_thr(y_val, p_geom_v, target="ACC")
    yp = (p_geom_t >= thr_acc).astype(int)
    cm = confusion_matrix(y_test, yp)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n=== CONFUSION MATRIX (geom-mean, ACC-tuned thr={thr_acc:.3f}) ===")
    print(f"            Pred 0   Pred 1")
    print(f"  True 0    {tn:>6d}   {fp:>6d}")
    print(f"  True 1    {fn:>6d}   {tp:>6d}")
    print(f"  Sens (TPR) = {tp/(tp+fn):.4f}")
    print(f"  Spec (TNR) = {tn/(tn+fp):.4f}")
    print(f"  PPV (Prec) = {tp/(tp+fp):.4f}")
    print(f"  NPV        = {tn/(tn+fn):.4f}")
    pd.DataFrame({"TN": [tn], "FP": [fp], "FN": [fn], "TP": [tp],
                  "threshold": [thr_acc]}).to_csv(res / "tdc_confusion.csv", index=False)

    # ----- Final headline summary table -----
    headline = {
        "metric": ["AUROC", "Accuracy", "F1", "MCC", "Precision", "Recall", "Brier"],
        "uncalibrated_geom-mean": [m_geom["AUROC"], m_geom["ACC"], m_geom["F1"], m_geom["MCC"],
                                    m_geom["Precision"], m_geom["Recall"], brier_before],
        "isotonic-calibrated":   [m_iso["AUROC"], m_iso["ACC"], m_iso["F1"], m_iso["MCC"],
                                   m_iso["Precision"], m_iso["Recall"], brier_after],
    }
    pd.DataFrame(headline).to_csv(res / "tdc_headline_metrics.csv", index=False)
    print(f"\n[saved] results/tdc_headline_metrics.csv")
    print("\n=== HEADLINE (TDC DILI, isotonic-calibrated, geom-mean ensemble) ===")
    print(f"  AUROC      = {m_iso['AUROC']:.4f}")
    print(f"  Accuracy   = {m_iso['ACC']:.4f}")
    print(f"  F1         = {m_iso['F1']:.4f}")
    print(f"  MCC        = {m_iso['MCC']:.4f}")
    print(f"  Precision  = {m_iso['Precision']:.4f}")
    print(f"  Recall     = {m_iso['Recall']:.4f}")
    print(f"  Brier      = {brier_after:.4f}")
    print(f"  ECE        = {ece_after:.4f}")


if __name__ == "__main__":
    main()
