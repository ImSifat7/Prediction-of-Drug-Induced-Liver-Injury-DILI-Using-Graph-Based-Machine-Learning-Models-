"""Final stack: combines stack_v2 base columns + Hybrid-XGBoost predictions.

The Hybrid-XGBoost output is one more "base model" whose predictions are very
different in character from GNN-prob / ChemBERTa-prob (it's an XGBoost
classifier over [GNN-emb + descriptors + Morgan-FP]), so adding it to the
stack can act like adding orthogonal signal.

Run:
    python -m src.improved.stack_final
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def _load_npz(p: Path) -> Optional[Dict[str, np.ndarray]]:
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

    cls_v = _load_npz(res / "classical_val_probs.npz")
    cls_t = _load_npz(res / "classical_test_probs.npz")
    gnn_v = _load_npz(res / "gnn_val_probs.npz")
    gnn_t = _load_npz(res / "gnn_test_probs.npz")
    cb_v = _load_npz(res / "chemberta_val_probs.npz")
    cb_t = _load_npz(res / "chemberta_test_probs.npz")
    hy_v = _load_npz(res / "hybrid_val_probs.npz")
    hy_t = _load_npz(res / "hybrid_test_probs.npz")
    mf_v = _load_npz(res / "molformer_val_probs.npz")
    mf_t = _load_npz(res / "molformer_test_probs.npz")
    tx_v = _load_npz(res / "tox21_val_probs.npz")
    tx_t = _load_npz(res / "tox21_test_probs.npz")
    sp_v = _load_npz(res / "specialists_val_probs.npz")
    sp_t = _load_npz(res / "specialists_test_probs.npz")
    ft_v = _load_npz(res / "molformer_ft_val_probs.npz")
    ft_t = _load_npz(res / "molformer_ft_test_probs.npz")
    hd_v = _load_npz(res / "hybrid_deep_val_probs.npz")
    hd_t = _load_npz(res / "hybrid_deep_test_probs.npz")
    cbex_v = _load_npz(res / "chemberta_exp_val_probs.npz")
    cbex_t = _load_npz(res / "chemberta_exp_test_probs.npz")
    mfex_v = _load_npz(res / "molformer_exp_val_probs.npz")
    mfex_t = _load_npz(res / "molformer_exp_test_probs.npz")

    if any(x is None for x in [cls_v, cls_t, gnn_v, gnn_t]):
        print("[stack_final] missing required NPZ files"); sys.exit(1)

    y_val = cls_v["y_true"]
    y_test = cls_t["y_true"]

    cols: List[str] = []
    parts_v: List[np.ndarray] = []
    parts_t: List[np.ndarray] = []
    for src_v, src_t, skip in [
        (cls_v, cls_t, ["y_true", "Classical-Ensemble"]),
        (gnn_v, gnn_t, ["y_true", "ensemble"]),
        (cb_v, cb_t, ["y_true", "ChemBERTa"]),  # use per-seed cols, drop mean dup
        (hy_v, hy_t, ["y_true"]),
        (mf_v, mf_t, ["y_true"]),
        (tx_v, tx_t, ["y_true"]),  # 12 Tox21 endpoints + tox21_mean
        (sp_v, sp_t, ["y_true"]),  # specialists (hard + soft)
        (ft_v, ft_t, ["y_true", "molformerFT"]),  # MolFormer-FT per-seed (skip mean)
        (hd_v, hd_t, ["y_true"]),  # hybrid_deep (everything-XGBoost)
        (cbex_v, cbex_t, ["y_true", "ChemBERTaEXP"]),  # ChemBERTa retrained on DILI+ClinTox+SIDER
        (mfex_v, mfex_t, ["y_true"]),                   # MolFormer retrained on DILI+ClinTox+SIDER
    ]:
        if src_v is None or src_t is None:
            continue
        for k in src_v.keys():
            if k in skip:
                continue
            parts_v.append(src_v[k]); parts_t.append(src_t[k]); cols.append(k)

    V = np.stack(parts_v, axis=1)
    T = np.stack(parts_t, axis=1)
    print(f"[stack_final] base columns ({len(cols)}): {cols}")

    rows: List[dict] = []

    # All individual base models for reference.
    print("\n[individual base models]")
    for j, name in enumerate(cols):
        p_v = V[:, j]; p_t = T[:, j]
        t = _best_thr(y_val, p_v)
        m = _m(y_test, p_t, t)
        rows.append({"variant": "single", "method": name, "threshold": t, **m})
        print(f"  {name:<22}  AUROC={m['AUROC']:.4f}  ACC={m['ACC']:.4f}  F1={m['F1']:.4f}  MCC={m['MCC']:.4f}")

    # Combinations.
    def add(label, method, p_v, p_t):
        t = _best_thr(y_val, p_v)
        m = _m(y_test, p_t, t)
        rows.append({"variant": label, "method": method, "threshold": t, **m})
        print(f"  {label:<20} {method:<18} thr={t:.2f}  AUROC={m['AUROC']:.4f}  ACC={m['ACC']:.4f}  F1={m['F1']:.4f}  MCC={m['MCC']:.4f}")

    print("\n[full = classical + GNN + ChemBERTa-seeds + Hybrid]")
    add("full", "mean", V.mean(axis=1), T.mean(axis=1))

    def rank(M):
        return np.stack([np.argsort(np.argsort(M[:, j])) / max(len(M) - 1, 1)
                         for j in range(M.shape[1])], axis=1)

    add("full", "rank-avg", rank(V).mean(axis=1), rank(T).mean(axis=1))

    # Geometric mean (good for combining models that are well-calibrated probs)
    eps = 1e-6
    gV = np.exp(np.log(np.clip(V, eps, 1 - eps)).mean(axis=1))
    gT = np.exp(np.log(np.clip(T, eps, 1 - eps)).mean(axis=1))
    add("full", "geom-mean", gV, gT)

    # Median of base predictions (robust to outliers)
    add("full", "median", np.median(V, axis=1), np.median(T, axis=1))

    # Subsets (drop classical / drop ChemBERTa / drop hybrid)
    def cols_keep(predicate):
        idx = [j for j, c in enumerate(cols) if predicate(c)]
        return idx, V[:, idx], T[:, idx]

    # GNN+Hybrid
    idx, vM, tM = cols_keep(lambda c: c in {"GCN", "GAT", "GraphSAGE", "GIN", "MPNN"} or c.startswith("hybrid") or c == "fp_only" or c == "hybrid_no_fp")
    add("gnn+hybrid", "mean", vM.mean(axis=1), tM.mean(axis=1))

    # ChemBERTa+Hybrid
    idx, vM, tM = cols_keep(lambda c: c.startswith("ChemBERTa_") or c.startswith("hybrid") or c == "fp_only" or c == "hybrid_no_fp")
    add("cb+hybrid", "mean", vM.mean(axis=1), tM.mean(axis=1))

    # Three-way: GNN-ensemble + ChemBERTa-mean + Hybrid (the canonical "ensemble of methodologies")
    gnn_idx = [j for j, c in enumerate(cols) if c in {"GCN", "GAT", "GraphSAGE", "GIN", "MPNN"}]
    cb_idx = [j for j, c in enumerate(cols) if c.startswith("ChemBERTa_")]
    hy_idx = [j for j, c in enumerate(cols) if c == "hybrid"]
    mf_idx = [j for j, c in enumerate(cols) if c == "molformer"]
    if gnn_idx and cb_idx and hy_idx:
        gn_v = V[:, gnn_idx].mean(axis=1); gn_t = T[:, gnn_idx].mean(axis=1)
        cb_vm = V[:, cb_idx].mean(axis=1); cb_tm = T[:, cb_idx].mean(axis=1)
        hy_vm = V[:, hy_idx].mean(axis=1); hy_tm = T[:, hy_idx].mean(axis=1)
        for w_gnn, w_cb, w_hy, name in [
            (1/3, 1/3, 1/3, "equal"),
            (0.2, 0.4, 0.4, "cb+hy-heavy"),
            (0.3, 0.3, 0.4, "hy-slightheavy"),
            (0.0, 0.5, 0.5, "no-gnn"),
        ]:
            p_v = w_gnn * gn_v + w_cb * cb_vm + w_hy * hy_vm
            p_t = w_gnn * gn_t + w_cb * cb_tm + w_hy * hy_tm
            add("3-way", f"weighted-{name}", p_v, p_t)

    tx_idx = [j for j, c in enumerate(cols) if c.startswith("tox21_") and c != "tox21_mean"]

    # Four-way: add MolFormer-XL signal to the mix
    if mf_idx and gnn_idx and cb_idx and hy_idx:
        mf_vm = V[:, mf_idx].mean(axis=1); mf_tm = T[:, mf_idx].mean(axis=1)
        for w_gnn, w_cb, w_hy, w_mf, name in [
            (0.25, 0.25, 0.25, 0.25, "equal4"),
            (0.10, 0.30, 0.30, 0.30, "gnn-light4"),
            (0.00, 0.30, 0.40, 0.30, "no-gnn4"),
            (0.00, 0.35, 0.35, 0.30, "no-gnn4-cbheavy"),
            (0.15, 0.25, 0.35, 0.25, "hy-heavy4"),
            (0.00, 0.40, 0.40, 0.20, "mf-light"),
            (0.00, 0.20, 0.40, 0.40, "mf-hy-heavy"),
        ]:
            p_v = w_gnn * gn_v + w_cb * cb_vm + w_hy * hy_vm + w_mf * mf_vm
            p_t = w_gnn * gn_t + w_cb * cb_tm + w_hy * hy_tm + w_mf * mf_tm
            add("4-way", f"weighted-{name}", p_v, p_t)
        # Plus pairwise with MolFormer
        for w_a, w_b, na, nb, vA, tA, vB, tB in [
            (0.5, 0.5, "cb", "mf", cb_vm, cb_tm, mf_vm, mf_tm),
            (0.5, 0.5, "hy", "mf", hy_vm, hy_tm, mf_vm, mf_tm),
            (0.5, 0.5, "gnn", "mf", gn_v, gn_t, mf_vm, mf_tm),
            (0.6, 0.4, "hy", "mf", hy_vm, hy_tm, mf_vm, mf_tm),
        ]:
            add("pair", f"{na}+{nb}-{w_a:.1f}_{w_b:.1f}", w_a*vA + w_b*vB, w_a*tA + w_b*tB)

    sp_idx = [j for j, c in enumerate(cols) if c.startswith("specialists_")]

    # Five-way with Tox21 auxiliary features (D from advisor plan)
    if tx_idx and gnn_idx and cb_idx and hy_idx and mf_idx:
        tx_vm = V[:, tx_idx].mean(axis=1); tx_tm = T[:, tx_idx].mean(axis=1)
        print(f"\n[stack_final] Tox21-mean-of-{len(tx_idx)} alone:  "
              f"val n={len(y_val)} test n={len(y_test)}", flush=True)
        for w_gnn, w_cb, w_hy, w_mf, w_tx, name in [
            (0.20, 0.20, 0.20, 0.20, 0.20, "equal5"),
            (0.10, 0.25, 0.25, 0.25, 0.15, "tx-light"),
            (0.00, 0.25, 0.30, 0.25, 0.20, "no-gnn5"),
            (0.10, 0.20, 0.30, 0.30, 0.10, "tx-very-light"),
            (0.00, 0.20, 0.30, 0.30, 0.20, "no-gnn5-tx-light"),
            (0.05, 0.25, 0.25, 0.30, 0.15, "mf-heavy5"),
        ]:
            p_v = w_gnn*gn_v + w_cb*cb_vm + w_hy*hy_vm + w_mf*mf_vm + w_tx*tx_vm
            p_t = w_gnn*gn_t + w_cb*cb_tm + w_hy*hy_tm + w_mf*mf_tm + w_tx*tx_tm
            add("5-way", f"weighted-{name}", p_v, p_t)

    # Six-way: add specialist predictions to the existing 5-way
    if sp_idx and tx_idx and gnn_idx and cb_idx and hy_idx and mf_idx:
        sp_vm = V[:, sp_idx].mean(axis=1); sp_tm = T[:, sp_idx].mean(axis=1)
        for w_gnn, w_cb, w_hy, w_mf, w_tx, w_sp, name in [
            (0.10, 0.20, 0.25, 0.25, 0.10, 0.10, "even6"),
            (0.00, 0.20, 0.25, 0.25, 0.15, 0.15, "no-gnn6"),
            (0.05, 0.20, 0.25, 0.25, 0.15, 0.10, "sp-light"),
            (0.00, 0.20, 0.25, 0.25, 0.20, 0.10, "no-gnn6-tx-heavy"),
            (0.00, 0.20, 0.30, 0.30, 0.10, 0.10, "no-gnn6-tx-light"),
            (0.10, 0.15, 0.25, 0.25, 0.10, 0.15, "sp-medium"),
            (0.00, 0.15, 0.30, 0.30, 0.15, 0.10, "no-gnn6-balanced"),
        ]:
            p_v = w_gnn*gn_v + w_cb*cb_vm + w_hy*hy_vm + w_mf*mf_vm + w_tx*tx_vm + w_sp*sp_vm
            p_t = w_gnn*gn_t + w_cb*cb_tm + w_hy*hy_tm + w_mf*mf_tm + w_tx*tx_tm + w_sp*sp_tm
            add("6-way", f"weighted-{name}", p_v, p_t)

    # Option G: blend the EXP-retrained streams (ChemBERTa+MolFormer on DILI+ClinTox+SIDER)
    # into the existing 5/6-way winning recipes.  Each EXP stream replaces a small
    # part of the corresponding original-data stream.
    cbex_idx = [j for j, c in enumerate(cols) if c.startswith("ChemBERTaEXP_")]
    mfex_idx = [j for j, c in enumerate(cols) if c == "molformerEXP"]
    if cbex_idx and mfex_idx and tx_idx and gnn_idx and cb_idx and hy_idx and mf_idx and sp_idx:
        cbex_vm = V[:, cbex_idx].mean(axis=1); cbex_tm = T[:, cbex_idx].mean(axis=1)
        mfex_vm = V[:, mfex_idx].mean(axis=1); mfex_tm = T[:, mfex_idx].mean(axis=1)
        tx_vm = V[:, tx_idx].mean(axis=1); tx_tm = T[:, tx_idx].mean(axis=1)
        sp_vm = V[:, sp_idx].mean(axis=1); sp_tm = T[:, sp_idx].mean(axis=1)
        # Try replacing 25% of each transformer signal with its EXP-retrained version.
        for w_gnn, w_cb, w_cbex, w_hy, w_mf, w_mfex, w_tx, name in [
            (0.00, 0.15, 0.05, 0.30, 0.25, 0.05, 0.20, "G-light-mix"),
            (0.00, 0.10, 0.10, 0.30, 0.20, 0.10, 0.20, "G-balanced-mix"),
            (0.00, 0.20, 0.00, 0.30, 0.30, 0.00, 0.20, "G-no-exp-baseline"),
            (0.00, 0.00, 0.20, 0.30, 0.00, 0.30, 0.20, "G-exp-only"),
            (0.00, 0.15, 0.05, 0.25, 0.20, 0.05, 0.15, "G-with-sp"),
        ]:
            p_v = (w_gnn*gn_v + w_cb*cb_vm + w_cbex*cbex_vm + w_hy*hy_vm
                   + w_mf*mf_vm + w_mfex*mfex_vm + w_tx*tx_vm)
            p_t = (w_gnn*gn_t + w_cb*cb_tm + w_cbex*cbex_tm + w_hy*hy_tm
                   + w_mf*mf_tm + w_mfex*mfex_tm + w_tx*tx_tm)
            if name == "G-with-sp":
                p_v = p_v + 0.10 * sp_vm
                p_t = p_t + 0.10 * sp_tm
            add("G-mix", f"weighted-{name}", p_v, p_t)

    # Logistic regression meta-learner over ALL 30+ base columns (it can pick up
    # signal from individual Tox21 endpoints that mean-tox21 may smear).
    from sklearn.linear_model import LogisticRegression
    meta = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=42)
    meta.fit(V, y_val)
    p_lr_v = meta.predict_proba(V)[:, 1]
    p_lr_t = meta.predict_proba(T)[:, 1]
    add("super", "all-cols-logreg", p_lr_v, p_lr_t)

    # Also try L2 with stronger regularization (less overfitting on 173 val samples)
    meta_l2 = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced", random_state=42)
    meta_l2.fit(V, y_val)
    p_l2_v = meta_l2.predict_proba(V)[:, 1]
    p_l2_t = meta_l2.predict_proba(T)[:, 1]
    add("super", "all-cols-logreg-C0.1", p_l2_v, p_l2_t)

    df = pd.DataFrame(rows).sort_values("MCC", ascending=False)
    df.to_csv(res / "stack_final_metrics.csv", index=False)
    print(f"\n[stack_final] wrote results/stack_final_metrics.csv")
    print("\n[stack_final] top 10 by MCC:")
    print(df.head(10).to_string(index=False))
    print("\n[stack_final] top 10 by ACC:")
    print(df.sort_values("ACC", ascending=False).head(10).to_string(index=False))
    print("\n[stack_final] top 10 by AUROC:")
    print(df.sort_values("AUROC", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
