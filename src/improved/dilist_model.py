"""Train the model PROPERLY on the full official FDA DILIst (1279 drugs).

Unlike the easy, saturated TDC-DILI slice, the full DILIst set is the realistic
DILI task and still has headroom. We do a rigorous evaluation:
  (1) feature-set x model comparison with REPEATED stratified 5-fold CV (3 repeats),
  (2) best-config out-of-fold AUROC + bootstrap 95% CI,
  (3) a LEARNING CURVE (AUROC vs training-set fraction) to show whether more data
      still helps on this harder task.

Everything is leakage-clean (CV folds; features fit inside each fold's train).

Run:  python -m src.improved.dilist_model
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from rdkit import Chem, RDLogger

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.improved.tdc_official_v2 import precompute_blocks, make_model, RES
    from src.improved.stats_utils import auroc_ci
else:
    from .tdc_official_v2 import precompute_blocks, make_model, RES
    from .stats_utils import auroc_ci

BASE = Path(__file__).resolve().parents[2]


def standardize(df, smi_col, lab_col):
    rows = []
    for s, l in zip(df[smi_col], df[lab_col]):
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            continue
        try:
            lab = int(float(l))
        except Exception:
            continue
        rows.append((Chem.MolToSmiles(m), lab))
    out = pd.DataFrame(rows, columns=["smiles", "label"]).drop_duplicates("smiles").reset_index(drop=True)
    return out


def featurize(smiles, kinds):
    blocks = precompute_blocks(list(smiles), kinds)
    X = np.concatenate([blocks[k] for k in kinds], axis=1).astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(X, -1e6, 1e6).astype(np.float32)


def cv_oof(X, y, models, n_splits=5, n_repeats=3, seed=0):
    """Repeated stratified CV; returns mean/std fold AUROC and pooled OOF preds (single-repeat)."""
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    fold_aucs = []
    for tr, te in rkf.split(X, y):
        spw = float((y[tr] == 0).sum()) / max(int((y[tr] == 1).sum()), 1)
        p = np.zeros(len(te))
        for mk in models:
            m = make_model(mk, spw, {"rs": 0})
            m.fit(X[tr], y[tr])
            p += m.predict_proba(X[te])[:, 1]
        p /= len(models)
        fold_aucs.append(roc_auc_score(y[te], p))
    # single 5-fold pass for OOF predictions (for CI)
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        spw = float((y[tr] == 0).sum()) / max(int((y[tr] == 1).sum()), 1)
        p = np.zeros(len(te))
        for mk in models:
            m = make_model(mk, spw, {"rs": 0})
            m.fit(X[tr], y[tr])
            p += m.predict_proba(X[te])[:, 1]
        oof[te] = p / len(models)
    return float(np.mean(fold_aucs)), float(np.std(fold_aucs)), oof


def main():
    df = pd.read_csv(BASE / "data/external/dilist_official_1279.csv")
    data = standardize(df, "smiles", "label")
    y = data["label"].values.astype(int)
    print(f"[dilist] official DILIst: {len(data)} unique mols  ({y.mean():.0%} DILI+)", flush=True)

    feature_sets = {
        "desc_morgan": ["desc", "morgan"],
        "rich": ["desc", "morgan", "avalon", "erg", "maccs"],
    }
    feats = {fs: featurize(data["smiles"], kinds) for fs, kinds in feature_sets.items()}
    models = ["xgb", "lgbm", "cat"]

    print("\n[dilist] === repeated 5-fold CV (3 repeats) — feature set x model ===", flush=True)
    rows = []
    best = (None, -1, None, None)
    for fs, X in feats.items():
        for mk in models:
            mean, std, _ = cv_oof(X, y, [mk])
            rows.append({"feature_set": fs, "model": mk, "AUROC_mean": mean, "AUROC_std": std})
            print(f"  {fs:12s} {mk:5s}  AUROC={mean:.4f} ± {std:.4f}", flush=True)
        mean, std, oof = cv_oof(X, y, models)  # ensemble
        rows.append({"feature_set": fs, "model": "ENS3", "AUROC_mean": mean, "AUROC_std": std})
        print(f"  {fs:12s} {'ENS3':5s}  AUROC={mean:.4f} ± {std:.4f}", flush=True)
        if mean > best[1]:
            best = (fs, mean, std, oof)
    pd.DataFrame(rows).to_csv(RES / "dilist_cv.csv", index=False)

    fs_b, mean_b, std_b, oof_b = best
    auc, lo, hi = auroc_ci(y, oof_b, n_bootstrap=2000)
    print(f"\n[dilist] BEST: {fs_b} / ENS3  CV AUROC = {mean_b:.4f} ± {std_b:.4f}", flush=True)
    print(f"[dilist] out-of-fold AUROC = {auc:.4f}  95% CI [{lo:.4f}, {hi:.4f}]", flush=True)

    # ---- learning curve on the best feature set (ensemble) ----
    print("\n[dilist] === learning curve (does more data help?) ===", flush=True)
    Xb = feats[fs_b]
    fracs = [0.2, 0.4, 0.6, 0.8, 1.0]
    lc_rows = []
    rng = np.random.default_rng(0)
    for fr in fracs:
        aucs = []
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        for tr, te in skf.split(Xb, y):
            k = max(int(len(tr) * fr), 20)
            sub = rng.choice(tr, k, replace=False)
            spw = float((y[sub] == 0).sum()) / max(int((y[sub] == 1).sum()), 1)
            p = np.zeros(len(te))
            for mk in models:
                m = make_model(mk, spw, {"rs": 0}); m.fit(Xb[sub], y[sub])
                p += m.predict_proba(Xb[te])[:, 1]
            aucs.append(roc_auc_score(y[te], p / len(models)))
        lc_rows.append({"train_frac": fr, "n_train": int(len(y) * 0.8 * fr),
                        "AUROC_mean": float(np.mean(aucs)), "AUROC_std": float(np.std(aucs))})
        print(f"  frac={fr:.1f}  n~{int(len(y)*0.8*fr):4d}  AUROC={np.mean(aucs):.4f}", flush=True)
    lc = pd.DataFrame(lc_rows); lc.to_csv(RES / "dilist_learning_curve.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.errorbar(lc["n_train"], lc["AUROC_mean"], yerr=lc["AUROC_std"],
                marker="o", capsize=4, color="#3A6EA5")
    ax.set_xlabel("Training molecules"); ax.set_ylabel("CV AUROC")
    ax.set_title("DILIst learning curve — does more data help on the realistic task?")
    ax.grid(alpha=0.3); plt.tight_layout()
    fig.savefig(RES / "dilist_learning_curve.png", dpi=130); plt.close(fig)
    print("[dilist] saved -> results/dilist_cv.csv, dilist_learning_curve.csv/.png", flush=True)


if __name__ == "__main__":
    main()
