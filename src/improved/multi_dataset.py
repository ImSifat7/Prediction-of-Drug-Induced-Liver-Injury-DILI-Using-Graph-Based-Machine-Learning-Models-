"""Multi-dataset DILI study: cross-dataset generalization + merged-training ablation.

Datasets (all standardized to canonical SMILES + binary DILI label, deduplicated
by InChIKey connectivity layer):
  - TDC-DILI   (data/tdc_dili.csv)                       475  — the benchmark
  - DILIrank   (data/dili_with_smiles.csv)               ~982 — FDA severity ranking
  - DILIst     (data/external/dilist_goldstandard_1111)  1111 — FDA expanded set

These share the FDA DILI lineage (TDC ⊂ DILIrank ⊂ DILIst-ish), so we FIRST report
the InChIKey overlap, then guard every train/test comparison against leakage by
removing test-set molecules (by InChIKey) from any training pool.

Two experiments, using the selected model (RDKit desc + Morgan -> XGBoost):
  (1) Cross-dataset AUROC matrix: train on A, test on B (A minus B-overlap).
      Diagonal = 5-fold stratified CV within a dataset.
  (2) Merged-training ablation on the official TDC test (96 mols): does adding
      DILIrank + DILIst to the training pool help vs TDC-train-only? Leakage-clean.

Run:  python -m src.improved.multi_dataset
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from rdkit import Chem, RDLogger

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.improved.tdc_official_v2 import precompute_blocks, make_model, SEEDS, RES, DATA_PATH
    from src.improved.stats_utils import delong_test
else:
    from .tdc_official_v2 import precompute_blocks, make_model, SEEDS, RES, DATA_PATH
    from .stats_utils import delong_test

BASE = Path(__file__).resolve().parents[2]


def standardize(df, smi_col, lab_col, name):
    rows = []
    for s, l in zip(df[smi_col], df[lab_col]):
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            continue
        try:
            ik = Chem.MolToInchiKey(m)[:14]
        except Exception:
            ik = None
        if not ik:
            continue
        try:
            lab = int(float(l))
        except Exception:
            continue
        rows.append((Chem.MolToSmiles(m), lab, ik))
    out = (pd.DataFrame(rows, columns=["smiles", "label", "ikey"])
           .dropna().drop_duplicates("ikey").reset_index(drop=True))
    print(f"  {name:10s}: {len(out):5d} unique mols  "
          f"({out.label.mean():.0%} DILI+)", flush=True)
    return out


def build_features(datasets):
    """Precompute desc+morgan for every unique SMILES across all datasets."""
    uniq = sorted({s for d in datasets.values() for s in d["smiles"]})
    blocks = precompute_blocks(uniq, ["desc", "morgan"])
    X = np.concatenate([blocks["desc"], blocks["morgan"]], axis=1).astype(np.float64)
    # sanitize: some large/exotic molecules yield inf / huge finite descriptors
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e6, 1e6).astype(np.float32)
    idx = {s: i for i, s in enumerate(uniq)}
    return X, idx


def Xy(df, X, idx):
    rows = [idx[s] for s in df["smiles"]]
    return X[rows], df["label"].values.astype(int)


def train_eval(Xtr, ytr, Xte, yte, seed=0):
    spw = float((ytr == 0).sum()) / max(int((ytr == 1).sum()), 1)
    m = make_model("xgb", spw, {"rs": seed})
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, p), p


def cv_auroc(Xd, yd):
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    aucs = []
    for tr, te in skf.split(Xd, yd):
        a, _ = train_eval(Xd[tr], yd[tr], Xd[te], yd[te])
        aucs.append(a)
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    print("=== Standardizing datasets (canonical SMILES + InChIKey dedup) ===", flush=True)
    tdc = standardize(pd.read_csv(BASE / "data/tdc_dili.csv"), "Drug", "Y", "TDC-DILI")
    dilirank = standardize(pd.read_csv(BASE / "data/dili_with_smiles.csv"), "smiles", "label", "DILIrank")
    dilist = standardize(pd.read_csv(BASE / "data/external/dilist_goldstandard_1111.csv"),
                         "smiles_r", "TOXICITY", "DILIst")
    datasets = {"TDC-DILI": tdc, "DILIrank": dilirank, "DILIst": dilist}

    # ---- overlap matrix ----
    print("\n=== InChIKey overlap between datasets (# shared molecules) ===", flush=True)
    names = list(datasets)
    ikeys = {n: set(datasets[n]["ikey"]) for n in names}
    print(f"  {'':10s}" + "".join(f"{n:>10s}" for n in names), flush=True)
    for a in names:
        line = f"  {a:10s}"
        for b in names:
            line += f"{len(ikeys[a] & ikeys[b]):>10d}"
        print(line, flush=True)

    X, idx = build_features(datasets)

    # ---- (1) cross-dataset AUROC matrix (train row -> test col) ----
    print("\n=== (1) Cross-dataset AUROC:  train (row) -> test (col) ===", flush=True)
    print("      (off-diagonal: test mols removed from train by InChIKey; diagonal: 5-fold CV)", flush=True)
    mat = pd.DataFrame(index=names, columns=names, dtype=object)
    for a in names:
        for b in names:
            if a == b:
                Xd, yd = Xy(datasets[a], X, idx)
                mean, std = cv_auroc(Xd, yd)
                mat.loc[a, b] = f"{mean:.3f}±{std:.3f}"
            else:
                te = datasets[b]
                tr = datasets[a][~datasets[a]["ikey"].isin(set(te["ikey"]))]
                Xtr, ytr = Xy(tr, X, idx)
                Xte, yte = Xy(te, X, idx)
                auc, _ = train_eval(Xtr, ytr, Xte, yte)
                mat.loc[a, b] = f"{auc:.3f}"
    print(mat.to_string(), flush=True)
    mat.to_csv(RES / "multi_dataset_crossval.csv")

    # ---- (2) merged-training ablation on official TDC test ----
    print("\n=== (2) Does adding DILIrank+DILIst to training help the TDC benchmark? ===", flush=True)
    from tdc.benchmark_group import admet_group
    group = admet_group(path=str(DATA_PATH))
    b = group.get("DILI")
    test_df = standardize(b["test"], "Drug", "Y", "TDC-test")
    trv_df = standardize(b["train_val"], "Drug", "Y", "TDC-trainval")
    test_ikeys = set(test_df["ikey"])

    # extra pool = DILIrank ∪ DILIst, minus anything in the TDC TEST set (leakage guard)
    extra = pd.concat([dilirank, dilist]).drop_duplicates("ikey")
    extra = extra[~extra["ikey"].isin(test_ikeys)]
    # also don't double-count molecules already in TDC train_val
    extra_only = extra[~extra["ikey"].isin(set(trv_df["ikey"]))]
    print(f"  TDC train_val = {len(trv_df)} | TDC test = {len(test_df)} | "
          f"extra DILI mols added (leakage-filtered) = {len(extra_only)}", flush=True)

    Xte, yte = Xy(test_df, X, idx)
    base_aucs, aug_aucs = [], []
    base_p = np.zeros(len(yte)); aug_p = np.zeros(len(yte))
    merged = pd.concat([trv_df, extra_only]).drop_duplicates("ikey")
    for seed in SEEDS:
        Xtr0, ytr0 = Xy(trv_df, X, idx)
        a0, p0 = train_eval(Xtr0, ytr0, Xte, yte, seed)
        Xtr1, ytr1 = Xy(merged, X, idx)
        a1, p1 = train_eval(Xtr1, ytr1, Xte, yte, seed)
        base_aucs.append(a0); aug_aucs.append(a1)
        base_p += p0; aug_p += p1
    base_p /= len(SEEDS); aug_p /= len(SEEDS)
    dl = delong_test(yte, aug_p, base_p)
    print(f"  TDC-train-only     : AUROC {np.mean(base_aucs):.4f} ± {np.std(base_aucs):.4f}", flush=True)
    print(f"  + DILIrank + DILIst: AUROC {np.mean(aug_aucs):.4f} ± {np.std(aug_aucs):.4f}", flush=True)
    print(f"  delta = {np.mean(aug_aucs)-np.mean(base_aucs):+.4f}   "
          f"DeLong p = {dl['p_value']:.4g}", flush=True)
    pd.DataFrame([{
        "train": "TDC only", "auroc_mean": np.mean(base_aucs), "auroc_std": np.std(base_aucs)},
        {"train": "TDC + DILIrank + DILIst", "auroc_mean": np.mean(aug_aucs),
         "auroc_std": np.std(aug_aucs), "delong_p": dl["p_value"],
         "n_extra": len(extra_only)}]).to_csv(RES / "multi_dataset_merged.csv", index=False)
    print("\n[multi] saved -> results/multi_dataset_crossval.csv, multi_dataset_merged.csv", flush=True)


if __name__ == "__main__":
    main()
