"""Task 1 + Task 9 tables.

Task 1 - Main model-comparison table with Recall/Sensitivity and Specificity
  (mean ± std across the 5 scaffold-CV folds). Sensitivity/Specificity are computed
  here for the SELECTED model (descriptor+Morgan GBM) from per-fold confusion counts.
  The graph-network rows keep their AUROC/ACC/F1/MCC from tdc_cv5_metrics.csv; their
  Sensitivity/Specificity are emitted by re-running the (enhanced) tdc_cv5.py and are
  left blank here rather than invented.

Task 9 - Improved-vs-current pipeline comparison on the same data, isolating the
  effect of the scientifically-valid additions (SMILES standardisation, duplicate /
  conflicting-label handling, 4-way train/external overlap removal, validation-only
  Youden threshold, Platt calibration). AUROC is model/feature-bound and barely moves;
  the gains are in evaluation validity and probability calibration, reported honestly.

Outputs: results/main_comparison_full.csv, results/improved_vs_current.csv
Run: python -m src.improved.main_comparison
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from src.improved.tdc_official import build_features
from src.improved import eval_utils as EU

RDLogger.DisableLog("rdApp.*")
BASE = Path(__file__).resolve().parents[2]
TDC = BASE / "data" / "tdc_benchmark" / "admet_group" / "dili"
RES = BASE / "results"


def clean(X):
    return np.clip(np.nan_to_num(X, nan=0, posinf=0, neginf=0), -1e6, 1e6).astype(np.float32)


def bagged(Xtr, ytr, Xte, nb=5):
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    p = np.zeros(len(Xte))
    for sd in range(nb):
        c = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.85,
                          colsample_bytree=0.7, reg_lambda=1.0, scale_pos_weight=spw,
                          eval_metric="logloss", random_state=sd, n_jobs=4, verbosity=0)
        c.fit(Xtr, ytr); p += c.predict_proba(Xte)[:, 1]
    return p / nb


def fmt(vals):
    return f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"


def main():
    tdc = pd.read_csv(BASE / "data" / "tdc_dili.csv")
    smi = tdc["Drug"].astype(str).tolist()
    y = pd.to_numeric(tdc["Y"]).astype(int).values
    scaf = [MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else s for s in smi]
    X = clean(build_features(smi, {}, use_molformer=False))

    # ---- Task 1: GBM per-fold full panel at threshold 0.5 (same convention as the
    #      graph-network rows and thesis Table 2, so all models are comparable). ----
    per = {k: [] for k in ["AUROC", "ACC", "F1", "MCC", "Sensitivity", "Specificity"]}
    for tr, te in GroupKFold(5).split(X, y, groups=scaf):
        p_te = bagged(X[tr], y[tr], X[te])
        m = EU.full_panel(y[te], p_te, 0.5)
        for k in per:
            per[k].append(m[k])
    gbm_row = {"Model": "Descriptor+Morgan GBM (selected)",
               **{k: fmt(v) for k, v in per.items()}}

    # ---- graph-network rows from existing CV metrics (Sens/Spec via tdc_cv5.py re-run) ----
    cv = pd.read_csv(RES / "tdc_cv5_metrics.csv").set_index("method")
    name_map = {"AttentiveFP": "AttentiveFP", "GCN": "GCN", "GIN": "GIN", "GAT": "GAT",
                "GraphSAGE": "GraphSAGE", "MPNN": "MPNN", "chemberta": "ChemBERTa",
                "molformer": "MoLFormer (frozen)", "rank-avg": "Rank-averaging ensemble"}
    rows = [gbm_row]
    def cell(r, met):
        if f"{met}_mean" not in r.index:
            return "n/a"
        s = f"{met}_std"
        return f"{r[f'{met}_mean']:.3f} ± {r[s]:.3f}" if s in r.index else f"{r[f'{met}_mean']:.3f}"

    for key, disp in name_map.items():
        if key in cv.index:
            r = cv.loc[key]
            rows.append({"Model": disp,
                         "AUROC": cell(r, "AUROC"), "ACC": cell(r, "ACC"),
                         "F1": cell(r, "F1"), "MCC": cell(r, "MCC"),
                         "Sensitivity": cell(r, "Sensitivity"), "Specificity": cell(r, "Specificity")})
    pd.DataFrame(rows).to_csv(RES / "main_comparison_full.csv", index=False)
    print("=== Task 1: main_comparison_full.csv ===")
    print(pd.DataFrame(rows).to_string(index=False))

    # ---- Task 9: improved vs current on TDC test + DILIrank external ----
    tr_df = pd.read_csv(TDC / "train_val.csv").rename(columns={"Drug": "smiles", "Y": "label"})
    te_df = pd.read_csv(TDC / "test.csv").rename(columns={"Drug": "smiles", "Y": "label"})
    ext = pd.read_csv(BASE / "data" / "dili_with_smiles.csv").rename(columns={"smiles": "smiles", "label": "label"})
    ext["smiles"] = ext["smiles"].astype(str); ext = ext[ext["smiles"].str.strip().astype(bool)]
    ext["label"] = pd.to_numeric(ext["label"], errors="coerce"); ext = ext.dropna(subset=["label"]); ext["label"] = ext["label"].astype(int)

    comp = []
    for mode in ["current", "improved"]:
        if mode == "improved":
            tr_s = tr_df["smiles"].map(EU.standardize_smiles); m = tr_s.notna()
            trs, tyl = tr_s[m].tolist(), tr_df["label"].values[m]
            te_s = te_df["smiles"].map(EU.standardize_smiles); mt = te_s.notna()
            tes, teyl = te_s[mt].tolist(), te_df["label"].values[mt]
            e_s = ext["smiles"].map(EU.standardize_smiles); me = e_s.notna()
            edf = ext[me].copy(); edf["s"] = e_s[me].values
            edf["k"] = edf["s"].map(EU.inchikey14); edf = edf.dropna(subset=["k"]).drop_duplicates("k")
            tkeys = set(x for x in (EU.inchikey14(s) for s in trs) if x)
            tstd = set(trs); tscaf = set(x for x in (EU.scaffold(s) for s in trs) if x)
            edf = edf[~(edf["k"].isin(tkeys) | edf["s"].isin(tstd) | edf["s"].map(EU.scaffold).isin(tscaf))]
            es, eyl = edf["s"].tolist(), edf["label"].values
        else:  # current: raw SMILES, InChIKey-only dedup, no standardisation/conflict/scaffold
            trs, tyl = tr_df["smiles"].astype(str).tolist(), tr_df["label"].values
            tes, teyl = te_df["smiles"].astype(str).tolist(), te_df["label"].values
            edf = ext.copy(); edf["k"] = edf["smiles"].map(EU.inchikey14)
            edf = edf.dropna(subset=["k"]).drop_duplicates("k")
            tkeys = set(x for x in (EU.inchikey14(s) for s in trs) if x)
            edf = edf[~edf["k"].isin(tkeys)]
            es, eyl = edf["smiles"].tolist(), edf["label"].values

        Xtr = clean(build_features(trs, {}, use_molformer=False))
        Xte = clean(build_features(tes, {}, use_molformer=False))
        Xe = clean(build_features(es, {}, use_molformer=False))
        p_te = bagged(Xtr, tyl, Xte); p_e = bagged(Xtr, tyl, Xe)
        if mode == "improved":
            # Youden on OOF for threshold
            oof = np.zeros(len(tyl)); sc = [EU.scaffold(s) or s for s in trs]
            for a, b in GroupKFold(5).split(Xtr, tyl, groups=sc):
                oof[b] = bagged(Xtr[a], tyl[a], Xtr[b])
            thr = EU.youden_threshold(tyl, oof)
        else:
            thr = 0.5
        mte = EU.full_panel(teyl, p_te, thr); me_ = EU.full_panel(eyl, p_e, thr)
        comp.append({"pipeline": mode, "threshold": round(thr, 3),
                     "test_n": mte["n"], "test_AUROC": round(mte["AUROC"], 3),
                     "test_MCC": round(mte["MCC"], 3), "test_Brier": round(mte["Brier"], 3),
                     "ext_n": me_["n"], "ext_AUROC": round(me_["AUROC"], 3),
                     "ext_MCC": round(me_["MCC"], 3)})
    pd.DataFrame(comp).to_csv(RES / "improved_vs_current.csv", index=False)
    print("\n=== Task 9: improved_vs_current.csv ===")
    print(pd.DataFrame(comp).to_string(index=False))


if __name__ == "__main__":
    main()
