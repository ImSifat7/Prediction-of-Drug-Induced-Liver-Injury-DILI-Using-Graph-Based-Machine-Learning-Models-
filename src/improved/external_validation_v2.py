"""External validation v2: adds a TDC-test CONTROL + DILIrank label-confidence
stratification, so the generalisation gap is not confounded by (a) a pipeline
bug or (b) TDC's narrower label definition.

TDC-DILI (Xu et al. 2015) binarised ONLY DILIrank's vMost-DILI-concern (=1) and
vNo-DILI-concern (=0) drugs, excluding the noisy vLess / vAmbiguous middle. The
broad n~982 set keeps that middle, so a raw external AUROC mixes true
generalisation loss with label noise. We therefore also report the external
subset restricted to TDC-comparable confidence (vMost vs vNo only).

Run:  python -m src.improved.external_validation_v2
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.improved.tdc_official import build_features
    from src.improved.stats_utils import auroc_ci
    from src.improved.external_validation import (
        inchikey_skeleton, murcko_scaffold, bagged_xgb_fit_predict, best_mcc_threshold,
    )
else:
    from .tdc_official import build_features
    from .stats_utils import auroc_ci
    from .external_validation import (
        inchikey_skeleton, murcko_scaffold, bagged_xgb_fit_predict, best_mcc_threshold,
    )

BASE = Path(__file__).resolve().parents[2]
TDC_DIR = BASE / "data" / "tdc_benchmark" / "admet_group" / "dili"
EXTERNAL_CSV = BASE / "data" / "dili_with_smiles.csv"
DILIRANK_XLSX = BASE / "data" / "DILIrank2.xlsx"
RES = BASE / "results"


def _norm(s):
    return str(s).strip().lower()


def report(tag, y, p, thr):
    y = np.asarray(y); p = np.asarray(p)
    if len(np.unique(y)) < 2:
        print(f"  [{tag:22s}] n={len(y)} single-class -> skipped")
        return None
    yhat = (p >= thr).astype(int)
    auc, lo, hi = auroc_ci(y, p, n_bootstrap=2000, seed=42)
    row = {"set": tag, "n": len(y), "pos": int(y.sum()), "pos_rate": round(float(y.mean()), 3),
           "AUROC": round(auc, 4), "CI_low": round(lo, 4), "CI_high": round(hi, 4),
           "ACC": round(accuracy_score(y, yhat), 4),
           "F1": round(f1_score(y, yhat, zero_division=0), 4),
           "MCC": round(matthews_corrcoef(y, yhat), 4), "threshold": round(thr, 3)}
    print(f"  [{tag:22s}] n={row['n']:3d} pos={row['pos_rate']:.2f}  "
          f"AUROC={auc:.4f} [{lo:.3f},{hi:.3f}]  ACC={row['ACC']:.3f} "
          f"F1={row['F1']:.3f} MCC={row['MCC']:.3f}", flush=True)
    return row


def main():
    t0 = time.time(); RES.mkdir(exist_ok=True)

    def col(df, *cands):
        for c in cands:
            if c in df.columns:
                return c
        raise KeyError(cands)

    # ---- TDC universe -------------------------------------------------------
    tr_val = pd.read_csv(TDC_DIR / "train_val.csv")
    tdc_test = pd.read_csv(TDC_DIR / "test.csv")
    tr_smi, tr_y = col(tr_val, "Drug", "smiles"), col(tr_val, "Y", "label")
    te_smi, te_y = col(tdc_test, "Drug", "smiles"), col(tdc_test, "Y", "label")
    tdc_keys = {k for k in (inchikey_skeleton(s) for s in
                            list(tr_val[tr_smi]) + list(tdc_test[te_smi])) if k}
    tdc_train_scaffolds = {sc for sc in (murcko_scaffold(s) for s in tr_val[tr_smi]) if sc}

    # ---- train frozen headline model ---------------------------------------
    y_tr = pd.to_numeric(tr_val[tr_y], errors="coerce").fillna(0).astype(int).values
    X_tr = build_features(list(tr_val[tr_smi]), {}, use_molformer=False)

    # ---- CONTROL: frozen model on the official TDC test set ----------------
    X_te = build_features(list(tdc_test[te_smi]), {}, use_molformer=False)
    y_te = pd.to_numeric(tdc_test[te_y], errors="coerce").fillna(0).astype(int).values
    p_te = bagged_xgb_fit_predict(X_tr, y_tr, X_te)
    p_tr_insample = bagged_xgb_fit_predict(X_tr, y_tr, X_tr)
    thr = best_mcc_threshold(y_tr, p_tr_insample)
    print(f"[ext2] threshold(max-MCC on TDC train)={thr:.3f}")
    print("\n=== CONTROL: frozen model on official TDC test (expect ~0.88-0.92) ===")
    rows = []
    rows.append(report("CONTROL_tdc_test", y_te, p_te, thr))

    # ---- DILIrank concern categories ---------------------------------------
    raw = pd.read_excel(DILIRANK_XLSX, header=1)  # row 0 is the dataset title
    name_c = col(raw, "CompoundName")
    concern_c = col(raw, "vDILI-Concern")
    raw["_name"] = raw[name_c].map(_norm)
    concern_map = dict(zip(raw["_name"], raw[concern_c].astype(str)))
    print("\n[ext2] DILIrank vDILI-Concern distribution:",
          dict(raw[concern_c].value_counts(dropna=False)))

    # ---- external set -------------------------------------------------------
    ext = pd.read_csv(EXTERNAL_CSV)
    e_name = col(ext, "drug_name", "name")
    e_smi = col(ext, "smiles", "Drug")
    e_y = col(ext, "label", "Y")
    ext = ext[[e_name, e_smi, e_y]].rename(columns={e_name: "name", e_smi: "smiles", e_y: "label"})
    ext["smiles"] = ext["smiles"].astype(str)
    ext = ext[ext["smiles"].str.strip().astype(bool)]
    ext["label"] = pd.to_numeric(ext["label"], errors="coerce")
    ext = ext.dropna(subset=["label"]); ext["label"] = ext["label"].astype(int)
    ext["key"] = ext["smiles"].map(inchikey_skeleton)
    ext = ext.dropna(subset=["key"]).drop_duplicates(subset=["key"], keep="first")
    n_pre = len(ext)
    ext = ext[~ext["key"].isin(tdc_keys)].copy()
    print(f"[ext2] external truly-external n={len(ext)} (removed {n_pre-len(ext)} TDC-overlap)")

    ext["concern"] = ext["name"].map(_norm).map(concern_map).fillna("unknown")
    ext["scaffold"] = ext["smiles"].map(murcko_scaffold)
    ext["novel_scaffold"] = ~ext["scaffold"].isin(tdc_train_scaffolds)
    print("[ext2] external concern breakdown:", dict(ext["concern"].value_counts()))

    X_ext = build_features(list(ext["smiles"]), {}, use_molformer=False)
    p_ext = bagged_xgb_fit_predict(X_tr, y_tr, X_ext)
    y_ext = ext["label"].values.astype(int)
    ext["prob"] = p_ext

    # high-confidence = TDC-comparable label definition (vMost vs vNo only)
    cl = ext["concern"].str.lower()
    hi_mask = cl.str.contains("most") | cl.str.contains("no-dili") | cl.str.contains("no dili")
    nv = ext["novel_scaffold"].values

    print("\n=== EXTERNAL VALIDATION (headline desc+Morgan -> bagged XGBoost) ===")
    rows.append(report("external_all", y_ext, p_ext, thr))
    rows.append(report("external_novel_scaffold", y_ext[nv], p_ext[nv], thr))
    rows.append(report("external_highconf", y_ext[hi_mask.values], p_ext[hi_mask.values], thr))
    rows.append(report("external_highconf_novelscaf",
                       y_ext[(hi_mask.values) & nv], p_ext[(hi_mask.values) & nv], thr))

    # ---- 7. LEARNABILITY control: 5-fold CV trained ON external high-conf ---
    # Separates "model overfits TDC" from "this data's ceiling is just lower".
    from sklearn.model_selection import StratifiedKFold
    Xh = np.clip(np.nan_to_num(X_ext[hi_mask.values], nan=0.0, posinf=0.0, neginf=0.0),
                 -1e6, 1e6).astype(np.float32)
    yh = y_ext[hi_mask.values]
    if len(np.unique(yh)) == 2 and len(yh) >= 50:
        oof = np.zeros(len(yh))
        for tr_idx, te_idx in StratifiedKFold(5, shuffle=True, random_state=0).split(Xh, yh):
            oof[te_idx] = bagged_xgb_fit_predict(Xh[tr_idx], yh[tr_idx], Xh[te_idx])
        rows.append(report("LEARNABILITY_cv_on_external", yh, oof, thr))

    rows = [r for r in rows if r]
    pd.DataFrame(rows).to_csv(RES / "external_validation_v2.csv", index=False)
    ext[["name", "smiles", "label", "concern", "novel_scaffold", "prob"]].to_csv(
        RES / "external_predictions_v2.csv", index=False)
    print(f"\n[ext2] saved -> {RES/'external_validation_v2.csv'}")
    print(f"[ext2] saved -> {RES/'external_predictions_v2.csv'}")
    print(f"[ext2] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
