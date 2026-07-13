"""External validation of the headline TDC-DILI model on held-out DILIrank chemistry.

WHY THIS EXISTS
  The headline AUROC (0.920 +/- 0.014) rests on a SINGLE fixed 96-molecule TDC
  test set. A reviewer's first question is: does it generalise to molecules from
  outside that split?  This module answers that.

DESIGN (leakage-controlled)
  * Model   : the EXACT headline recipe -- RDKit-2D descriptors + Morgan-ECFP-2048
              -> bagged XGBoost (src.improved.tdc_official.build_features /
              bagged-XGB params), trained ONCE on the full official TDC train_val
              set, then frozen.
  * External: the broader DILIrank-derived set (data/dili_with_smiles.csv, n~982),
              MINUS every molecule that appears anywhere in TDC-DILI (train_val U
              test). Overlap is removed by InChIKey *skeleton* (first 14 chars) so
              salt / protonation / stereo variants are also stripped -- a strict,
              conservative dedup that strengthens the "unseen" claim.
  * Reports two cuts, each with a 95% bootstrap CI:
      (A) ALL non-overlapping external molecules.
      (B) NOVEL-SCAFFOLD subset -- external molecules whose Bemis-Murcko scaffold
          is ALSO absent from TDC training. This is the true generalisation-to-
          new-chemistry number.

Run:
    python -m src.improved.external_validation
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef
from xgboost import XGBClassifier

from rdkit import Chem, RDLogger
from rdkit.Chem import inchi
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.improved.tdc_official import build_features
    from src.improved.stats_utils import auroc_ci
else:
    from .tdc_official import build_features
    from .stats_utils import auroc_ci

BASE = Path(__file__).resolve().parents[2]
TDC_DIR = BASE / "data" / "tdc_benchmark" / "admet_group" / "dili"
EXTERNAL_CSV = BASE / "data" / "dili_with_smiles.csv"
RES = BASE / "results"
N_BAG = 5


# ---- structure keys ---------------------------------------------------------
def inchikey_skeleton(smiles: str) -> str | None:
    """InChIKey first block (connectivity skeleton). None if unparseable/empty."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        key = inchi.MolToInchiKey(mol)
    except Exception:
        return None
    return key.split("-")[0] if key else None


def murcko_scaffold(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaf) if scaf is not None else None
    except Exception:
        return None


def bagged_xgb_fit_predict(X_tr, y_tr, X_ext):
    """Headline bagged-XGBoost: 5 bags, fit on train_val, mean prob on external."""
    pos = int((y_tr == 1).sum()); neg = int((y_tr == 0).sum())
    spw = neg / max(pos, 1)
    pred = np.zeros(X_ext.shape[0])
    for sd in range(N_BAG):
        clf = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.7, reg_lambda=1.0,
            scale_pos_weight=spw, eval_metric="logloss",
            random_state=sd, n_jobs=4, verbosity=0,
        )
        clf.fit(X_tr, y_tr, verbose=False)
        pred += clf.predict_proba(X_ext)[:, 1]
    return pred / N_BAG


def best_mcc_threshold(y, p) -> float:
    """Threshold maximising MCC, chosen on TRAINING preds only (no external peek)."""
    grid = np.linspace(0.05, 0.95, 181)
    best_t, best_m = 0.5, -2.0
    for t in grid:
        m = matthews_corrcoef(y, (p >= t).astype(int))
        if m > best_m:
            best_m, best_t = m, t
    return best_t


def report(tag, y, p, thr):
    yhat = (p >= thr).astype(int)
    auc, lo, hi = auroc_ci(y, p, n_bootstrap=2000, seed=42)
    row = {
        "set": tag, "n": len(y), "pos": int(y.sum()), "neg": int((y == 0).sum()),
        "pos_rate": round(float(y.mean()), 3),
        "AUROC": round(auc, 4), "AUROC_CI_low": round(lo, 4), "AUROC_CI_high": round(hi, 4),
        "ACC": round(accuracy_score(y, yhat), 4),
        "F1": round(f1_score(y, yhat, zero_division=0), 4),
        "MCC": round(matthews_corrcoef(y, yhat), 4),
        "threshold": round(thr, 3),
    }
    print(f"  [{tag:16s}] n={row['n']:3d} pos_rate={row['pos_rate']:.2f} "
          f"AUROC={auc:.4f} [{lo:.3f},{hi:.3f}]  ACC={row['ACC']:.3f} "
          f"F1={row['F1']:.3f} MCC={row['MCC']:.3f}", flush=True)
    return row


def main():
    t0 = time.time()
    RES.mkdir(exist_ok=True)

    # ---- 1. TDC (the model's universe) -------------------------------------
    tr_val = pd.read_csv(TDC_DIR / "train_val.csv")
    tdc_test = pd.read_csv(TDC_DIR / "test.csv")
    # normalise column names (TDC uses Drug / Y)
    def col(df, *cands):
        for c in cands:
            if c in df.columns:
                return c
        raise KeyError(cands)
    tr_smi, tr_y = col(tr_val, "Drug", "smiles"), col(tr_val, "Y", "label")
    te_smi = col(tdc_test, "Drug", "smiles")

    tdc_all_smiles = list(tr_val[tr_smi]) + list(tdc_test[te_smi])
    tdc_keys = {k for k in (inchikey_skeleton(s) for s in tdc_all_smiles) if k}
    tdc_train_scaffolds = {sc for sc in (murcko_scaffold(s) for s in tr_val[tr_smi]) if sc}
    print(f"[ext] TDC train_val={len(tr_val)} test={len(tdc_test)}  "
          f"unique InChIKey-skeletons={len(tdc_keys)}  "
          f"train scaffolds={len(tdc_train_scaffolds)}", flush=True)

    # ---- 2. external candidates --------------------------------------------
    ext = pd.read_csv(EXTERNAL_CSV)
    e_smi = col(ext, "smiles", "Drug")
    e_y = col(ext, "label", "Y")
    ext = ext[[e_smi, e_y]].rename(columns={e_smi: "smiles", e_y: "label"})
    n0 = len(ext)
    ext["smiles"] = ext["smiles"].astype(str)
    ext = ext[ext["smiles"].str.strip().astype(bool)]                       # drop empty
    ext["label"] = pd.to_numeric(ext["label"], errors="coerce")
    ext = ext.dropna(subset=["label"])
    ext["label"] = ext["label"].astype(int)
    ext["key"] = ext["smiles"].map(inchikey_skeleton)
    ext = ext.dropna(subset=["key"])                                        # drop unparseable
    n_parsed = len(ext)
    ext = ext.drop_duplicates(subset=["key"], keep="first")                 # internal dedup
    n_dedup = len(ext)

    # ---- 3. remove TDC overlap (the key step) ------------------------------
    in_tdc = ext["key"].isin(tdc_keys)
    n_overlap = int(in_tdc.sum())
    ext = ext[~in_tdc].copy()
    print(f"[ext] external: raw={n0} parsed={n_parsed} internal-dedup={n_dedup} "
          f"-> removed {n_overlap} TDC-overlapping -> {len(ext)} truly-external", flush=True)

    # ---- 4. novel-scaffold flag --------------------------------------------
    ext["scaffold"] = ext["smiles"].map(murcko_scaffold)
    ext["novel_scaffold"] = ~ext["scaffold"].isin(tdc_train_scaffolds)
    n_novel = int(ext["novel_scaffold"].sum())
    print(f"[ext] of those, {n_novel} have a scaffold UNSEEN in TDC training "
          f"(novel-scaffold cut)", flush=True)

    # ---- 5. train headline model on full TDC train_val, freeze -------------
    y_tr = pd.to_numeric(tr_val[tr_y], errors="coerce").fillna(0).astype(int).values
    X_tr = build_features(list(tr_val[tr_smi]), {}, use_molformer=False)
    p_tr = None  # for threshold selection we refit-free: use in-sample preds below

    X_ext = build_features(list(ext["smiles"]), {}, use_molformer=False)
    p_ext = bagged_xgb_fit_predict(X_tr, y_tr, X_ext)

    # threshold chosen on TRAIN preds (in-sample) -> no external peeking
    p_tr_insample = bagged_xgb_fit_predict(X_tr, y_tr, X_tr)
    thr = best_mcc_threshold(y_tr, p_tr_insample)
    print(f"[ext] decision threshold (max-MCC on TDC train) = {thr:.3f}", flush=True)

    y_ext = ext["label"].values.astype(int)

    # ---- 6. report ----------------------------------------------------------
    print("\n=== EXTERNAL VALIDATION (headline desc+Morgan -> bagged XGBoost) ===")
    rows = [report("external_all", y_ext, p_ext, thr)]
    nv = ext["novel_scaffold"].values
    if n_novel >= 10 and len(np.unique(y_ext[nv])) == 2:
        rows.append(report("novel_scaffold", y_ext[nv], p_ext[nv], thr))
    else:
        print(f"  [novel_scaffold ] skipped (n={n_novel}, needs >=10 with both classes)")

    out = RES / "external_validation.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    # also persist per-molecule predictions for error analysis later
    ext_out = ext[["smiles", "label", "novel_scaffold"]].copy()
    ext_out["prob"] = p_ext
    ext_out["pred"] = (p_ext >= thr).astype(int)
    ext_out.to_csv(RES / "external_predictions.csv", index=False)
    print(f"\n[ext] saved -> {out}")
    print(f"[ext] saved -> {RES / 'external_predictions.csv'}")
    print(f"[ext] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
