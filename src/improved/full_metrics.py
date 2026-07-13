"""Compute the full metric suite (AUROC, ACC, F1, MCC, precision, recall) for the
two configurations that were previously reported by AUROC only, so every results
table in the thesis can show all metrics from real computed values:

  1. Descriptor+Morgan bagged-XGBoost on TDC-DILI, 5-fold scaffold CV
     (comparable protocol to the graph-network table in tdc_cv5_metrics.csv).
  2. Descriptor+Morgan XGB+LGBM+CatBoost ensemble on the full official FDA DILIst
     set (1,165 unique drugs), repeated 5-fold stratified CV.

Writes results/full_metrics_computed.csv.  Run: python -m src.improved.full_metrics
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score)
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.improved.tdc_official import build_features
RDLogger.DisableLog("rdApp.*")
BASE = Path(__file__).resolve().parents[2]
RES = BASE / "results"


def clean(X):
    return np.clip(np.nan_to_num(X, nan=0, posinf=0, neginf=0), -1e6, 1e6).astype(np.float32)


def full(y, p, thr=0.5):
    yh = (p >= thr).astype(int)
    return dict(AUROC=round(roc_auc_score(y, p), 3), ACC=round(accuracy_score(y, yh), 3),
                F1=round(f1_score(y, yh, zero_division=0), 3), MCC=round(matthews_corrcoef(y, yh), 3),
                Prec=round(precision_score(y, yh, zero_division=0), 3),
                Rec=round(recall_score(y, yh, zero_division=0), 3))


def xgb(spw, seed=0):
    return XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.85,
                         colsample_bytree=0.7, reg_lambda=1.0, scale_pos_weight=spw,
                         eval_metric="logloss", random_state=seed, n_jobs=4, verbosity=0)


def main():
    rows = []

    # ---- 1. TDC desc+Morgan GBM, 5-fold scaffold CV ----
    tdc = pd.read_csv(BASE / "data" / "tdc_dili.csv")
    smi = tdc["Drug"].astype(str).tolist()
    y = pd.to_numeric(tdc["Y"]).astype(int).values
    scaf = [MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else s for s in smi]
    X = clean(build_features(smi, {}, use_molformer=False))
    oof = np.zeros(len(y)); fold_auc = []
    for tr, te in GroupKFold(5).split(X, y, groups=scaf):
        spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        pr = np.zeros(len(te))
        for sd in range(5):
            c = xgb(spw, sd); c.fit(X[tr], y[tr]); pr += c.predict_proba(X[te])[:, 1]
        oof[te] = pr / 5
        fold_auc.append(roc_auc_score(y[te], oof[te]))
    m = full(y, oof)
    m.update(model="TDC desc+Morgan GBM (5-fold scaffold CV)",
             AUROC_std=round(float(np.std(fold_auc)), 3))
    rows.append(m)
    print(m)

    # ---- 2. DILIst desc+Morgan XGB+LGBM+CatBoost ensemble, repeated 5-fold CV ----
    dl = pd.read_csv(BASE / "data" / "external" / "dilist_official_1279.csv").dropna(subset=["smiles"])
    dl["smiles"] = dl["smiles"].astype(str); dl = dl[dl["smiles"].str.strip().astype(bool)]
    def ik(s):
        mm = Chem.MolFromSmiles(s)
        try:
            return inchi.MolToInchiKey(mm).split("-")[0] if mm else None
        except Exception:
            return None
    dl["k"] = dl["smiles"].map(ik); dl = dl.dropna(subset=["k"]).drop_duplicates("k")
    yd = pd.to_numeric(dl["label"]).astype(int).values
    Xd = clean(build_features(dl["smiles"].tolist(), {}, use_molformer=False))
    oofd = np.zeros(len(yd)); cnt = np.zeros(len(yd))
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0).split(Xd, yd):
        spw = (yd[tr] == 0).sum() / max((yd[tr] == 1).sum(), 1)
        xg = xgb(spw); xg.fit(Xd[tr], yd[tr]); p = xg.predict_proba(Xd[te])[:, 1]
        lg = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, subsample=0.85, colsample_bytree=0.7,
                                scale_pos_weight=spw, random_state=0, n_jobs=4, verbosity=-1)
        lg.fit(Xd[tr], yd[tr]); p += lg.predict_proba(Xd[te])[:, 1]
        cb = CatBoostClassifier(iterations=400, depth=4, learning_rate=0.05, scale_pos_weight=spw,
                                verbose=0, random_seed=0)
        cb.fit(Xd[tr], yd[tr]); p += cb.predict_proba(Xd[te])[:, 1]
        oofd[te] += p / 3; cnt[te] += 1
    oofd /= cnt
    m2 = full(yd, oofd)
    m2.update(model="DILIst desc+Morgan ENS3 (repeated 5-fold CV)", n=int(len(yd)), pos_rate=round(float(yd.mean()), 2))
    rows.append(m2)
    print(m2)

    RES.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RES / "full_metrics_computed.csv", index=False)
    print(f"saved -> {RES / 'full_metrics_computed.csv'}")


if __name__ == "__main__":
    main()
