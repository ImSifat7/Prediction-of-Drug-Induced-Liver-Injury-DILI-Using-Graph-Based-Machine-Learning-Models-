"""Official TDC ADMET-Group DILI benchmark harness (5-seed, leaderboard-comparable).

WHY THIS EXISTS
  The existing tdc_pipeline.py reports a single home-made scaffold split
  (70/10/20, seed=42).  That number is NOT comparable to the public TDC DILI
  leaderboard, which uses the OFFICIAL admet_group protocol: a fixed 96-molecule
  test set evaluated over 5 seeds, reporting mean +/- std AUROC.  This module
  runs models under that official protocol so results can be placed on the
  leaderboard and published with proper variance.

MODEL (Phase 1.1: MapLight-style feature union, the reproducible SOTA recipe)
  features = [ RDKit 2D descriptors || Morgan ECFP-2048 || MolFormer-XL emb ]
  bagged XGBoost (early-stop on the official valid split), predict the fixed test.

Run:
    python -m src.improved.tdc_official              # full (desc+morgan+molformer)
    python -m src.improved.tdc_official --fast       # desc+morgan only (seconds)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.improved.molformer import extract_molformer_embeddings
else:
    from .molformer import extract_molformer_embeddings

SEEDS = [1, 2, 3, 4, 5]
N_BAG = 5
FP_BITS = 2048
BASE = Path(__file__).resolve().parents[2]
DATA_PATH = BASE / "data" / "tdc_benchmark"   # admet_group cache dir
RES = BASE / "results"

# ---- feature builders -------------------------------------------------------
_DESC_NAMES = [d[0] for d in Descriptors._descList]
_DESC_CALC = MoleculeDescriptors.MolecularDescriptorCalculator(_DESC_NAMES)
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)


def _descriptors(mol) -> np.ndarray:
    if mol is None:
        return np.zeros(len(_DESC_NAMES), dtype=np.float32)
    vals = np.array(_DESC_CALC.CalcDescriptors(mol), dtype=np.float64)
    return np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _morgan(mol) -> np.ndarray:
    if mol is None:
        return np.zeros(FP_BITS, dtype=np.float32)
    return _MORGAN.GetFingerprintAsNumPy(mol).astype(np.float32)


def build_features(smiles_list, mf_lookup, use_molformer: bool) -> np.ndarray:
    """[desc || morgan || (molformer)] for each SMILES, in order."""
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    desc = np.stack([_descriptors(m) for m in mols], axis=0)
    fp = np.stack([_morgan(m) for m in mols], axis=0)
    blocks = [desc, fp]
    if use_molformer:
        emb = np.stack([mf_lookup[s] for s in smiles_list], axis=0)
        blocks.append(emb)
    return np.concatenate(blocks, axis=1)


def bagged_xgb(X_tr, y_tr, X_va, y_va, X_te) -> np.ndarray:
    pos = int((y_tr == 1).sum()); neg = int((y_tr == 0).sum())
    spw = neg / max(pos, 1)
    pred = np.zeros(X_te.shape[0])
    for sd in range(N_BAG):
        clf = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.7, reg_lambda=1.0,
            scale_pos_weight=spw, eval_metric="logloss",
            random_state=sd, n_jobs=4, verbosity=0,
        )
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        pred += clf.predict_proba(X_te)[:, 1]
    return pred / N_BAG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="skip MolFormer embeddings (descriptors+Morgan only)")
    args = ap.parse_args()
    use_mf = not args.fast

    from tdc.benchmark_group import admet_group
    RES.mkdir(exist_ok=True)
    DATA_PATH.mkdir(parents=True, exist_ok=True)

    print(f"[official] loading admet_group DILI benchmark (use_molformer={use_mf}) ...", flush=True)
    group = admet_group(path=str(DATA_PATH))
    benchmark = group.get("DILI")
    name = benchmark["name"]
    test_df = benchmark["test"]
    train_val_df = benchmark["train_val"]
    print(f"[official] name={name!r}  train_val={len(train_val_df)}  test={len(test_df)}  "
          f"test_pos_rate={test_df['Y'].mean():.3f}", flush=True)

    # MolFormer embeddings: extract ONCE for the union of all SMILES (test is
    # fixed; only the train/valid partition changes across seeds).
    mf_lookup = {}
    if use_mf:
        import torch
        all_smiles = sorted(set(train_val_df["Drug"]).union(set(test_df["Drug"])))
        cache = RES / "tdc_official_molformer_emb.npz"
        if cache.exists():
            d = np.load(cache, allow_pickle=True)
            cached = {s: e for s, e in zip(d["smiles"], d["emb"])}
            if set(cached) >= set(all_smiles):
                mf_lookup = cached
                print(f"[official] reused {len(mf_lookup)} cached MolFormer embeddings", flush=True)
        if not mf_lookup:
            emb = extract_molformer_embeddings(all_smiles, torch.device("cpu"))
            mf_lookup = {s: e for s, e in zip(all_smiles, emb)}
            np.savez(cache, smiles=np.array(all_smiles, dtype=object),
                     emb=np.stack([mf_lookup[s] for s in all_smiles]))

    X_te = build_features(list(test_df["Drug"]), mf_lookup, use_mf)
    y_te = test_df["Y"].values.astype(int)

    predictions_list = []
    per_seed = []
    for seed in SEEDS:
        t0 = time.time()
        tr_df, va_df = group.get_train_valid_split(benchmark=name, split_type="default", seed=seed)
        X_tr = build_features(list(tr_df["Drug"]), mf_lookup, use_mf)
        X_va = build_features(list(va_df["Drug"]), mf_lookup, use_mf)
        y_tr = tr_df["Y"].values.astype(int)
        y_va = va_df["Y"].values.astype(int)

        p_te = bagged_xgb(X_tr, y_tr, X_va, y_va, X_te)
        auc = roc_auc_score(y_te, p_te)
        per_seed.append(auc)
        predictions_list.append({name: p_te})
        print(f"  seed={seed}  tr={len(tr_df)} va={len(va_df)}  "
              f"test_AUROC={auc:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    results = group.evaluate_many(predictions_list)
    mean, std = results[name]
    print("\n=== OFFICIAL TDC DILI (5-seed admet_group) ===")
    print(f"  feature_dim = {X_te.shape[1]}  ({'desc+morgan+molformer' if use_mf else 'desc+morgan'})")
    print(f"  per-seed AUROC = {[round(a,4) for a in per_seed]}")
    print(f"  evaluate_many AUROC = {mean:.4f} +/- {std:.4f}")
    print(f"  (leaderboard refs: AttentiveFP 0.886, Chemprop-RDKit 0.887, MapLight+GNN 0.917)")

    out = RES / ("tdc_official_metrics_fast.csv" if args.fast else "tdc_official_metrics.csv")
    pd.DataFrame([{
        "feature_set": "desc+morgan" if args.fast else "desc+morgan+molformer",
        "feature_dim": X_te.shape[1],
        "AUROC_mean": mean, "AUROC_std": std,
        **{f"seed{s}": a for s, a in zip(SEEDS, per_seed)},
    }]).to_csv(out, index=False)
    print(f"[official] saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
