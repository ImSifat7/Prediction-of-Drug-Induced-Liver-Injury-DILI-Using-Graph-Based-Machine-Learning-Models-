"""Tox21 multi-task auxiliary features for DILI (Option D).

Approach (the practical version of "Tox21 multi-task pretraining"):
  1. For each of Tox21's 12 toxicity endpoints, train a Random Forest classifier
     on Morgan-FP-2048 features.
  2. Apply each trained classifier to every DILIrank molecule -> 12 toxicity
     probability scores per molecule.
  3. Save these 12 columns as additional features that the stack can consume.

This is the lightweight version of multi-task pretraining: instead of fine-tuning
a 47M-parameter transformer on Tox21 (which is risky and CPU-bound), we train
12 fast classifiers and use their PREDICTIONS as auxiliary signal. The intuition:
if a molecule is predicted to be "mitochondrial-membrane-potential-disrupting"
(SR-MMP) or "p53-activating" (SR-p53), that's directly relevant to DILI.

Run:
    python -m src.improved.tox21_aux
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.improved.data_utils import load_dataset, scaffold_split
else:
    from .data_utils import load_dataset, scaffold_split

RDLogger.DisableLog("rdApp.*")


TOX21_ENDPOINTS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]
N_BITS = 2048
RADIUS = 2


def morgan(smiles: str, n_bits: int = N_BITS, radius: int = RADIUS) -> np.ndarray:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def main():
    base = Path(__file__).resolve().parents[2]
    print("[tox21_aux] loading Tox21 ...", flush=True)
    tox = pd.read_csv(base / "data" / "tox21.csv")
    print(f"  Tox21 raw: {tox.shape}, columns: {tox.columns.tolist()[:5]}...", flush=True)

    print("[tox21_aux] computing Morgan FPs for Tox21 molecules ...", flush=True)
    t0 = time.time()
    tox_fps = np.stack([morgan(s) for s in tox["smiles"].tolist()], axis=0)
    print(f"  Tox21 FP matrix: {tox_fps.shape} in {time.time()-t0:.0f}s", flush=True)

    print("[tox21_aux] loading DILIrank ...", flush=True)
    graphs = load_dataset(base / "data" / "dili_clean.csv")
    dili_smiles = [g.smiles for g in graphs]
    train_idx, val_idx, test_idx = scaffold_split(graphs, 0.6, 0.2, 0.2, seed=42)
    print(f"  DILI split  tr={len(train_idx)} va={len(val_idx)} te={len(test_idx)}", flush=True)
    print("[tox21_aux] computing Morgan FPs for DILI molecules ...", flush=True)
    dili_fps = np.stack([morgan(s) for s in dili_smiles], axis=0)
    print(f"  DILI FP matrix: {dili_fps.shape}", flush=True)

    # Train one RF per endpoint, predict on all DILI molecules.
    all_preds = np.zeros((len(dili_smiles), len(TOX21_ENDPOINTS)), dtype=np.float32)
    endpoint_aurocs = []
    for j, ep in enumerate(TOX21_ENDPOINTS):
        y = tox[ep].values
        mask = ~np.isnan(y)
        y_lab = y[mask].astype(int)
        if len(np.unique(y_lab)) < 2:
            print(f"  [{ep}] skipped (only one class after dropping NaN)", flush=True)
            continue
        # 5-fold OOF for sanity, full-train for prediction onto DILI.
        from sklearn.model_selection import cross_val_score
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=2,
            n_jobs=4, random_state=42, class_weight="balanced",
        )
        try:
            cvauc = cross_val_score(clf, tox_fps[mask], y_lab, cv=3, scoring="roc_auc", n_jobs=1).mean()
        except Exception:
            cvauc = float("nan")
        clf.fit(tox_fps[mask], y_lab)
        probs = clf.predict_proba(dili_fps)[:, 1]
        all_preds[:, j] = probs
        endpoint_aurocs.append({"endpoint": ep, "n_labeled": int(mask.sum()),
                                "n_pos": int(y_lab.sum()), "cv_auroc": float(cvauc)})
        print(f"  [{ep:14s}] n={mask.sum():4d}  pos={int(y_lab.sum()):4d}  cv_auc={cvauc:.4f}", flush=True)

    pd.DataFrame(endpoint_aurocs).to_csv(base / "results" / "tox21_endpoint_aurocs.csv", index=False)

    # Build val/test arrays in the same indexing convention.
    val_arr = all_preds[val_idx]
    test_arr = all_preds[test_idx]
    train_arr = all_preds[train_idx]

    # Sanity: how predictive is the Tox21 mean of all 12 endpoints, *as a baseline*,
    # for DILI on the held-out test?
    y_test = np.array([int(graphs[i].y.item()) for i in test_idx])
    y_val = np.array([int(graphs[i].y.item()) for i in val_idx])
    mean_tox = test_arr.mean(axis=1)
    mean_val_tox = val_arr.mean(axis=1)
    print(f"\n[tox21_aux] Tox21-mean baseline on DILI test:  AUROC={roc_auc_score(y_test, mean_tox):.4f}", flush=True)
    print(f"[tox21_aux] Tox21-mean baseline on DILI val:    AUROC={roc_auc_score(y_val, mean_val_tox):.4f}", flush=True)

    # Save as per-endpoint columns for the stack.
    test_npz = {f"tox21_{ep}": test_arr[:, j] for j, ep in enumerate(TOX21_ENDPOINTS)}
    test_npz["y_true"] = y_test.astype(np.int64)
    test_npz["tox21_mean"] = mean_tox
    np.savez(base / "results" / "tox21_test_probs.npz", **test_npz)

    val_npz = {f"tox21_{ep}": val_arr[:, j] for j, ep in enumerate(TOX21_ENDPOINTS)}
    val_npz["y_true"] = y_val.astype(np.int64)
    val_npz["tox21_mean"] = mean_val_tox
    np.savez(base / "results" / "tox21_val_probs.npz", **val_npz)

    # Also save all 869 predictions for downstream use (e.g., concat with MolFormer emb).
    np.save(base / "results" / "tox21_all_preds.npy", all_preds)
    print("[tox21_aux] wrote results/tox21_{test,val}_probs.npz + tox21_endpoint_aurocs.csv", flush=True)


if __name__ == "__main__":
    main()
