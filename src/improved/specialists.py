"""Per-cluster specialist models (Option J from advisor discussion).

Idea (with the caveat that DILIrank's 521 training molecules is genuinely tiny
for this approach):

  1. Cluster all 869 DILI molecules in MolFormer-XL embedding space (K-means, K=4).
     The clusters are chemical-similarity groups: cluster 0 might be "small
     organics", cluster 3 might be "macrocyclic / peptidic", etc.

  2. For each cluster, train one XGBoost specialist on its training subset only,
     using a rich per-molecule feature vector:
       [MolFormer-emb (768) || Morgan-FP-512 || 14-d descriptors || 12 Tox21 endpoints]

  3. At inference, every test molecule has:
       - a hard cluster assignment (argmax of K-means distance) -> hard specialist prediction
       - soft cluster membership probabilities (softmax over neg-distance) -> soft blend

  4. Final specialist prediction = soft-weighted sum of all K specialists' outputs.

  5. We DON'T replace the existing ensemble — we add the specialist as one MORE
     base column for the stack. If specialists capture signal the global model
     misses, this should improve final ensemble metrics.

Run:
    python -m src.improved.specialists --k 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from xgboost import XGBClassifier

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.improved.data_utils import load_dataset, scaffold_split
else:
    from .data_utils import load_dataset, scaffold_split

RDLogger.DisableLog("rdApp.*")


def morgan(smiles: str, n_bits: int = 512, radius: int = 2) -> np.ndarray:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _best_thr(y, p):
    best_t, best_v = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        v = matthews_corrcoef(y, (p >= t).astype(int))
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


def soft_cluster_membership(centroids: np.ndarray, X: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Convert distances to soft cluster probabilities via negative-distance softmax."""
    dist = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=-1)  # [N, K]
    neg_d = -dist / temperature
    # numerical-stable softmax
    neg_d -= neg_d.max(axis=1, keepdims=True)
    expv = np.exp(neg_d)
    return expv / expv.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4, help="Number of scaffold clusters")
    ap.add_argument("--temperature", type=float, default=2.0)
    args = ap.parse_args()

    base = Path(__file__).resolve().parents[2]
    res = base / "results"
    print(f"[specialists] K={args.k}  temperature={args.temperature}", flush=True)

    # 1. Load data + same scaffold split as everything else.
    graphs = load_dataset(base / "data" / "dili_clean.csv")
    smiles = [g.smiles for g in graphs]
    labels = np.array([int(g.y.item()) for g in graphs], dtype=np.int64)
    train_idx, val_idx, test_idx = scaffold_split(graphs, 0.6, 0.2, 0.2, seed=42)
    y_train = labels[train_idx]; y_val = labels[val_idx]; y_test = labels[test_idx]

    # 2. Build feature blocks: MolFormer-emb + Morgan-FP + descriptors + Tox21
    print("[specialists] loading MolFormer embeddings (cached) ...", flush=True)
    molformer = np.load(res / "molformer_embeddings.npy")
    print(f"  MolFormer embeddings: {molformer.shape}", flush=True)

    print("[specialists] computing Morgan FPs ...", flush=True)
    fps = np.stack([morgan(s) for s in smiles], axis=0)

    print("[specialists] gathering descriptors from graphs ...", flush=True)
    desc = np.stack([g.u.cpu().numpy().reshape(-1) for g in graphs], axis=0)

    print("[specialists] loading Tox21 12-endpoint predictions ...", flush=True)
    tox = np.load(res / "tox21_all_preds.npy")
    print(f"  Tox21 predictions: {tox.shape}", flush=True)

    X_all = np.concatenate([molformer, fps, desc, tox], axis=1)
    print(f"[specialists] combined feature matrix: {X_all.shape}", flush=True)

    # 3. K-means on MolFormer embeddings (just the 768-d), TRAIN ONLY (no leakage).
    print(f"\n[specialists] K-means clustering on MolFormer embeddings (K={args.k}) ...", flush=True)
    km = KMeans(n_clusters=args.k, random_state=42, n_init=10)
    km.fit(molformer[train_idx])
    centroids = km.cluster_centers_

    # Cluster assignment for everyone.
    train_hard = km.predict(molformer[train_idx])
    val_hard = km.predict(molformer[val_idx])
    test_hard = km.predict(molformer[test_idx])
    print("  cluster sizes (train):", np.bincount(train_hard, minlength=args.k))
    print("  cluster sizes (val):  ", np.bincount(val_hard, minlength=args.k))
    print("  cluster sizes (test): ", np.bincount(test_hard, minlength=args.k))

    # 4. Train one specialist XGBoost per cluster.
    specialists = []
    cluster_pos_rates = []
    for k in range(args.k):
        mask = (train_hard == k)
        if mask.sum() < 30:
            print(f"  [cluster {k}] too few train samples ({mask.sum()}), skipping", flush=True)
            specialists.append(None)
            cluster_pos_rates.append(0.5)
            continue
        Xk = X_all[train_idx][mask]
        yk = y_train[mask]
        pos_rate = yk.mean()
        cluster_pos_rates.append(pos_rate)
        pos, neg = int(yk.sum()), int((1 - yk).sum())
        if pos == 0 or neg == 0:
            print(f"  [cluster {k}] mono-class (pos={pos}, neg={neg}), using global rate", flush=True)
            specialists.append(None)
            continue
        spw = neg / max(pos, 1)
        clf = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.6,
            reg_lambda=1.0, scale_pos_weight=spw,
            eval_metric="logloss", random_state=42, n_jobs=4, verbosity=0,
        )
        clf.fit(Xk, yk)
        specialists.append(clf)
        print(f"  [cluster {k}] trained on n={mask.sum()} (pos={pos} neg={neg} pos_rate={pos_rate:.2f})", flush=True)

    # 5. For val + test, get per-specialist predictions and soft-weighted blend.
    def predict_blend(X_emb, X_full):
        """X_emb: MolFormer embedding for clustering. X_full: full feature vector
        for specialist prediction. Returns soft-weighted blended probabilities."""
        memb = soft_cluster_membership(centroids, X_emb, temperature=args.temperature)
        # Per-specialist predictions: [N, K]
        N, K = X_full.shape[0], args.k
        spec_preds = np.zeros((N, K))
        for k in range(K):
            if specialists[k] is None:
                spec_preds[:, k] = cluster_pos_rates[k]
            else:
                spec_preds[:, k] = specialists[k].predict_proba(X_full)[:, 1]
        # Soft blend
        return (spec_preds * memb).sum(axis=1), spec_preds, memb

    val_X_emb = molformer[val_idx]; val_X_full = X_all[val_idx]
    test_X_emb = molformer[test_idx]; test_X_full = X_all[test_idx]
    p_val_blend, _, val_memb = predict_blend(val_X_emb, val_X_full)
    p_test_blend, _, test_memb = predict_blend(test_X_emb, test_X_full)

    # Also report hard routing for comparison
    p_val_hard = np.array([
        (specialists[val_hard[i]].predict_proba(val_X_full[i:i+1])[0, 1] if specialists[val_hard[i]] is not None
         else cluster_pos_rates[val_hard[i]])
        for i in range(len(val_idx))
    ])
    p_test_hard = np.array([
        (specialists[test_hard[i]].predict_proba(test_X_full[i:i+1])[0, 1] if specialists[test_hard[i]] is not None
         else cluster_pos_rates[test_hard[i]])
        for i in range(len(test_idx))
    ])

    print("\n[specialists] Hard routing (argmax cluster):")
    t = _best_thr(y_val, p_val_hard)
    m = _m(y_test, p_test_hard, t)
    print(f"  thr={t:.2f}  AUROC={m['AUROC']:.4f}  ACC={m['ACC']:.4f}  F1={m['F1']:.4f}  MCC={m['MCC']:.4f}", flush=True)

    print("\n[specialists] Soft routing (softmax cluster, T={}):".format(args.temperature))
    t = _best_thr(y_val, p_val_blend)
    m = _m(y_test, p_test_blend, t)
    print(f"  thr={t:.2f}  AUROC={m['AUROC']:.4f}  ACC={m['ACC']:.4f}  F1={m['F1']:.4f}  MCC={m['MCC']:.4f}", flush=True)

    # Save predictions for the stack.
    np.savez(
        res / "specialists_test_probs.npz",
        y_true=y_test, specialists_hard=p_test_hard, specialists_soft=p_test_blend,
    )
    np.savez(
        res / "specialists_val_probs.npz",
        y_true=y_val, specialists_hard=p_val_hard, specialists_soft=p_val_blend,
    )

    # Per-cluster diagnostic: hold-out validation AUC within each cluster.
    rows = []
    for k in range(args.k):
        v_mask = (val_hard == k)
        if v_mask.sum() < 5 or len(set(y_val[v_mask])) < 2 or specialists[k] is None:
            continue
        p_k = specialists[k].predict_proba(val_X_full[v_mask])[:, 1]
        auc_k = roc_auc_score(y_val[v_mask], p_k)
        rows.append({"cluster": k, "n_val": int(v_mask.sum()), "val_auc": float(auc_k),
                     "train_n": int((train_hard == k).sum())})
    pd.DataFrame(rows).to_csv(res / "specialists_per_cluster.csv", index=False)
    print(f"\n[specialists] wrote results/specialists_*.npz + specialists_per_cluster.csv", flush=True)


if __name__ == "__main__":
    main()
