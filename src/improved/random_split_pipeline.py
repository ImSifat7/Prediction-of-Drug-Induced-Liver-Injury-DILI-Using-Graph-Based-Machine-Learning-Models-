"""End-to-end retrain on a RANDOM 60/20/20 split (vs the scaffold split used
in the main pipeline).

Why this exists: many published DILI papers report random-split numbers.
Scaffold split is the hard split (test molecules have novel scaffolds);
random split lets information leak between similar molecules. Reporting
both is honest and gives a fuller picture of model performance under
different deployment scenarios.

Trains the FAST components only (no ChemBERTa fine-tune):
  - 5 GNNs x 3 seeds each       (~10 min)
  - MolFormer-XL frozen + XGBoost (~1 min, embeddings cached)
  - Hybrid GAT/GIN/SAGE + XGB    (~5 min)
  - Specialists (K-means + XGB)  (~5 min)
  - Tox21 12-endpoint re-index   (<1 min, predictions cached)
  - Final stack                  (<1 min)

All predictions are saved as `*_random_*.npz`. Stack output:
  results/stack_random_metrics.csv
  results/stack_random_log.txt

Run:
    python -m src.improved.random_split_pipeline
"""

from __future__ import annotations

import json
import random as _py_random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, roc_auc_score,
)
from xgboost import XGBClassifier

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.improved.data_utils import load_dataset
    from src.improved.models import build_model
    from src.improved.train_utils import (
        compute_pos_weight, make_loader, set_seed, train_model, evaluate,
    )
else:
    from .data_utils import load_dataset
    from .models import build_model
    from .train_utils import (
        compute_pos_weight, make_loader, set_seed, train_model, evaluate,
    )

RDLogger.DisableLog("rdApp.*")

SPLIT_SEED = 42
N_SEEDS = 3              # per-architecture seeds for GNN bagging
GNN_NAMES = ["GCN", "GAT", "GraphSAGE", "GIN", "MPNN"]
EPOCHS = 80
PATIENCE = 15
FP_BITS = 2048


def random_split(n: int, frac_train=0.6, frac_val=0.2, seed=42):
    rng = _py_random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    n_tr = int(frac_train * n)
    n_va = int(frac_val * n)
    return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]


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


def morgan(smiles: str, n_bits: int = FP_BITS, radius: int = 2) -> np.ndarray:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


@torch.no_grad()
def extract_pooled(model, loader, device) -> np.ndarray:
    model.eval().to(device)
    captured: List[np.ndarray] = []
    first_linear = model.fc[0]
    def _hook(_m, inputs, _o):
        captured.append(inputs[0].detach().cpu().numpy())
    h = first_linear.register_forward_hook(_hook)
    try:
        for batch in loader:
            batch = batch.to(device)
            _ = model(batch)
    finally:
        h.remove()
    return np.concatenate(captured, axis=0)


def main():
    base = Path(__file__).resolve().parents[2]
    res = base / "results"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[random_split] device={device}", flush=True)

    print("[random_split] loading graphs ...", flush=True)
    graphs = load_dataset(base / "data" / "dili_clean.csv")
    n = len(graphs)
    smiles = [g.smiles for g in graphs]
    labels = np.array([int(g.y.item()) for g in graphs], dtype=np.int64)

    train_idx, val_idx, test_idx = random_split(n, 0.6, 0.2, seed=SPLIT_SEED)
    print(f"[random_split] RANDOM split  tr={len(train_idx)} va={len(val_idx)} te={len(test_idx)}",
          flush=True)
    print(f"   pos_rate  tr={labels[train_idx].mean():.3f}  "
          f"va={labels[val_idx].mean():.3f}  te={labels[test_idx].mean():.3f}", flush=True)

    y_tr = labels[train_idx]; y_va = labels[val_idx]; y_te = labels[test_idx]
    in_dim = graphs[0].x.shape[1]; edge_dim = graphs[0].edge_attr.shape[1]
    desc_dim = graphs[0].u.shape[1]

    # -------------------------------------------------------------------
    # 1. GNNs: 5 architectures x N_SEEDS seeds, bagged predictions
    # -------------------------------------------------------------------
    gnn_val: Dict[str, np.ndarray] = {}
    gnn_test: Dict[str, np.ndarray] = {}
    gnn_emb_per_arch: Dict[str, np.ndarray] = {}  # for Hybrid
    t_start = time.time()
    for name in GNN_NAMES:
        cfg = json.loads((base / "configs" / f"{name}_best.json").read_text())["params"]
        bs = cfg["batch_size"]
        p_va_bag = np.zeros(len(val_idx))
        p_te_bag = np.zeros(len(test_idx))
        emb_bag = np.zeros((n, 0))  # filled after first seed (assumes consistent hidden dim)
        for sd in range(N_SEEDS):
            seed = SPLIT_SEED + sd
            set_seed(seed)
            model = build_model(name, in_dim, edge_dim, cfg, desc_dim=desc_dim)
            tr_loader = make_loader(graphs, train_idx, batch_size=bs, shuffle=True)
            va_loader = make_loader(graphs, val_idx, batch_size=bs, shuffle=False)
            te_loader = make_loader(graphs, test_idx, batch_size=bs, shuffle=False)
            pos_w = compute_pos_weight(graphs, train_idx)
            t0 = time.time()
            model, best_val = train_model(
                model, tr_loader, va_loader,
                lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                optimizer_name=cfg["optimizer_name"], pos_weight=pos_w,
                epochs=EPOCHS, patience=PATIENCE, device=device,
            )
            p_va = np.array(evaluate(model, va_loader, device)["y_prob"])
            p_te = np.array(evaluate(model, te_loader, device)["y_prob"])
            p_va_bag += p_va; p_te_bag += p_te
            if name in {"GAT", "GIN", "GraphSAGE"} and sd == 0:
                all_loader = make_loader(graphs, list(range(n)), batch_size=64, shuffle=False)
                emb_bag = extract_pooled(model, all_loader, device)
            print(f"  {name} sd={sd} val_auc={best_val:.4f} "
                  f"test_auc={roc_auc_score(y_te, p_te):.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        p_va_bag /= N_SEEDS; p_te_bag /= N_SEEDS
        gnn_val[name] = p_va_bag; gnn_test[name] = p_te_bag
        if name in {"GAT", "GIN", "GraphSAGE"}:
            gnn_emb_per_arch[name] = emb_bag
        auc = roc_auc_score(y_te, p_te_bag)
        print(f"  --> {name}  BAGGED test AUROC={auc:.4f}", flush=True)
    print(f"[random_split] GNN total: {(time.time()-t_start)/60:.1f} min", flush=True)

    np.savez(res / "gnn_random_val_probs.npz", y_true=y_va, **{k: gnn_val[k] for k in gnn_val})
    np.savez(res / "gnn_random_test_probs.npz", y_true=y_te, **{k: gnn_test[k] for k in gnn_test})

    # -------------------------------------------------------------------
    # 2. MolFormer-XL + bagged XGBoost on cached embeddings
    # -------------------------------------------------------------------
    print("\n[random_split] MolFormer-XL frozen + bagged XGBoost ...", flush=True)
    molformer = np.load(res / "molformer_embeddings.npy")
    X_tr = molformer[train_idx]; X_va = molformer[val_idx]; X_te = molformer[test_idx]
    pos = (y_tr == 1).sum(); neg = (y_tr == 0).sum()
    spw = float(neg) / max(int(pos), 1)
    mf_va = np.zeros(len(val_idx)); mf_te = np.zeros(len(test_idx))
    for sd in range(5):
        clf = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.7,
            reg_lambda=1.0, scale_pos_weight=spw,
            eval_metric="logloss", random_state=sd, n_jobs=4, verbosity=0,
        )
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        mf_va += clf.predict_proba(X_va)[:, 1]
        mf_te += clf.predict_proba(X_te)[:, 1]
    mf_va /= 5; mf_te /= 5
    print(f"  MolFormer test AUROC={roc_auc_score(y_te, mf_te):.4f}", flush=True)
    np.savez(res / "molformer_random_val_probs.npz", y_true=y_va, molformer=mf_va)
    np.savez(res / "molformer_random_test_probs.npz", y_true=y_te, molformer=mf_te)

    # -------------------------------------------------------------------
    # 3. Hybrid GAT/GIN/SAGE + global desc + Morgan-FP -> XGBoost
    # -------------------------------------------------------------------
    print("\n[random_split] Hybrid (GAT||GIN||SAGE || desc || Morgan-FP) -> XGBoost ...",
          flush=True)
    desc_block = np.stack([g.u.cpu().numpy().reshape(-1) for g in graphs], axis=0)
    fp_block = np.stack([morgan(s) for s in smiles], axis=0)
    hy_emb = np.concatenate([gnn_emb_per_arch[a] for a in ("GAT", "GIN", "GraphSAGE")], axis=1)
    X_hy = np.concatenate([hy_emb, desc_block, fp_block], axis=1)
    X_tr = X_hy[train_idx]; X_va = X_hy[val_idx]; X_te = X_hy[test_idx]
    hy_va = np.zeros(len(val_idx)); hy_te = np.zeros(len(test_idx))
    for sd in range(5):
        clf = XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.5,
            reg_lambda=1.0, scale_pos_weight=spw,
            eval_metric="logloss", random_state=sd, n_jobs=4, verbosity=0,
        )
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        hy_va += clf.predict_proba(X_va)[:, 1]
        hy_te += clf.predict_proba(X_te)[:, 1]
    hy_va /= 5; hy_te /= 5
    print(f"  Hybrid test AUROC={roc_auc_score(y_te, hy_te):.4f}", flush=True)
    np.savez(res / "hybrid_random_val_probs.npz", y_true=y_va, hybrid=hy_va)
    np.savez(res / "hybrid_random_test_probs.npz", y_true=y_te, hybrid=hy_te)

    # -------------------------------------------------------------------
    # 4. Specialists (K-means on MolFormer, per-cluster XGBoost, soft routing)
    # -------------------------------------------------------------------
    print("\n[random_split] Specialists (K=2 on MolFormer-emb) ...", flush=True)
    K = 2
    km = KMeans(n_clusters=K, random_state=SPLIT_SEED, n_init="auto")
    km.fit(molformer[train_idx])
    feats = np.concatenate([molformer, desc_block, fp_block], axis=1)
    train_cluster = km.predict(molformer[train_idx])
    clfs = []
    for k in range(K):
        mask = (train_cluster == k)
        if mask.sum() < 10:
            clfs.append(None); continue
        Xk = feats[train_idx][mask]; yk = y_tr[mask]
        pos_k = (yk == 1).sum(); neg_k = (yk == 0).sum()
        spw_k = float(neg_k) / max(int(pos_k), 1)
        clf = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.7,
            reg_lambda=1.0, scale_pos_weight=spw_k,
            eval_metric="logloss", random_state=SPLIT_SEED, n_jobs=4, verbosity=0,
        )
        clf.fit(Xk, yk); clfs.append(clf)
    def soft_pred(arr_idx):
        d = np.linalg.norm(molformer[arr_idx][:, None, :] - km.cluster_centers_[None, :, :], axis=2)
        w = np.exp(-d); w /= w.sum(axis=1, keepdims=True)
        probs = np.zeros((len(arr_idx), K))
        for k in range(K):
            if clfs[k] is not None:
                probs[:, k] = clfs[k].predict_proba(feats[arr_idx])[:, 1]
        return (probs * w).sum(axis=1)
    sp_va = soft_pred(val_idx)
    sp_te = soft_pred(test_idx)
    print(f"  Specialists test AUROC={roc_auc_score(y_te, sp_te):.4f}", flush=True)
    np.savez(res / "specialists_random_val_probs.npz", y_true=y_va, specialists_soft=sp_va)
    np.savez(res / "specialists_random_test_probs.npz", y_true=y_te, specialists_soft=sp_te)

    # -------------------------------------------------------------------
    # 5. Tox21 re-index (predictions already exist for all 869 molecules)
    # -------------------------------------------------------------------
    print("\n[random_split] Tox21 12-endpoint re-index ...", flush=True)
    tox_all = np.load(res / "tox21_all_preds.npy")  # (869, 12)
    tox_endpoints = [f"tox21_{i}" for i in range(tox_all.shape[1])]
    tx_va_dict = {ep: tox_all[val_idx, i] for i, ep in enumerate(tox_endpoints)}
    tx_te_dict = {ep: tox_all[test_idx, i] for i, ep in enumerate(tox_endpoints)}
    np.savez(res / "tox21_random_val_probs.npz", y_true=y_va, **tx_va_dict,
             tox21_mean=tox_all[val_idx].mean(axis=1))
    np.savez(res / "tox21_random_test_probs.npz", y_true=y_te, **tx_te_dict,
             tox21_mean=tox_all[test_idx].mean(axis=1))
    print(f"  Tox21 mean-of-12 test AUROC={roc_auc_score(y_te, tox_all[test_idx].mean(axis=1)):.4f}",
          flush=True)

    # -------------------------------------------------------------------
    # 6. STACK on random-split predictions
    # -------------------------------------------------------------------
    print("\n[random_split] STACK ====================", flush=True)
    parts_v, parts_t, cols = [], [], []
    for name in GNN_NAMES:
        parts_v.append(gnn_val[name]); parts_t.append(gnn_test[name]); cols.append(name)
    parts_v.append(mf_va); parts_t.append(mf_te); cols.append("molformer")
    parts_v.append(hy_va); parts_t.append(hy_te); cols.append("hybrid")
    parts_v.append(sp_va); parts_t.append(sp_te); cols.append("specialists_soft")
    tx_v = tox_all[val_idx].mean(axis=1)
    tx_t = tox_all[test_idx].mean(axis=1)
    parts_v.append(tx_v); parts_t.append(tx_t); cols.append("tox21_mean")

    V = np.stack(parts_v, axis=1); T = np.stack(parts_t, axis=1)
    print(f"  {len(cols)} base columns: {cols}", flush=True)

    rows = []
    def add(label, method, p_v, p_t):
        for tgt in ("MCC", "ACC", "F1"):
            t = _best_thr(y_va, p_v, target=tgt)
            m = _m(y_te, p_t, t)
            rows.append({"variant": label, "method": method, "tuned_for": tgt,
                         "threshold": t, **m})

    print("\n[individual base models]")
    for j, c in enumerate(cols):
        t = _best_thr(y_va, V[:, j])
        m = _m(y_te, T[:, j], t)
        print(f"  {c:<20}  AUROC={m['AUROC']:.4f}  ACC={m['ACC']:.4f}  "
              f"F1={m['F1']:.4f}  MCC={m['MCC']:.4f}", flush=True)
        rows.append({"variant": "single", "method": c, "tuned_for": "MCC",
                     "threshold": t, **m})

    print("\n[ensembles]")
    add("full", "mean", V.mean(axis=1), T.mean(axis=1))
    eps = 1e-6
    gV = np.exp(np.log(np.clip(V, eps, 1 - eps)).mean(axis=1))
    gT = np.exp(np.log(np.clip(T, eps, 1 - eps)).mean(axis=1))
    add("full", "geom-mean", gV, gT)
    add("full", "median", np.median(V, axis=1), np.median(T, axis=1))
    def rank(M):
        return np.stack([np.argsort(np.argsort(M[:, j])) / max(len(M) - 1, 1)
                         for j in range(M.shape[1])], axis=1)
    add("full", "rank-avg", rank(V).mean(axis=1), rank(T).mean(axis=1))

    gn_v = V[:, :5].mean(axis=1); gn_t = T[:, :5].mean(axis=1)
    mf_v_b = V[:, 5];  mf_t_b = T[:, 5]
    hy_v_b = V[:, 6];  hy_t_b = T[:, 6]
    sp_v_b = V[:, 7];  sp_t_b = T[:, 7]
    tx_v_b = V[:, 8];  tx_t_b = T[:, 8]

    for w_gnn, w_hy, w_mf, w_tx, name in [
        (0.25, 0.25, 0.25, 0.25, "equal4"),
        (0.10, 0.30, 0.40, 0.20, "mf-heavy4"),
        (0.20, 0.30, 0.30, 0.20, "balanced4"),
        (0.00, 0.40, 0.40, 0.20, "no-gnn4"),
    ]:
        p_v = w_gnn*gn_v + w_hy*hy_v_b + w_mf*mf_v_b + w_tx*tx_v_b
        p_t = w_gnn*gn_t + w_hy*hy_t_b + w_mf*mf_t_b + w_tx*tx_t_b
        add("4-way", f"weighted-{name}", p_v, p_t)

    for w_gnn, w_hy, w_mf, w_sp, w_tx, name in [
        (0.20, 0.20, 0.20, 0.20, 0.20, "equal5"),
        (0.10, 0.25, 0.30, 0.20, 0.15, "mf-heavy5"),
        (0.10, 0.30, 0.30, 0.15, 0.15, "hy+mf-heavy5"),
        (0.00, 0.30, 0.30, 0.25, 0.15, "no-gnn5"),
        (0.00, 0.25, 0.30, 0.30, 0.15, "no-gnn5-sp-heavy"),
    ]:
        p_v = w_gnn*gn_v + w_hy*hy_v_b + w_mf*mf_v_b + w_sp*sp_v_b + w_tx*tx_v_b
        p_t = w_gnn*gn_t + w_hy*hy_t_b + w_mf*mf_t_b + w_sp*sp_t_b + w_tx*tx_t_b
        add("5-way", f"weighted-{name}", p_v, p_t)

    meta = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=42)
    meta.fit(V, y_va)
    add("meta", "logreg-C1", meta.predict_proba(V)[:, 1], meta.predict_proba(T)[:, 1])
    meta2 = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced", random_state=42)
    meta2.fit(V, y_va)
    add("meta", "logreg-C0.1", meta2.predict_proba(V)[:, 1], meta2.predict_proba(T)[:, 1])

    df = pd.DataFrame(rows)
    df.to_csv(res / "stack_random_metrics.csv", index=False)
    print(f"\n[random_split] wrote results/stack_random_metrics.csv", flush=True)

    print("\n[TOP 15 by AUROC]")
    print(df.sort_values("AUROC", ascending=False).head(15).to_string(index=False))
    print("\n[TOP 10 by ACC]")
    print(df.sort_values("ACC", ascending=False).head(10).to_string(index=False))
    print("\n[TOP 10 by MCC]")
    print(df.sort_values("MCC", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
