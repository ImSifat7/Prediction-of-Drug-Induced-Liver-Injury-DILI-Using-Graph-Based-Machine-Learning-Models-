"""End-to-end pipeline on the TDC DILI benchmark (Xu et al. 2015, 475 drugs).

Why this exists: DILIrank 2.0 ceilings around AUROC 0.74-0.76 due to dataset
size (521 train) and label noise (vMost+vLess mixed).  TDC DILI is the
canonical modern benchmark — balanced (50/50), clean expert labels, larger
fraction usable, and a standard scaffold split that the TDC leaderboard
uses.  Published SOTA on TDC DILI: 0.82-0.89 AUROC.

Pipeline (same architecture as DILIrank pipeline, retrained on TDC):
  1. Build graphs with 54-dim atoms + 14-dim global descriptors
  2. Scaffold split 70/10/20  (TDC standard)
  3. Train 5 GNNs x 3 seeds (bagged)
  4. MolFormer-XL frozen embeddings + bagged XGBoost
  5. ChemBERTa-77M fine-tune (3 seeds)
  6. Hybrid GAT||GIN||SAGE + global + Morgan -> XGBoost
  7. Tox21 12-endpoint re-prediction (the RFs were trained on Tox21, so they
     work on any SMILES — re-predict for the 475 TDC molecules)
  8. Final stack

Run:
    python -m src.improved.tdc_pipeline
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
    from src.improved.data_utils import (
        smiles_to_data, standardize_smiles, scaffold_split,
    )
    from src.improved.models import build_model
    from src.improved.train_utils import (
        compute_pos_weight, make_loader, set_seed, train_model, evaluate,
    )
else:
    from .data_utils import (
        smiles_to_data, standardize_smiles, scaffold_split,
    )
    from .models import build_model
    from .train_utils import (
        compute_pos_weight, make_loader, set_seed, train_model, evaluate,
    )

RDLogger.DisableLog("rdApp.*")

SEED = 42
N_GNN_SEEDS = 3
GNN_NAMES = ["GCN", "GAT", "GraphSAGE", "GIN", "MPNN"]
EPOCHS = 80
PATIENCE = 15
FP_BITS = 2048
N_BAG_XGB = 5


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


def load_tdc_dili():
    """Read TDC DILI csv, standardize SMILES, build graphs."""
    df = pd.read_csv("data/tdc_dili.csv")
    rows = []
    for _, r in df.iterrows():
        canon = standardize_smiles(r["Drug"])
        if canon is None:
            continue
        rows.append((canon, int(r["Y"])))
    seen, kept = set(), []
    for s, y in rows:
        if s in seen:
            continue
        seen.add(s); kept.append((s, y))
    print(f"[tdc] after standardize+dedup: {len(kept)} compounds  "
          f"(pos_rate={np.mean([y for _, y in kept]):.3f})", flush=True)
    graphs = []
    for s, y in kept:
        g = smiles_to_data(s, int(y))
        if g is None:
            continue
        graphs.append(g)
    print(f"[tdc] graphs built: {len(graphs)}", flush=True)
    return graphs


def main():
    base = Path(__file__).resolve().parents[2]
    res = base / "results"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[tdc] device={device}", flush=True)

    # ---------- 1. Data ----------
    graphs = load_tdc_dili()
    smiles = [g.smiles for g in graphs]
    labels = np.array([int(g.y.item()) for g in graphs], dtype=np.int64)

    # TDC standard split: 70/10/20 scaffold (same Bemis-Murcko function).
    train_idx, val_idx, test_idx = scaffold_split(graphs, 0.7, 0.1, 0.2, seed=SEED)
    print(f"[tdc] scaffold split 70/10/20  tr={len(train_idx)} va={len(val_idx)} te={len(test_idx)}",
          flush=True)
    print(f"   pos_rate  tr={labels[train_idx].mean():.3f}  "
          f"va={labels[val_idx].mean():.3f}  te={labels[test_idx].mean():.3f}", flush=True)

    y_tr = labels[train_idx]; y_va = labels[val_idx]; y_te = labels[test_idx]
    in_dim = graphs[0].x.shape[1]
    edge_dim = graphs[0].edge_attr.shape[1]
    desc_dim = graphs[0].u.shape[1]

    # ---------- 2. GNNs ----------
    print("\n[tdc] training 5 GNNs x 3 seeds each ...", flush=True)
    gnn_val: Dict[str, np.ndarray] = {}
    gnn_test: Dict[str, np.ndarray] = {}
    gnn_emb_per_arch: Dict[str, np.ndarray] = {}
    t_start = time.time()
    for name in GNN_NAMES:
        cfg = json.loads((base / "configs" / f"{name}_best.json").read_text())["params"]
        bs = cfg["batch_size"]
        p_va_bag = np.zeros(len(val_idx))
        p_te_bag = np.zeros(len(test_idx))
        emb_bag = None
        for sd in range(N_GNN_SEEDS):
            set_seed(SEED + sd)
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
                all_loader = make_loader(graphs, list(range(len(graphs))), batch_size=64, shuffle=False)
                emb_bag = extract_pooled(model, all_loader, device)
            print(f"  {name} sd={sd} val_auc={best_val:.4f} "
                  f"test_auc={roc_auc_score(y_te, p_te):.4f}  ({time.time()-t0:.0f}s)",
                  flush=True)
        p_va_bag /= N_GNN_SEEDS; p_te_bag /= N_GNN_SEEDS
        gnn_val[name] = p_va_bag; gnn_test[name] = p_te_bag
        if emb_bag is not None:
            gnn_emb_per_arch[name] = emb_bag
        print(f"  --> {name} BAGGED test AUROC={roc_auc_score(y_te, p_te_bag):.4f}",
              flush=True)
    print(f"[tdc] GNN total: {(time.time()-t_start)/60:.1f} min", flush=True)
    np.savez(res / "tdc_gnn_val_probs.npz", y_true=y_va, **gnn_val)
    np.savez(res / "tdc_gnn_test_probs.npz", y_true=y_te, **gnn_test)

    # ---------- 3. MolFormer-XL embeddings on TDC molecules ----------
    print("\n[tdc] extracting MolFormer-XL embeddings for TDC molecules ...", flush=True)
    from src.improved.molformer import _patch_transformers_for_molformer
    _patch_transformers_for_molformer()
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
    mfm = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct",
                                    deterministic_eval=True, trust_remote_code=True).to(device)
    mfm.eval()
    embs = []
    bs_emb = 32
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(smiles), bs_emb):
            chunk = smiles[i:i+bs_emb]
            enc = tok(chunk, padding=True, return_tensors="pt").to(device)
            out = mfm(**enc)
            pooled = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None \
                     else out.last_hidden_state.mean(dim=1)
            embs.append(pooled.detach().cpu().numpy())
    embs = np.concatenate(embs, axis=0)
    print(f"  MolFormer embeddings shape={embs.shape}, took {time.time()-t0:.1f}s", flush=True)
    np.save(res / "tdc_molformer_embeddings.npy", embs)

    # MolFormer bagged XGBoost
    X_tr = embs[train_idx]; X_va = embs[val_idx]; X_te = embs[test_idx]
    pos = (y_tr == 1).sum(); neg = (y_tr == 0).sum()
    spw = float(neg) / max(int(pos), 1)
    mf_va = np.zeros(len(val_idx)); mf_te = np.zeros(len(test_idx))
    for sd in range(N_BAG_XGB):
        clf = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.7,
            reg_lambda=1.0, scale_pos_weight=spw,
            eval_metric="logloss", random_state=sd, n_jobs=4, verbosity=0,
        )
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        mf_va += clf.predict_proba(X_va)[:, 1]
        mf_te += clf.predict_proba(X_te)[:, 1]
    mf_va /= N_BAG_XGB; mf_te /= N_BAG_XGB
    print(f"  MolFormer test AUROC={roc_auc_score(y_te, mf_te):.4f}", flush=True)
    np.savez(res / "tdc_molformer_val_probs.npz", y_true=y_va, molformer=mf_va)
    np.savez(res / "tdc_molformer_test_probs.npz", y_true=y_te, molformer=mf_te)

    # ---------- 4. Hybrid (GAT||GIN||SAGE + global desc + Morgan FP) ----------
    print("\n[tdc] Hybrid (GNN-emb || desc || Morgan-FP) -> XGBoost ...", flush=True)
    desc_block = np.stack([g.u.cpu().numpy().reshape(-1) for g in graphs], axis=0)
    fp_block = np.stack([morgan(s) for s in smiles], axis=0)
    hy_emb = np.concatenate([gnn_emb_per_arch[a] for a in ("GAT", "GIN", "GraphSAGE")], axis=1)
    X_hy = np.concatenate([hy_emb, desc_block, fp_block], axis=1)
    X_tr = X_hy[train_idx]; X_va = X_hy[val_idx]; X_te = X_hy[test_idx]
    hy_va = np.zeros(len(val_idx)); hy_te = np.zeros(len(test_idx))
    for sd in range(N_BAG_XGB):
        clf = XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.5,
            reg_lambda=1.0, scale_pos_weight=spw,
            eval_metric="logloss", random_state=sd, n_jobs=4, verbosity=0,
        )
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        hy_va += clf.predict_proba(X_va)[:, 1]
        hy_te += clf.predict_proba(X_te)[:, 1]
    hy_va /= N_BAG_XGB; hy_te /= N_BAG_XGB
    print(f"  Hybrid test AUROC={roc_auc_score(y_te, hy_te):.4f}", flush=True)
    np.savez(res / "tdc_hybrid_val_probs.npz", y_true=y_va, hybrid=hy_va)
    np.savez(res / "tdc_hybrid_test_probs.npz", y_true=y_te, hybrid=hy_te)

    # ---------- 5. ChemBERTa fine-tune (3 seeds) ----------
    print("\n[tdc] ChemBERTa-77M fine-tune (3 seeds, 25 epochs each) ...", flush=True)
    from transformers import AutoTokenizer as T2, AutoModelForSequenceClassification
    import torch.nn as nn
    from torch.utils.data import DataLoader as TD, TensorDataset
    cb_tok = T2.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
    cb_va_bag = np.zeros(len(val_idx)); cb_te_bag = np.zeros(len(test_idx))
    cb_val_perseed, cb_test_perseed = {}, {}
    for sd, seed in enumerate([42, 7, 13]):
        set_seed(seed)
        model = AutoModelForSequenceClassification.from_pretrained(
            "DeepChem/ChemBERTa-77M-MTR", num_labels=1, ignore_mismatched_sizes=True,
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
        pos_w = torch.tensor([spw], dtype=torch.float32, device=device)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        def enc(idx_list):
            t = cb_tok([smiles[i] for i in idx_list], padding=True, truncation=True,
                       max_length=200, return_tensors="pt")
            return t.input_ids, t.attention_mask
        tr_ids, tr_mask = enc(train_idx); va_ids, va_mask = enc(val_idx); te_ids, te_mask = enc(test_idx)
        tr_y = torch.tensor(y_tr, dtype=torch.float32)
        train_ds = TensorDataset(tr_ids, tr_mask, tr_y)
        loader = TD(train_ds, batch_size=16, shuffle=True)
        best_val = -1.0; best_va_p = None; best_te_p = None
        patience = 5; bad = 0
        for ep in range(25):
            model.train()
            t_ep = time.time(); ep_loss = 0; nb = 0
            for ids, mask, y in loader:
                ids, mask, y = ids.to(device), mask.to(device), y.to(device)
                logits = model(input_ids=ids, attention_mask=mask).logits.view(-1)
                loss = bce(logits, y)
                opt.zero_grad(); loss.backward(); opt.step()
                ep_loss += loss.item(); nb += 1
            model.eval()
            with torch.no_grad():
                p_va = torch.sigmoid(
                    model(input_ids=va_ids.to(device),
                          attention_mask=va_mask.to(device)).logits.view(-1)
                ).cpu().numpy()
                p_te = torch.sigmoid(
                    model(input_ids=te_ids.to(device),
                          attention_mask=te_mask.to(device)).logits.view(-1)
                ).cpu().numpy()
            va_auc = roc_auc_score(y_va, p_va)
            if va_auc > best_val:
                best_val = va_auc; best_va_p = p_va.copy(); best_te_p = p_te.copy(); bad = 0
            else:
                bad += 1
            if (ep + 1) % 5 == 0 or bad >= patience:
                print(f"   sd={seed} ep{ep+1:02d} loss={ep_loss/nb:.4f} val_auc={va_auc:.4f}"
                      f"  best={best_val:.4f} ({time.time()-t_ep:.0f}s)", flush=True)
            if bad >= patience:
                print(f"   sd={seed} early-stop ep{ep+1}", flush=True); break
        cb_va_bag += best_va_p; cb_te_bag += best_te_p
        cb_val_perseed[f"ChemBERTa_s{seed}"] = best_va_p
        cb_test_perseed[f"ChemBERTa_s{seed}"] = best_te_p
        print(f"  CB seed={seed} val={best_val:.4f} test={roc_auc_score(y_te, best_te_p):.4f}",
              flush=True)
    cb_va_bag /= 3; cb_te_bag /= 3
    print(f"  ChemBERTa MEAN-of-3 test AUROC={roc_auc_score(y_te, cb_te_bag):.4f}", flush=True)
    np.savez(res / "tdc_chemberta_val_probs.npz", y_true=y_va, ChemBERTa=cb_va_bag, **cb_val_perseed)
    np.savez(res / "tdc_chemberta_test_probs.npz", y_true=y_te, ChemBERTa=cb_te_bag, **cb_test_perseed)

    # ---------- 6. Specialists ----------
    print("\n[tdc] Specialists (K=2 on MolFormer-emb) ...", flush=True)
    K = 2
    km = KMeans(n_clusters=K, random_state=SEED, n_init="auto")
    km.fit(embs[train_idx])
    feats = np.concatenate([embs, desc_block, fp_block], axis=1)
    train_cluster = km.predict(embs[train_idx])
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
            eval_metric="logloss", random_state=SEED, n_jobs=4, verbosity=0,
        )
        clf.fit(Xk, yk); clfs.append(clf)
    def soft_pred(arr_idx):
        d = np.linalg.norm(embs[arr_idx][:, None, :] - km.cluster_centers_[None, :, :], axis=2)
        w = np.exp(-d); w /= w.sum(axis=1, keepdims=True)
        probs = np.zeros((len(arr_idx), K))
        for k in range(K):
            if clfs[k] is not None:
                probs[:, k] = clfs[k].predict_proba(feats[arr_idx])[:, 1]
        return (probs * w).sum(axis=1)
    sp_va = soft_pred(val_idx); sp_te = soft_pred(test_idx)
    print(f"  Specialists test AUROC={roc_auc_score(y_te, sp_te):.4f}", flush=True)
    np.savez(res / "tdc_specialists_val_probs.npz", y_true=y_va, specialists_soft=sp_va)
    np.savez(res / "tdc_specialists_test_probs.npz", y_true=y_te, specialists_soft=sp_te)

    # ---------- 7. Stack ----------
    print("\n[tdc] FINAL STACK ====================", flush=True)
    parts_v, parts_t, cols = [], [], []
    for name in GNN_NAMES:
        parts_v.append(gnn_val[name]); parts_t.append(gnn_test[name]); cols.append(name)
    parts_v.append(mf_va); parts_t.append(mf_te); cols.append("molformer")
    parts_v.append(cb_va_bag); parts_t.append(cb_te_bag); cols.append("chemberta")
    parts_v.append(hy_va); parts_t.append(hy_te); cols.append("hybrid")
    parts_v.append(sp_va); parts_t.append(sp_te); cols.append("specialists")
    V = np.stack(parts_v, axis=1); T = np.stack(parts_t, axis=1)

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
        print(f"  {c:<18}  AUROC={m['AUROC']:.4f}  ACC={m['ACC']:.4f}  "
              f"F1={m['F1']:.4f}  MCC={m['MCC']:.4f}", flush=True)
        rows.append({"variant": "single", "method": c, "tuned_for": "MCC",
                     "threshold": t, **m})

    # Ensembles
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
    cb_v_b = V[:, 6];  cb_t_b = T[:, 6]
    hy_v_b = V[:, 7];  hy_t_b = T[:, 7]
    sp_v_b = V[:, 8];  sp_t_b = T[:, 8]

    for w_gnn, w_cb, w_hy, w_mf, name in [
        (0.25, 0.25, 0.25, 0.25, "equal4"),
        (0.10, 0.30, 0.30, 0.30, "gnn-light"),
        (0.00, 0.35, 0.35, 0.30, "no-gnn"),
        (0.00, 0.30, 0.40, 0.30, "no-gnn-hy-heavy"),
    ]:
        p_v = w_gnn*gn_v + w_cb*cb_v_b + w_hy*hy_v_b + w_mf*mf_v_b
        p_t = w_gnn*gn_t + w_cb*cb_t_b + w_hy*hy_t_b + w_mf*mf_t_b
        add("4-way", f"weighted-{name}", p_v, p_t)

    for w_gnn, w_cb, w_hy, w_mf, w_sp, name in [
        (0.20, 0.20, 0.20, 0.20, 0.20, "equal5"),
        (0.10, 0.25, 0.25, 0.25, 0.15, "gnn-light5"),
        (0.00, 0.30, 0.30, 0.25, 0.15, "no-gnn5"),
        (0.00, 0.25, 0.30, 0.25, 0.20, "no-gnn5-sp-heavy"),
        (0.05, 0.25, 0.30, 0.30, 0.10, "mf-heavy5"),
    ]:
        p_v = w_gnn*gn_v + w_cb*cb_v_b + w_hy*hy_v_b + w_mf*mf_v_b + w_sp*sp_v_b
        p_t = w_gnn*gn_t + w_cb*cb_t_b + w_hy*hy_t_b + w_mf*mf_t_b + w_sp*sp_t_b
        add("5-way", f"weighted-{name}", p_v, p_t)

    meta = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=42)
    meta.fit(V, y_va)
    add("meta", "logreg-C1", meta.predict_proba(V)[:, 1], meta.predict_proba(T)[:, 1])
    meta2 = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced", random_state=42)
    meta2.fit(V, y_va)
    add("meta", "logreg-C0.1", meta2.predict_proba(V)[:, 1], meta2.predict_proba(T)[:, 1])

    df = pd.DataFrame(rows)
    df.to_csv(res / "tdc_stack_metrics.csv", index=False)
    print("\n[TOP 15 by AUROC]")
    print(df.sort_values("AUROC", ascending=False).head(15).to_string(index=False))
    print("\n[TOP 10 by ACC]")
    print(df.sort_values("ACC", ascending=False).head(10).to_string(index=False))
    print("\n[TOP 10 by MCC]")
    print(df.sort_values("MCC", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
