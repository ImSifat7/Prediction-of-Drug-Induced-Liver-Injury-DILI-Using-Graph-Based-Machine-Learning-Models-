"""5-fold scaffold-grouped CV on TDC DILI.

Per fold, retrains every base model and re-stacks. Reports mean ± std across
folds for AUROC / ACC / F1 / MCC. This is the standard cross-validated
evaluation required for publication.

Per-fold base models:
  - 5 GNNs (GCN/GAT/SAGE/GIN/MPNN) x 2 seeds each
  - MolFormer-XL frozen embeddings + bagged-3 XGBoost
  - AttentiveFP x 2 seeds
  - Hybrid GNN-emb + desc + Morgan-FP -> XGBoost (bagged-3)
  - ChemBERTa-77M fine-tune x 2 seeds
  - Specialists (K=2 K-means + XGBoost)
  - Final stack: 9-way mean + geom-mean + rank-avg + XGBoost meta

Run:
    python -m src.improved.tdc_cv5
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, roc_auc_score,
)
from torch_geometric.loader import DataLoader as PyGLoader
from torch_geometric.nn import AttentiveFP
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from xgboost import XGBClassifier

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.improved.data_utils import smiles_to_data, standardize_smiles
    from src.improved.models import build_model
    from src.improved.train_utils import (
        compute_pos_weight, make_loader, set_seed, train_model, evaluate,
    )
else:
    from .data_utils import smiles_to_data, standardize_smiles
    from .models import build_model
    from .train_utils import (
        compute_pos_weight, make_loader, set_seed, train_model, evaluate,
    )

RDLogger.DisableLog("rdApp.*")

GNN_NAMES = ["GCN", "GAT", "GraphSAGE", "GIN", "MPNN"]
GNN_SEEDS = 2
AFP_SEEDS = 2
CB_SEEDS = 2
EPOCHS_GNN = 60
EPOCHS_AFP = 70
EPOCHS_CB = 15
PATIENCE = 10
FP_BITS = 2048
N_BAG_XGB = 3
N_FOLDS = 5


def murcko(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    if m is None: return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
    except Exception:
        return ""


def scaffold_kfold(graphs, n_splits=5, seed=42):
    """Group molecules by Bemis-Murcko scaffold; deal scaffold-groups into K balanced folds."""
    import random as _r
    rng = _r.Random(seed)
    scafs: Dict[str, List[int]] = {}
    for i, g in enumerate(graphs):
        s = murcko(g.smiles)
        scafs.setdefault(s, []).append(i)
    groups = sorted(scafs.values(), key=lambda xs: (-len(xs), rng.random()))
    fold_sizes = [0] * n_splits
    fold_assignment = [[] for _ in range(n_splits)]
    for grp in groups:
        i = fold_sizes.index(min(fold_sizes))
        fold_assignment[i].extend(grp)
        fold_sizes[i] += len(grp)
    return fold_assignment


def morgan(smiles: str, n_bits: int = FP_BITS, radius: int = 2) -> np.ndarray:
    m = Chem.MolFromSmiles(smiles)
    if m is None: return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


@torch.no_grad()
def extract_pooled(model, loader, device):
    model.eval().to(device)
    captured = []
    first_linear = model.fc[0]
    def _hook(_m, inputs, _o):
        captured.append(inputs[0].detach().cpu().numpy())
    h = first_linear.register_forward_hook(_hook)
    try:
        for b in loader:
            b = b.to(device); _ = model(b)
    finally:
        h.remove()
    return np.concatenate(captured, axis=0)


def load_tdc():
    df = pd.read_csv("data/tdc_dili.csv")
    rows = []
    for _, r in df.iterrows():
        c = standardize_smiles(r["Drug"])
        if c is not None: rows.append((c, int(r["Y"])))
    seen, kept = set(), []
    for s, y in rows:
        if s not in seen:
            seen.add(s); kept.append((s, y))
    return [g for g in (smiles_to_data(s, y) for s, y in kept) if g is not None]


def extract_molformer_embeddings(smiles_list, device):
    """Extract MolFormer-XL embeddings for an arbitrary SMILES list (or reuse cache if same set)."""
    from src.improved.molformer import _patch_transformers_for_molformer
    _patch_transformers_for_molformer()
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
    mfm = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct",
                                    deterministic_eval=True, trust_remote_code=True).to(device)
    mfm.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(smiles_list), 32):
            chunk = smiles_list[i:i+32]
            enc = tok(chunk, padding=True, return_tensors="pt").to(device)
            o = mfm(**enc)
            pooled = o.pooler_output if hasattr(o, "pooler_output") and o.pooler_output is not None \
                     else o.last_hidden_state.mean(dim=1)
            out.append(pooled.detach().cpu().numpy())
    del mfm
    if device.type == "cuda": torch.cuda.empty_cache()
    return np.concatenate(out, axis=0)


def train_afp(seed, graphs, tr, va, te, in_dim, edge_dim, device, epochs=EPOCHS_AFP):
    set_seed(seed)
    afp = AttentiveFP(in_channels=in_dim, hidden_channels=128, out_channels=1,
                      edge_dim=edge_dim, num_layers=3, num_timesteps=3, dropout=0.2).to(device)
    pos_w = torch.tensor([compute_pos_weight(graphs, tr)], dtype=torch.float32, device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(afp.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    trl = PyGLoader([graphs[i] for i in tr], batch_size=64, shuffle=True)
    val = PyGLoader([graphs[i] for i in va], batch_size=64, shuffle=False)
    tel = PyGLoader([graphs[i] for i in te], batch_size=64, shuffle=False)
    def predict(loader):
        afp.eval()
        ys, ps = [], []
        with torch.no_grad():
            for b in loader:
                b = b.to(device)
                ps.extend(torch.sigmoid(afp.forward(b.x, b.edge_index, b.edge_attr, b.batch).view(-1)).cpu().numpy().tolist())
                ys.extend(b.y.view(-1).cpu().numpy().tolist())
        return np.array(ys), np.array(ps)
    best_val = -1.0; best_p_va = None; best_p_te = None; bad = 0
    for ep in range(epochs):
        afp.train()
        for b in trl:
            b = b.to(device)
            logits = afp.forward(b.x, b.edge_index, b.edge_attr, b.batch).view(-1)
            loss = bce(logits, b.y.view(-1).float())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(afp.parameters(), 1.0)
            opt.step()
        sched.step()
        y_va, p_va = predict(val); y_te, p_te = predict(tel)
        va_auc = roc_auc_score(y_va, p_va)
        if va_auc > best_val:
            best_val = va_auc; best_p_va = p_va.copy(); best_p_te = p_te.copy(); bad = 0
        else:
            bad += 1
        if bad >= PATIENCE: break
    return best_val, best_p_va, best_p_te


def train_cb(seed, smiles, y, tr, va, te, device, tok, spw, epochs=EPOCHS_CB):
    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        "DeepChem/ChemBERTa-77M-MTR", num_labels=1, ignore_mismatched_sizes=True,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    pos_w = torch.tensor([spw], dtype=torch.float32, device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    def enc(idx):
        t = tok([smiles[i] for i in idx], padding=True, truncation=True, max_length=200,
                return_tensors="pt")
        return t.input_ids, t.attention_mask
    tr_ids, tr_mask = enc(tr); va_ids, va_mask = enc(va); te_ids, te_mask = enc(te)
    tr_y = torch.tensor(y[tr], dtype=torch.float32)
    ds = TensorDataset(tr_ids, tr_mask, tr_y)
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    best_val = -1.0; best_p_va = None; best_p_te = None; bad = 0
    for ep in range(epochs):
        model.train()
        for ids, mask, yy in loader:
            ids, mask, yy = ids.to(device), mask.to(device), yy.to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits.view(-1)
            loss = bce(logits, yy)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            p_va = torch.sigmoid(model(input_ids=va_ids.to(device), attention_mask=va_mask.to(device)).logits.view(-1)).cpu().numpy()
            p_te = torch.sigmoid(model(input_ids=te_ids.to(device), attention_mask=te_mask.to(device)).logits.view(-1)).cpu().numpy()
        va_auc = roc_auc_score(y[va], p_va)
        if va_auc > best_val:
            best_val = va_auc; best_p_va = p_va.copy(); best_p_te = p_te.copy(); bad = 0
        else:
            bad += 1
        if bad >= 4: break
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    return best_val, best_p_va, best_p_te


def _best_thr(y, p):
    bt, bv = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        v = matthews_corrcoef(y, (p >= t).astype(int))
        if v > bv: bv, bt = v, float(t)
    return bt


def _m(y, p, t):
    yp = (p >= t).astype(int)
    return {
        "AUROC": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
        "ACC": float(accuracy_score(y, yp)),
        "F1": float(f1_score(y, yp, zero_division=0)),
        "MCC": float(matthews_corrcoef(y, yp)),
    }


def run_fold(fold_id, train_idx, val_idx, test_idx, graphs, smiles, labels,
             desc_block, fp_block, molformer_embs, device):
    print(f"\n========== FOLD {fold_id+1}/{N_FOLDS}  tr/va/te={len(train_idx)}/{len(val_idx)}/{len(test_idx)} ==========",
          flush=True)
    y_tr = labels[train_idx]; y_va = labels[val_idx]; y_te = labels[test_idx]
    in_dim = graphs[0].x.shape[1]; edge_dim = graphs[0].edge_attr.shape[1]
    desc_dim = graphs[0].u.shape[1]
    pos = (y_tr == 1).sum(); neg = (y_tr == 0).sum()
    spw = float(neg) / max(int(pos), 1)
    base_dir = Path(__file__).resolve().parents[2]

    parts_v, parts_t, cols = [], [], []
    gnn_embs_for_hybrid = {}

    # GNNs
    for name in GNN_NAMES:
        cfg = json.loads((base_dir / "configs" / f"{name}_best.json").read_text())["params"]
        bs = cfg["batch_size"]
        pv = np.zeros(len(val_idx)); pt = np.zeros(len(test_idx))
        for sd in range(GNN_SEEDS):
            set_seed(42 + fold_id * 100 + sd)
            model = build_model(name, in_dim, edge_dim, cfg, desc_dim=desc_dim)
            trl = make_loader(graphs, train_idx, batch_size=bs, shuffle=True)
            val = make_loader(graphs, val_idx,   batch_size=bs, shuffle=False)
            tel = make_loader(graphs, test_idx,  batch_size=bs, shuffle=False)
            pos_w = compute_pos_weight(graphs, train_idx)
            t0 = time.time()
            model, _ = train_model(
                model, trl, val,
                lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                optimizer_name=cfg["optimizer_name"], pos_weight=pos_w,
                epochs=EPOCHS_GNN, patience=PATIENCE, device=device,
            )
            p_va = np.array(evaluate(model, val, device)["y_prob"])
            p_te = np.array(evaluate(model, tel, device)["y_prob"])
            pv += p_va; pt += p_te
            if name in {"GAT", "GIN", "GraphSAGE"} and sd == 0:
                all_loader = make_loader(graphs, list(range(len(graphs))), batch_size=64, shuffle=False)
                gnn_embs_for_hybrid[name] = extract_pooled(model, all_loader, device)
            print(f"  [F{fold_id+1}] {name} sd={sd} val_auc={roc_auc_score(y_va, p_va):.4f} "
                  f"test_auc={roc_auc_score(y_te, p_te):.4f} ({time.time()-t0:.0f}s)", flush=True)
        pv /= GNN_SEEDS; pt /= GNN_SEEDS
        parts_v.append(pv); parts_t.append(pt); cols.append(name)
        print(f"  [F{fold_id+1}] --> {name} bagged AUROC={roc_auc_score(y_te, pt):.4f}", flush=True)

    # MolFormer XGB
    X_tr = molformer_embs[train_idx]; X_va = molformer_embs[val_idx]; X_te = molformer_embs[test_idx]
    mf_v = np.zeros(len(val_idx)); mf_t = np.zeros(len(test_idx))
    for s in range(N_BAG_XGB):
        clf = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                            subsample=0.85, colsample_bytree=0.7, reg_lambda=1.0,
                            scale_pos_weight=spw, eval_metric="logloss",
                            random_state=s, n_jobs=4, verbosity=0)
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        mf_v += clf.predict_proba(X_va)[:, 1]
        mf_t += clf.predict_proba(X_te)[:, 1]
    mf_v /= N_BAG_XGB; mf_t /= N_BAG_XGB
    parts_v.append(mf_v); parts_t.append(mf_t); cols.append("molformer")
    print(f"  [F{fold_id+1}] MolFormer AUROC={roc_auc_score(y_te, mf_t):.4f}", flush=True)

    # AttentiveFP
    afp_v = np.zeros(len(val_idx)); afp_t = np.zeros(len(test_idx))
    for sd in range(AFP_SEEDS):
        seed = 42 + fold_id * 100 + sd
        bv, pv_, pt_ = train_afp(seed, graphs, train_idx, val_idx, test_idx, in_dim, edge_dim, device)
        afp_v += pv_; afp_t += pt_
        print(f"  [F{fold_id+1}] AttentiveFP sd={sd} val={bv:.4f} test={roc_auc_score(y_te, pt_):.4f}",
              flush=True)
    afp_v /= AFP_SEEDS; afp_t /= AFP_SEEDS
    parts_v.append(afp_v); parts_t.append(afp_t); cols.append("AttentiveFP")
    print(f"  [F{fold_id+1}] AttentiveFP bagged AUROC={roc_auc_score(y_te, afp_t):.4f}", flush=True)

    # Hybrid XGB
    hy_emb = np.concatenate([gnn_embs_for_hybrid[a] for a in ("GAT", "GIN", "GraphSAGE")], axis=1)
    X_hy = np.concatenate([hy_emb, desc_block, fp_block], axis=1)
    Xh_tr = X_hy[train_idx]; Xh_va = X_hy[val_idx]; Xh_te = X_hy[test_idx]
    hy_v = np.zeros(len(val_idx)); hy_t = np.zeros(len(test_idx))
    for s in range(N_BAG_XGB):
        clf = XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05,
                            subsample=0.85, colsample_bytree=0.5, reg_lambda=1.0,
                            scale_pos_weight=spw, eval_metric="logloss",
                            random_state=s, n_jobs=4, verbosity=0)
        clf.fit(Xh_tr, y_tr, eval_set=[(Xh_va, y_va)], verbose=False)
        hy_v += clf.predict_proba(Xh_va)[:, 1]
        hy_t += clf.predict_proba(Xh_te)[:, 1]
    hy_v /= N_BAG_XGB; hy_t /= N_BAG_XGB
    parts_v.append(hy_v); parts_t.append(hy_t); cols.append("hybrid")
    print(f"  [F{fold_id+1}] Hybrid AUROC={roc_auc_score(y_te, hy_t):.4f}", flush=True)

    # Specialists
    K = 2
    km = KMeans(n_clusters=K, random_state=42, n_init="auto")
    km.fit(molformer_embs[train_idx])
    feats = np.concatenate([molformer_embs, desc_block, fp_block], axis=1)
    train_cluster = km.predict(molformer_embs[train_idx])
    clfs = []
    for k in range(K):
        mask = (train_cluster == k)
        if mask.sum() < 10: clfs.append(None); continue
        Xk = feats[train_idx][mask]; yk = y_tr[mask]
        pk = (yk == 1).sum(); nk = (yk == 0).sum()
        spw_k = float(nk) / max(int(pk), 1)
        c = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                          subsample=0.85, colsample_bytree=0.7, reg_lambda=1.0,
                          scale_pos_weight=spw_k, eval_metric="logloss",
                          random_state=42, n_jobs=4, verbosity=0)
        c.fit(Xk, yk); clfs.append(c)
    def soft(idx):
        d = np.linalg.norm(molformer_embs[idx][:, None, :] - km.cluster_centers_[None, :, :], axis=2)
        w = np.exp(-d); w /= w.sum(axis=1, keepdims=True)
        probs = np.zeros((len(idx), K))
        for k in range(K):
            if clfs[k] is not None:
                probs[:, k] = clfs[k].predict_proba(feats[idx])[:, 1]
        return (probs * w).sum(axis=1)
    sp_v = soft(val_idx); sp_t = soft(test_idx)
    parts_v.append(sp_v); parts_t.append(sp_t); cols.append("specialists")
    print(f"  [F{fold_id+1}] Specialists AUROC={roc_auc_score(y_te, sp_t):.4f}", flush=True)

    # ChemBERTa
    tok = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
    cb_v = np.zeros(len(val_idx)); cb_t = np.zeros(len(test_idx))
    for sd in range(CB_SEEDS):
        seed = 42 + fold_id * 100 + sd
        t0 = time.time()
        bv, pv_, pt_ = train_cb(seed, smiles, labels, train_idx, val_idx, test_idx, device, tok, spw)
        cb_v += pv_; cb_t += pt_
        print(f"  [F{fold_id+1}] CB sd={sd} val={bv:.4f} test={roc_auc_score(y_te, pt_):.4f} ({time.time()-t0:.0f}s)",
              flush=True)
    cb_v /= CB_SEEDS; cb_t /= CB_SEEDS
    parts_v.append(cb_v); parts_t.append(cb_t); cols.append("chemberta")
    print(f"  [F{fold_id+1}] ChemBERTa AUROC={roc_auc_score(y_te, cb_t):.4f}", flush=True)

    V = np.stack(parts_v, axis=1); T = np.stack(parts_t, axis=1)

    # ensemble = full mean + geom-mean + rank-avg + XGBoost meta
    eps = 1e-6
    p_mean_v = V.mean(axis=1); p_mean_t = T.mean(axis=1)
    p_geom_v = np.exp(np.log(np.clip(V, eps, 1 - eps)).mean(axis=1))
    p_geom_t = np.exp(np.log(np.clip(T, eps, 1 - eps)).mean(axis=1))
    def rank(M):
        return np.stack([np.argsort(np.argsort(M[:, j])) / max(len(M) - 1, 1)
                         for j in range(M.shape[1])], axis=1)
    p_rank_v = rank(V).mean(axis=1); p_rank_t = rank(T).mean(axis=1)

    # XGBoost meta (bagged)
    meta_v = np.zeros(len(val_idx)); meta_t = np.zeros(len(test_idx))
    for s in range(5):
        m_ = XGBClassifier(n_estimators=150, max_depth=2, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8,
                           reg_lambda=1.5, scale_pos_weight=spw,
                           eval_metric="logloss", random_state=s, n_jobs=4, verbosity=0)
        m_.fit(V, y_va)
        meta_v += m_.predict_proba(V)[:, 1]
        meta_t += m_.predict_proba(T)[:, 1]
    meta_v /= 5; meta_t /= 5

    ensembles = {
        "mean": (p_mean_v, p_mean_t),
        "geom-mean": (p_geom_v, p_geom_t),
        "rank-avg": (p_rank_v, p_rank_t),
        "meta-XGB": (meta_v, meta_t),
    }
    fold_results = {}
    for ename, (pv, pt) in ensembles.items():
        t = _best_thr(y_va, pv)
        m = _m(y_te, pt, t)
        fold_results[ename] = m
        print(f"  [F{fold_id+1}] {ename:<10}  AUROC={m['AUROC']:.4f}  ACC={m['ACC']:.4f}  "
              f"F1={m['F1']:.4f}  MCC={m['MCC']:.4f}", flush=True)
    # also single-model best on this fold
    for j, c in enumerate(cols):
        t = _best_thr(y_va, V[:, j])
        m = _m(y_te, T[:, j], t)
        fold_results[c] = m
    return fold_results, cols, V, T, y_va, y_te


def main():
    base = Path(__file__).resolve().parents[2]
    res = base / "results"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cv5] device={device}", flush=True)

    graphs = load_tdc()
    smiles = [g.smiles for g in graphs]
    labels = np.array([int(g.y.item()) for g in graphs], dtype=np.int64)
    n = len(graphs)
    print(f"[cv5] loaded {n} graphs", flush=True)

    desc_block = np.stack([g.u.cpu().numpy().reshape(-1) for g in graphs], axis=0)
    fp_block = np.stack([morgan(s) for s in smiles], axis=0)

    # MolFormer embeddings (use cache if available)
    cache = res / "tdc_molformer_embeddings.npy"
    if cache.exists():
        molformer_embs = np.load(cache)
        if molformer_embs.shape[0] != n:
            molformer_embs = extract_molformer_embeddings(smiles, device)
            np.save(cache, molformer_embs)
    else:
        molformer_embs = extract_molformer_embeddings(smiles, device)
        np.save(cache, molformer_embs)
    print(f"[cv5] MolFormer embeddings shape={molformer_embs.shape}", flush=True)

    # 5-fold scaffold CV: deal scaffold groups into 5 balanced folds
    folds = scaffold_kfold(graphs, n_splits=N_FOLDS, seed=42)
    fold_sizes = [len(f) for f in folds]
    print(f"[cv5] fold sizes: {fold_sizes}  sum={sum(fold_sizes)} (vs n={n})", flush=True)

    all_fold_results = []
    for fid in range(N_FOLDS):
        # fold fid is the TEST fold; (fid+1)%5 is the VAL fold; rest is TRAIN
        test_idx = folds[fid]
        val_idx = folds[(fid + 1) % N_FOLDS]
        train_idx = sum([folds[k] for k in range(N_FOLDS) if k != fid and k != (fid + 1) % N_FOLDS], [])
        results, cols, V, T, y_va, y_te = run_fold(
            fid, train_idx, val_idx, test_idx, graphs, smiles, labels,
            desc_block, fp_block, molformer_embs, device,
        )
        all_fold_results.append(results)
        # incremental save
        partial = {"fold": fid + 1, **{f"{name}__{m}": results[name][m]
                                       for name in results for m in ("AUROC", "ACC", "F1", "MCC")}}
        df_partial = pd.DataFrame(all_fold_results)
        df_partial.to_json(res / "tdc_cv5_partial.json", orient="records")
        print(f"\n[cv5] fold {fid+1} done; saved partial results to tdc_cv5_partial.json\n", flush=True)

    # Aggregate
    print("\n========== 5-FOLD CV AGGREGATE ==========")
    keys = list(all_fold_results[0].keys())
    agg = {}
    for k in keys:
        for met in ("AUROC", "ACC", "F1", "MCC"):
            vals = [fr[k][met] for fr in all_fold_results]
            agg.setdefault(k, {})[f"{met}_mean"] = float(np.mean(vals))
            agg[k][f"{met}_std"] = float(np.std(vals))
    df = pd.DataFrame(agg).T.reset_index().rename(columns={"index": "method"})
    df.to_csv(res / "tdc_cv5_metrics.csv", index=False)
    print(df.to_string(index=False))
    # winners
    print("\n[BEST mean AUROC]")
    print(df.sort_values("AUROC_mean", ascending=False).head(8).to_string(index=False))
    print("\n[BEST mean ACC]")
    print(df.sort_values("ACC_mean", ascending=False).head(8).to_string(index=False))
    print("\n[BEST mean MCC]")
    print(df.sort_values("MCC_mean", ascending=False).head(8).to_string(index=False))


if __name__ == "__main__":
    main()
