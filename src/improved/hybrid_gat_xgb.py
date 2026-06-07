"""Hybrid GAT/GIN -> XGBoost framework (advisor's plan, Phase 3 May 28).

Approach:
  1. Train each tuned GNN once (or load the best seed's weights) and extract the
     pre-FC pooled graph embedding for every molecule.
  2. Concatenate per-molecule:
        [GAT_emb || GIN_emb || GraphSAGE_emb || descriptors || Morgan FP-512]
     Total ~64 + 64 + 64 + 14 + 512 = ~720-dim "rich vector".
  3. Train XGBoost on this vector, scaffold-split val for early-stop / threshold.

Why this should beat single GNN + classical separately:
  - GNN graph emb captures *local* substructure context the FP misses
  - FP captures *count* of fragments the GNN may smear during pooling
  - Descriptors give global molecular properties
  - XGBoost handles non-linear feature interactions the linear FC head cannot

Run:
    python -m src.improved.hybrid_gat_xgb
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
from rdkit import Chem
from rdkit.Chem import AllChem
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
    from src.improved.models import build_model
    from src.improved.train_utils import (
        compute_pos_weight, make_loader, set_seed, train_model,
    )
else:
    from .data_utils import load_dataset, scaffold_split
    from .models import build_model
    from .train_utils import (
        compute_pos_weight, make_loader, set_seed, train_model,
    )


GNN_NAMES = ["GAT", "GIN", "GraphSAGE"]  # use top-3 GNNs as feature extractors
SEED = 42
EPOCHS = 80


@torch.no_grad()
def extract_pooled_embedding(model: torch.nn.Module, loader, device) -> np.ndarray:
    """Capture the input that the model's FC head receives, via a forward hook.

    That input is the (pooled graph embedding [+ descriptor-MLP output if fused])
    vector. We hook the first Linear layer inside model.fc and stash its input
    each batch, then concatenate over the dataset.
    """
    model.eval().to(device)
    captured: List[np.ndarray] = []
    first_linear = model.fc[0]  # nn.Sequential, first module is nn.Linear

    def _hook(_module, inputs, _output):
        x_in = inputs[0]
        captured.append(x_in.detach().cpu().numpy())

    h = first_linear.register_forward_hook(_hook)
    try:
        for batch in loader:
            batch = batch.to(device)
            _ = model(batch)
    finally:
        h.remove()
    return np.concatenate(captured, axis=0)


def morgan_fp(smiles: str, n_bits: int = 512, radius: int = 2) -> np.ndarray:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _metrics(y, p, t):
    yp = (p >= t).astype(int)
    return {
        "AUROC": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
        "ACC":   float(accuracy_score(y, yp)),
        "F1":    float(f1_score(y, yp, zero_division=0)),
        "MCC":   float(matthews_corrcoef(y, yp)),
    }


def _best_thr(y, p):
    best_t, best_v = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 81):
        yp = (p >= t).astype(int)
        v = matthews_corrcoef(y, yp)
        if v > best_v:
            best_v, best_t = float(v), float(t)
    return best_t


def main():
    base = Path(__file__).resolve().parents[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[hybrid] device={device}", flush=True)

    print("[hybrid] loading data + descriptors...", flush=True)
    graphs = load_dataset(base / "data" / "dili_clean.csv")
    in_dim = graphs[0].x.shape[1]
    edge_dim = graphs[0].edge_attr.shape[1]
    desc_dim = graphs[0].u.shape[1]
    train_idx, val_idx, test_idx = scaffold_split(graphs, 0.6, 0.2, 0.2, seed=42)
    y_train = np.array([int(graphs[i].y.item()) for i in train_idx])
    y_val = np.array([int(graphs[i].y.item()) for i in val_idx])
    y_test = np.array([int(graphs[i].y.item()) for i in test_idx])

    # Build per-molecule feature blocks.
    print("[hybrid] computing Morgan FP-512 ...", flush=True)
    fp_all = np.stack([morgan_fp(g.smiles) for g in graphs], axis=0)
    desc_all = np.stack([g.u.cpu().numpy().reshape(-1) for g in graphs], axis=0)

    embs_dict: Dict[str, np.ndarray] = {}
    for name in GNN_NAMES:
        print(f"\n[hybrid] training {name} for embedding extraction...", flush=True)
        cfg = json.loads((base / "configs" / f"{name}_best.json").read_text())["params"]
        set_seed(SEED)
        model = build_model(name, in_dim, edge_dim, cfg, desc_dim=desc_dim)
        tr_loader = make_loader(graphs, train_idx, batch_size=cfg["batch_size"], shuffle=True)
        va_loader = make_loader(graphs, val_idx, batch_size=cfg["batch_size"], shuffle=False)
        pos_w = compute_pos_weight(graphs, train_idx)
        t0 = time.time()
        model, best_val = train_model(
            model, tr_loader, va_loader,
            lr=cfg["lr"], weight_decay=cfg["weight_decay"],
            optimizer_name=cfg["optimizer_name"], pos_weight=pos_w,
            epochs=EPOCHS, patience=15, device=device,
        )
        print(f"  trained in {time.time()-t0:.0f}s  best_val_auc={best_val:.4f}", flush=True)
        # Extract embeddings for all molecules in fixed order, batch_size=64.
        all_loader = make_loader(graphs, list(range(len(graphs))), batch_size=64, shuffle=False)
        emb = extract_pooled_embedding(model, all_loader, device)
        embs_dict[name] = emb
        print(f"  {name} embedding shape: {emb.shape}", flush=True)

    # Build the combined feature matrix:
    #   [GAT_emb || GIN_emb || GraphSAGE_emb || descriptors || FP]
    blocks = [embs_dict[n] for n in GNN_NAMES] + [desc_all, fp_all]
    X_all = np.concatenate(blocks, axis=1)
    print(f"\n[hybrid] combined feature matrix: {X_all.shape}", flush=True)

    X_train = X_all[train_idx]
    X_val = X_all[val_idx]
    X_test = X_all[test_idx]

    rows = []

    print("\n[hybrid] training XGBoost...", flush=True)
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    spw = neg / max(pos, 1)
    clf = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, scale_pos_weight=spw,
        eval_metric="logloss", random_state=42, n_jobs=4, verbosity=0,
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    p_val = clf.predict_proba(X_val)[:, 1]
    p_test = clf.predict_proba(X_test)[:, 1]
    t = _best_thr(y_val, p_val)
    m = _metrics(y_test, p_test, t)
    print(f"  Hybrid-XGBoost  thr={t:.2f}  AUROC={m['AUROC']:.4f}  "
          f"ACC={m['ACC']:.4f}  F1={m['F1']:.4f}  MCC={m['MCC']:.4f}", flush=True)
    rows.append({"variant": "hybrid", "method": "GNNxN+desc+FP -> XGBoost",
                 "threshold": t, **m})

    # Ablation: remove FP block
    print("\n[hybrid] ablation: GNN_emb + descriptors (no FP) -> XGBoost", flush=True)
    blocks_no_fp = [embs_dict[n] for n in GNN_NAMES] + [desc_all]
    X_no_fp = np.concatenate(blocks_no_fp, axis=1)
    clf2 = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, scale_pos_weight=spw,
        eval_metric="logloss", random_state=42, n_jobs=4, verbosity=0,
    )
    clf2.fit(X_no_fp[train_idx], y_train, eval_set=[(X_no_fp[val_idx], y_val)], verbose=False)
    p_val2 = clf2.predict_proba(X_no_fp[val_idx])[:, 1]
    p_test2 = clf2.predict_proba(X_no_fp[test_idx])[:, 1]
    t2 = _best_thr(y_val, p_val2)
    m2 = _metrics(y_test, p_test2, t2)
    print(f"  Hybrid-XGBoost (no FP)  thr={t2:.2f}  AUROC={m2['AUROC']:.4f}  "
          f"ACC={m2['ACC']:.4f}  F1={m2['F1']:.4f}  MCC={m2['MCC']:.4f}", flush=True)
    rows.append({"variant": "hybrid", "method": "GNNxN+desc (no FP) -> XGBoost",
                 "threshold": t2, **m2})

    # Ablation: FP-only baseline (to confirm GNN emb adds value)
    print("\n[hybrid] ablation: FP-only -> XGBoost (reference baseline)", flush=True)
    clf3 = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, scale_pos_weight=spw,
        eval_metric="logloss", random_state=42, n_jobs=4, verbosity=0,
    )
    clf3.fit(fp_all[train_idx], y_train, eval_set=[(fp_all[val_idx], y_val)], verbose=False)
    p_val3 = clf3.predict_proba(fp_all[val_idx])[:, 1]
    p_test3 = clf3.predict_proba(fp_all[test_idx])[:, 1]
    t3 = _best_thr(y_val, p_val3)
    m3 = _metrics(y_test, p_test3, t3)
    print(f"  XGBoost on FP-only  thr={t3:.2f}  AUROC={m3['AUROC']:.4f}  "
          f"ACC={m3['ACC']:.4f}  F1={m3['F1']:.4f}  MCC={m3['MCC']:.4f}", flush=True)
    rows.append({"variant": "ablation", "method": "FP-only -> XGBoost",
                 "threshold": t3, **m3})

    out = base / "results" / "hybrid_metrics.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    np.savez(base / "results" / "hybrid_test_probs.npz",
             y_true=y_test, hybrid=p_test, hybrid_no_fp=p_test2, fp_only=p_test3)
    np.savez(base / "results" / "hybrid_val_probs.npz",
             y_true=y_val, hybrid=p_val, hybrid_no_fp=p_val2, fp_only=p_val3)
    print(f"\n[hybrid] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
