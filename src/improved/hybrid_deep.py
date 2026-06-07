"""Deeper Hybrid (Option F).

Combines EVERY embedding/feature block we have into one XGBoost classifier:

  [GAT_emb (128) ||
   GIN_emb (128) ||
   GraphSAGE_emb (128) ||
   MolFormer_emb (768)  ||      # frozen, but proven useful
   global descriptors (14) ||
   Brenk/PAINS/NIH/heteroaromatics already in the 14
   Morgan-FP-2048 ||
   Tox21 endpoint predictions (12)
  ] -> XGBoost classifier

The intuition: each block captures a different aspect of the molecule. XGBoost
learns which features to weight per-decision (non-linear feature interactions).

This is a "kitchen sink" approach — but on a small dataset it's *risky* (more
features = more overfit). We mitigate with:
  - n_estimators=500, max_depth=4 (shallow trees)
  - colsample_bytree=0.5 (random feature subsampling per tree)
  - subsample=0.8
  - scale_pos_weight handles class imbalance
  - 5-seed bagging across XGBoost random_state

Run:
    python -m src.improved.hybrid_deep
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
from rdkit import Chem, RDLogger
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

RDLogger.DisableLog("rdApp.*")

GNN_NAMES = ["GAT", "GIN", "GraphSAGE"]
SEED = 42
EPOCHS = 80
FP_BITS = 2048


@torch.no_grad()
def extract_pooled(model: torch.nn.Module, loader, device) -> np.ndarray:
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


def morgan(smiles: str, n_bits: int = FP_BITS, radius: int = 2) -> np.ndarray:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _best_thr(y, p, metric="MCC"):
    fns = {"MCC": matthews_corrcoef, "ACC": accuracy_score,
           "F1": lambda y, yp: f1_score(y, yp, zero_division=0)}
    fn = fns[metric]
    best_t, best_v = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        v = fn(y, (p >= t).astype(int))
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


def main():
    base = Path(__file__).resolve().parents[2]
    res = base / "results"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[hybrid_deep] device={device}", flush=True)

    print("[hybrid_deep] loading data + descriptors ...", flush=True)
    graphs = load_dataset(base / "data" / "dili_clean.csv")
    smiles = [g.smiles for g in graphs]
    labels = np.array([int(g.y.item()) for g in graphs], dtype=np.int64)
    train_idx, val_idx, test_idx = scaffold_split(graphs, 0.6, 0.2, 0.2, seed=42)
    y_train = labels[train_idx]; y_val = labels[val_idx]; y_test = labels[test_idx]

    in_dim = graphs[0].x.shape[1]
    edge_dim = graphs[0].edge_attr.shape[1]
    desc_dim = graphs[0].u.shape[1]

    # ----- Train each GNN once, extract embeddings -----
    embs_dict: Dict[str, np.ndarray] = {}
    for name in GNN_NAMES:
        print(f"\n[hybrid_deep] training {name} ...", flush=True)
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
        print(f"  trained in {time.time()-t0:.0f}s  best_val={best_val:.4f}", flush=True)
        all_loader = make_loader(graphs, list(range(len(graphs))), batch_size=64, shuffle=False)
        emb = extract_pooled(model, all_loader, device)
        embs_dict[name] = emb
        print(f"  {name} embedding shape: {emb.shape}", flush=True)

    # ----- Load other feature blocks -----
    print("\n[hybrid_deep] loading MolFormer embeddings (cached) ...", flush=True)
    molformer = np.load(res / "molformer_embeddings.npy")
    print(f"  MolFormer: {molformer.shape}", flush=True)

    print("[hybrid_deep] computing Morgan FPs ...", flush=True)
    fps = np.stack([morgan(s) for s in smiles], axis=0)
    print(f"  Morgan-FP-{FP_BITS}: {fps.shape}", flush=True)

    print("[hybrid_deep] gathering descriptors ...", flush=True)
    desc = np.stack([g.u.cpu().numpy().reshape(-1) for g in graphs], axis=0)
    print(f"  descriptors: {desc.shape}", flush=True)

    print("[hybrid_deep] loading Tox21 12-endpoint predictions ...", flush=True)
    tox = np.load(res / "tox21_all_preds.npy")
    print(f"  Tox21: {tox.shape}", flush=True)

    blocks = [embs_dict[n] for n in GNN_NAMES] + [molformer, desc, fps, tox]
    X_all = np.concatenate(blocks, axis=1)
    print(f"\n[hybrid_deep] FULL feature matrix: {X_all.shape}", flush=True)

    X_train = X_all[train_idx]; X_val = X_all[val_idx]; X_test = X_all[test_idx]

    pos = (y_train == 1).sum(); neg = (y_train == 0).sum()
    spw = float(neg) / max(int(pos), 1)
    print(f"\n[hybrid_deep] training 5-seed bagged XGBoost (max_depth=4, colsample=0.5) ...", flush=True)
    val_preds = np.zeros(len(y_val))
    test_preds = np.zeros(len(y_test))
    n_bag = 5
    for s in range(n_bag):
        clf = XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.5,
            reg_lambda=1.5, scale_pos_weight=spw,
            eval_metric="logloss", random_state=s, n_jobs=4, verbosity=0,
        )
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        val_preds += clf.predict_proba(X_val)[:, 1]
        test_preds += clf.predict_proba(X_test)[:, 1]
    val_preds /= n_bag; test_preds /= n_bag

    rows = []
    for tgt in ("MCC", "ACC", "F1"):
        t = _best_thr(y_val, val_preds, metric=tgt)
        m = _m(y_test, test_preds, t)
        rows.append({"target": tgt, "threshold": t, **m})
        print(f"  thr={t:.2f} (tuned-for-{tgt})  AUROC={m['AUROC']:.4f}  "
              f"ACC={m['ACC']:.4f}  F1={m['F1']:.4f}  MCC={m['MCC']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(res / "hybrid_deep_metrics.csv", index=False)

    np.savez(res / "hybrid_deep_test_probs.npz", y_true=y_test, hybrid_deep=test_preds)
    np.savez(res / "hybrid_deep_val_probs.npz", y_true=y_val, hybrid_deep=val_preds)
    print(f"\n[hybrid_deep] wrote results/hybrid_deep_*.npz + hybrid_deep_metrics.csv", flush=True)


if __name__ == "__main__":
    main()
