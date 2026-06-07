"""AttentiveFP on TDC DILI — the canonical published-SOTA model on this benchmark.

PyG provides AttentiveFP natively. Per the TDC leaderboard, this model
achieves ~0.886 AUROC on TDC DILI scaffold split. Adding it as a base model
to our stacking ensemble should push the headline up.

Run:
    python -m src.improved.tdc_attentivefp
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef
from torch_geometric.loader import DataLoader as PyGLoader
from torch_geometric.nn import AttentiveFP
from rdkit import RDLogger

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.improved.data_utils import scaffold_split, smiles_to_data, standardize_smiles
    from src.improved.train_utils import compute_pos_weight, set_seed
else:
    from .data_utils import scaffold_split, smiles_to_data, standardize_smiles
    from .train_utils import compute_pos_weight, set_seed

RDLogger.DisableLog("rdApp.*")


def load_tdc():
    df = pd.read_csv("data/tdc_dili.csv")
    rows = []
    for _, r in df.iterrows():
        c = standardize_smiles(r["Drug"])
        if c is not None:
            rows.append((c, int(r["Y"])))
    seen, kept = set(), []
    for s, y in rows:
        if s not in seen:
            seen.add(s); kept.append((s, y))
    return [g for g in (smiles_to_data(s, y) for s, y in kept) if g is not None]


class AFPClassifier(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden=128, layers=3, timesteps=3, dropout=0.2):
        super().__init__()
        self.afp = AttentiveFP(
            in_channels=in_dim, hidden_channels=hidden, out_channels=1,
            edge_dim=edge_dim, num_layers=layers, num_timesteps=timesteps, dropout=dropout,
        )

    def forward(self, data):
        return self.afp(data.x, data.edge_index, data.edge_attr, data.batch).view(-1)


def train_one(seed: int, graphs, train_idx, val_idx, test_idx, in_dim, edge_dim, device,
              epochs=100, patience=15, bs=64):
    set_seed(seed)
    model = AFPClassifier(in_dim, edge_dim, hidden=128, layers=3, timesteps=3, dropout=0.2).to(device)
    pos_w = torch.tensor([compute_pos_weight(graphs, train_idx)], dtype=torch.float32, device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    tr_loader = PyGLoader([graphs[i] for i in train_idx], batch_size=bs, shuffle=True)
    va_loader = PyGLoader([graphs[i] for i in val_idx],   batch_size=bs, shuffle=False)
    te_loader = PyGLoader([graphs[i] for i in test_idx],  batch_size=bs, shuffle=False)

    def predict(loader):
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for b in loader:
                b = b.to(device)
                ps.extend(torch.sigmoid(model(b)).cpu().numpy().tolist())
                ys.extend(b.y.view(-1).cpu().numpy().tolist())
        return np.array(ys), np.array(ps)

    best_val_auc = -1.0; best_p_va = None; best_p_te = None; bad = 0
    for ep in range(epochs):
        model.train()
        t0 = time.time(); ep_loss = 0; nb = 0
        for b in tr_loader:
            b = b.to(device)
            logits = model(b)
            loss = bce(logits, b.y.view(-1).float())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item(); nb += 1
        sched.step()
        y_va, p_va = predict(va_loader)
        y_te, p_te = predict(te_loader)
        va_auc = roc_auc_score(y_va, p_va)
        te_auc = roc_auc_score(y_te, p_te)
        if va_auc > best_val_auc:
            best_val_auc = va_auc; best_p_va = p_va.copy(); best_p_te = p_te.copy(); bad = 0
        else:
            bad += 1
        if (ep + 1) % 10 == 0 or bad >= patience:
            print(f"  sd={seed} ep{ep+1:03d} loss={ep_loss/nb:.4f} "
                  f"val={va_auc:.4f} test={te_auc:.4f} best_val={best_val_auc:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if bad >= patience:
            print(f"  sd={seed} early-stop ep{ep+1}", flush=True); break
    return best_val_auc, best_p_va, best_p_te


def main():
    base = Path(__file__).resolve().parents[2]
    res = base / "results"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[afp] device={device}", flush=True)

    graphs = load_tdc()
    train_idx, val_idx, test_idx = scaffold_split(graphs, 0.7, 0.1, 0.2, seed=42)
    y_va = np.array([int(graphs[i].y.item()) for i in val_idx])
    y_te = np.array([int(graphs[i].y.item()) for i in test_idx])
    in_dim = graphs[0].x.shape[1]; edge_dim = graphs[0].edge_attr.shape[1]
    print(f"[afp] n={len(graphs)} tr/va/te = {len(train_idx)}/{len(val_idx)}/{len(test_idx)}",
          flush=True)

    n_seeds = 3
    p_va_bag = np.zeros(len(val_idx))
    p_te_bag = np.zeros(len(test_idx))
    seed_preds_va, seed_preds_te = {}, {}
    for sd, seed in enumerate([42, 7, 13]):
        print(f"\n[afp] seed {seed} ({sd+1}/{n_seeds}) ...", flush=True)
        best_val, p_va, p_te = train_one(seed, graphs, train_idx, val_idx, test_idx,
                                          in_dim, edge_dim, device)
        p_va_bag += p_va; p_te_bag += p_te
        seed_preds_va[f"AttentiveFP_s{seed}"] = p_va
        seed_preds_te[f"AttentiveFP_s{seed}"] = p_te
        print(f"  --> seed {seed}: val_auc={best_val:.4f} test_auc={roc_auc_score(y_te, p_te):.4f}",
              flush=True)
    p_va_bag /= n_seeds; p_te_bag /= n_seeds
    print(f"\n[afp] MEAN-of-{n_seeds} test: AUROC={roc_auc_score(y_te, p_te_bag):.4f}", flush=True)

    np.savez(res / "tdc_attentivefp_val_probs.npz", y_true=y_va,
             AttentiveFP=p_va_bag, **seed_preds_va)
    np.savez(res / "tdc_attentivefp_test_probs.npz", y_true=y_te,
             AttentiveFP=p_te_bag, **seed_preds_te)
    print("[afp] wrote tdc_attentivefp_*.npz", flush=True)


if __name__ == "__main__":
    main()
