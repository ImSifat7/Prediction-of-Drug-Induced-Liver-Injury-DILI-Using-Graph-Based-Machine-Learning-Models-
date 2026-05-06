"""Training loop with imbalance handling, early stopping, LR scheduling, deterministic seeding.

Key correctness points (no data leakage):
- pos_weight for BCEWithLogitsLoss is computed from the *training* indices only.
- Early stopping uses validation AUROC; the test set is never seen during training.
- All randomness is seeded so runs are reproducible.
"""

from __future__ import annotations

import copy
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_pos_weight(graphs: List[Data], indices: List[int]) -> float:
    labels = [int(graphs[i].y.item()) for i in indices]
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0:
        return 1.0
    return float(neg) / float(pos)


def make_loader(graphs: List[Data], indices: List[int], batch_size: int, shuffle: bool) -> DataLoader:
    subset = [graphs[i] for i in indices]
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)


def get_optimizer(model: nn.Module, name: str, lr: float, weight_decay: float):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer: {name}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device) -> dict:
    model.eval()
    y_true: list[float] = []
    y_prob: list[float] = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        prob = torch.sigmoid(out).detach().cpu().numpy().tolist()
        y_prob.extend(prob)
        y_true.extend(batch.y.view(-1).detach().cpu().numpy().tolist())
    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]

    if len(set(y_true)) < 2:
        auroc = float("nan")
    else:
        auroc = roc_auc_score(y_true, y_prob)

    return {
        "AUROC": auroc,
        "ACC": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "y_true": y_true,
        "y_prob": y_prob,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    lr: float,
    weight_decay: float,
    optimizer_name: str,
    pos_weight: float,
    epochs: int = 150,
    patience: int = 20,
    device="cpu",
    verbose: bool = False,
):
    """Train with class-weighted BCE, ReduceLROnPlateau, early stopping on val AUROC."""
    model = model.to(device)
    optim = get_optimizer(model, optimizer_name, lr, weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="max", factor=0.5, patience=8)
    pw = torch.tensor([pos_weight], device=device, dtype=torch.float)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)

    best_state = None
    best_auc = -1.0
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optim.zero_grad()
            out = model(batch)
            loss = crit(out, batch.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()
            total += loss.item()
        train_loss = total / max(len(train_loader), 1)

        val = evaluate(model, val_loader, device)
        auc = val["AUROC"] if val["AUROC"] == val["AUROC"] else -1.0  # NaN guard
        sched.step(auc)

        if auc > best_auc:
            best_auc = auc
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1

        if verbose:
            print(f"  ep{ep:03d} loss={train_loss:.4f} val_auc={auc:.4f} best={best_auc:.4f}")

        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc
