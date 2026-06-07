"""Fine-tune ChemBERTa-2 (DeepChem/ChemBERTa-77M-MTR) on DILI.

Uses the same scaffold split (seed=42, 60/20/20) as the GNN/classical pipelines so the
saved probabilities can be slotted directly into stack.py for the final ensemble.

Outputs:
  results/chemberta_test_probs.npz   — y_true + ChemBERTa test probs (mean over seeds)
  results/chemberta_val_probs.npz    — y_true + ChemBERTa val probs (mean over seeds)
  results/chemberta_metrics.csv      — per-seed val/test AUROC

Run:
    python -m src.improved.chemberta            # full run (3 seeds, ~3-5 h on CPU)
    python -m src.improved.chemberta --quick    # smoke test (1 seed, 2 epochs)
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.improved.data_utils import load_dataset, scaffold_split
else:
    from .data_utils import load_dataset, scaffold_split


MODEL_NAME = "DeepChem/ChemBERTa-77M-MTR"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 25
PATIENCE = 5
LR = 2e-5
WEIGHT_DECAY = 0.01
DEFAULT_SEEDS = [42, 7, 13]


class SmilesDataset(Dataset):
    """Dataset for ChemBERTa fine-tuning, optional on-the-fly SMILES augmentation.

    Why augment:
      - ChemBERTa sees the SMILES as a token sequence — the same molecule has
        many valid SMILES strings (random atom ordering). Different orderings
        give different token sequences, which is a free way to ~5–10x the
        effective training set on a small dataset like DILIrank (521 train rows).
      - Crucial: augment only in training mode. Val/test must use canonical
        SMILES so the test number is stable and reproducible.

    augment: probability of returning a random SMILES variant per __getitem__
             call. 0.0 = always canonical (eval/test). ~0.8 in training.
    """

    def __init__(self, smiles: List[str], labels: np.ndarray, tokenizer, max_len: int,
                 augment: float = 0.0):
        self.smiles = smiles
        self.labels = labels
        self.tok = tokenizer
        self.max_len = max_len
        self.augment = augment
        # Pre-parse to RDKit mols once. None for any invalid input.
        self._mols = [Chem.MolFromSmiles(s) for s in smiles]

    def __len__(self):
        return len(self.smiles)

    def _smiles_at(self, i: int) -> str:
        if self.augment <= 0.0:
            return self.smiles[i]
        mol = self._mols[i]
        if mol is None:
            return self.smiles[i]
        if np.random.random() > self.augment:
            return self.smiles[i]  # keep canonical (1-augment) of the time
        try:
            return Chem.MolToSmiles(mol, canonical=False, doRandom=True)
        except Exception:
            return self.smiles[i]

    def __getitem__(self, i: int):
        smi = self._smiles_at(i)
        enc = self.tok(
            smi,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(float(self.labels[i]), dtype=torch.float),
        }


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    y_true: List[float] = []
    y_prob: List[float] = []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        am = batch["attention_mask"].to(device)
        out = model(input_ids=ids, attention_mask=am).logits.squeeze(-1)
        prob = torch.sigmoid(out).detach().cpu().numpy().tolist()
        y_prob.extend(prob)
        y_true.extend(batch["label"].numpy().tolist())
    auroc = float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else float("nan")
    return {
        "AUROC": auroc,
        "y_true": np.asarray(y_true),
        "y_prob": np.asarray(y_prob),
    }


def train_one_seed(
    seed: int,
    smiles_tr: List[str], y_tr: np.ndarray,
    smiles_va: List[str], y_va: np.ndarray,
    smiles_te: List[str], y_te: np.ndarray,
    epochs: int,
    patience: int,
    device,
) -> Tuple[float, dict, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # num_labels=1 -> single-logit head, paired with BCEWithLogitsLoss for binary cls.
    # problem_type="regression" suppresses the cross-entropy path inside HF's forward.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=1, problem_type="regression"
    ).to(device)

    pos = float((y_tr == 1).sum())
    neg = float((y_tr == 0).sum())
    pos_w = torch.tensor([neg / max(pos, 1.0)], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    tr_ds = SmilesDataset(smiles_tr, y_tr, tokenizer, MAX_LEN, augment=0.8)
    va_ds = SmilesDataset(smiles_va, y_va, tokenizer, MAX_LEN, augment=0.0)
    te_ds = SmilesDataset(smiles_te, y_te, tokenizer, MAX_LEN, augment=0.0)
    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    te_dl = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    best_auc = -1.0
    best_state = None
    bad = 0
    for ep in range(1, epochs + 1):
        model.train()
        t_ep = time.time()
        total = 0.0
        for batch in tr_dl:
            ids = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            y = batch["label"].to(device)
            opt.zero_grad()
            logits = model(input_ids=ids, attention_mask=am).logits.squeeze(-1)
            loss = crit(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        v = evaluate(model, va_dl, device)
        print(
            f"  seed={seed} ep{ep:02d} loss={total/max(len(tr_dl),1):.4f} "
            f"val_auc={v['AUROC']:.4f} best={max(best_auc, v['AUROC']):.4f} "
            f"({time.time()-t_ep:.0f}s)",
            flush=True,
        )
        if v["AUROC"] > best_auc:
            best_auc = v["AUROC"]
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            print(f"  seed={seed} early stop at ep{ep}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    val_res = evaluate(model, va_dl, device)
    test_res = evaluate(model, te_dl, device)
    return float(best_auc), val_res, test_res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 seed, 2 epochs (smoke test)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    args = ap.parse_args()

    if args.quick:
        seeds = [42]
        epochs = 2
        patience = 5
    else:
        seeds = args.seeds if args.seeds else DEFAULT_SEEDS
        epochs = args.epochs
        patience = args.patience

    base = Path(__file__).resolve().parents[2]
    data_path = base / "data" / "dili_clean.csv"
    results_dir = base / "results"
    results_dir.mkdir(exist_ok=True)

    print(f"[chemberta] model={MODEL_NAME}  seeds={seeds}  epochs={epochs}", flush=True)

    print("[chemberta] loading data", flush=True)
    graphs = load_dataset(data_path)
    smiles_all = [g.smiles for g in graphs]
    labels = np.array([int(g.y.item()) for g in graphs], dtype=np.int64)
    train_idx, val_idx, test_idx = scaffold_split(graphs, 0.6, 0.2, 0.2, seed=42)
    print(f"[chemberta] split  tr={len(train_idx)} va={len(val_idx)} te={len(test_idx)}", flush=True)

    smiles_tr = [smiles_all[i] for i in train_idx]
    smiles_va = [smiles_all[i] for i in val_idx]
    smiles_te = [smiles_all[i] for i in test_idx]
    y_tr = labels[train_idx]
    y_va = labels[val_idx]
    y_te = labels[test_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[chemberta] device={device}", flush=True)

    val_probs_seeds: List[np.ndarray] = []
    test_probs_seeds: List[np.ndarray] = []
    metric_rows = []

    t_total = time.time()
    for seed in seeds:
        print(f"\n[chemberta] === seed {seed} ===", flush=True)
        t0 = time.time()
        best_val_auc, val_res, test_res = train_one_seed(
            seed,
            smiles_tr, y_tr, smiles_va, y_va, smiles_te, y_te,
            epochs=epochs, patience=patience, device=device,
        )
        elapsed = time.time() - t0
        print(
            f"[chemberta] seed={seed} best_val_auc={best_val_auc:.4f} "
            f"test_auc={test_res['AUROC']:.4f} elapsed={elapsed:.0f}s",
            flush=True,
        )
        val_probs_seeds.append(val_res["y_prob"])
        test_probs_seeds.append(test_res["y_prob"])
        metric_rows.append({
            "seed": seed,
            "val_AUROC": best_val_auc,
            "test_AUROC": float(test_res["AUROC"]),
            "elapsed_sec": float(elapsed),
        })

    val_probs_mean = np.mean(np.stack(val_probs_seeds), axis=0)
    test_probs_mean = np.mean(np.stack(test_probs_seeds), axis=0)

    # Save per-seed predictions as separate columns so the meta-learner can
    # treat each seed as an independent base model (greatly improves stacking
    # — XGBoost benefits from seeing the spread between seeds, not just the mean).
    seed_test_cols = {f"ChemBERTa_s{seed}": test_probs_seeds[i]
                      for i, seed in enumerate(seeds)}
    seed_val_cols = {f"ChemBERTa_s{seed}": val_probs_seeds[i]
                     for i, seed in enumerate(seeds)}

    np.savez(
        results_dir / "chemberta_test_probs.npz",
        y_true=y_te.astype(np.int64),
        ChemBERTa=test_probs_mean,
        **seed_test_cols,
    )
    np.savez(
        results_dir / "chemberta_val_probs.npz",
        y_true=y_va.astype(np.int64),
        ChemBERTa=val_probs_mean,
        **seed_val_cols,
    )
    pd.DataFrame(metric_rows).to_csv(results_dir / "chemberta_metrics.csv", index=False)

    yp = (test_probs_mean >= 0.5).astype(int)
    print(
        f"\n[chemberta] mean-of-{len(seeds)}  AUROC={roc_auc_score(y_te, test_probs_mean):.4f}  "
        f"ACC={accuracy_score(y_te, yp):.4f}  F1={f1_score(y_te, yp, zero_division=0):.4f}  "
        f"MCC={matthews_corrcoef(y_te, yp):.4f}",
        flush=True,
    )
    print(f"[chemberta] total elapsed {time.time()-t_total:.0f}s", flush=True)


if __name__ == "__main__":
    main()
