"""Fine-tune MolFormer-XL on DILIrank (Option E).

What this does vs molformer.py (which just extracts frozen embeddings):
  - Loads the MolFormer-XL transformer with shims for transformers 5.x.
  - Adds a classification head (linear -> 1 logit).
  - Fine-tunes BOTH the transformer backbone (low LR) AND the head (higher LR)
    on the 521-molecule DILIrank training set.
  - Uses weighted BCE loss for class imbalance + gradient clipping + early stop
    on val AUROC.
  - Saves per-seed test/val predictions for the stack.

Why this can help vs frozen embeddings:
  - The pretrained MolFormer captures generic chemistry. Fine-tuning adapts it
    specifically to the DILI prediction task — the encoder learns to weight
    DILI-relevant motifs higher.

Risk:
  - 47M parameters on 521 train samples is a serious overfit risk.
  - We freeze the bottom 6 transformer layers (only fine-tune the top 6 + head)
    to control this. We also use low LR (2e-5) and aggressive early stopping.

Run:
    python -m src.improved.molformer_finetune --seeds 42 7 13 --epochs 20
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
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch.utils.data import Dataset, DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.improved.data_utils import load_dataset, scaffold_split
    from src.improved.molformer import _patch_transformers_for_molformer, MODEL_NAME
else:
    from .data_utils import load_dataset, scaffold_split
    from .molformer import _patch_transformers_for_molformer, MODEL_NAME


class MolFormerWithHead(nn.Module):
    """MolFormer-XL encoder + classification head."""

    def __init__(self, encoder, hidden_dim: int = 768, dropout: float = 0.2):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state  # [B, T, H]
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = self.dropout(pooled)
        return self.head(pooled).squeeze(-1)


class SmilesDataset(Dataset):
    def __init__(self, smiles, labels, tokenizer, max_len: int = 202):
        self.smiles = smiles
        self.labels = labels
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, i):
        enc = self.tok(
            self.smiles[i],
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
def evaluate(model, loader, device):
    model.eval()
    y_true, y_prob = [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        am = batch["attention_mask"].to(device)
        logits = model(ids, am)
        y_prob.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        y_true.extend(batch["label"].numpy().tolist())
    return {
        "AUROC": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
        "y_true": np.asarray(y_true),
        "y_prob": np.asarray(y_prob),
    }


def freeze_lower_layers(encoder, n_keep_trainable: int = 6):
    """Freeze all but the top N transformer blocks + final layer-norm.
    This controls overfit on our tiny 521-sample training set."""
    # Freeze embeddings
    for p in encoder.embeddings.parameters():
        p.requires_grad = False
    # Freeze lower transformer blocks
    layers = encoder.encoder.layer
    n_total = len(layers)
    n_freeze = max(0, n_total - n_keep_trainable)
    for i in range(n_freeze):
        for p in layers[i].parameters():
            p.requires_grad = False
    n_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    n_total_params = sum(p.numel() for p in encoder.parameters())
    print(f"  encoder: {n_freeze}/{n_total} layers frozen  "
          f"trainable={n_trainable/1e6:.1f}M / total={n_total_params/1e6:.1f}M", flush=True)


def train_one_seed(seed, smiles_tr, y_tr, smiles_va, y_va, smiles_te, y_te,
                   epochs, patience, device, batch_size=8, lr_encoder=2e-5, lr_head=1e-3,
                   n_keep_trainable=6, max_len=202):
    _patch_transformers_for_molformer()
    from transformers import AutoTokenizer, AutoModel

    torch.manual_seed(seed)
    np.random.seed(seed)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    encoder = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, deterministic_eval=True)
    freeze_lower_layers(encoder, n_keep_trainable=n_keep_trainable)

    model = MolFormerWithHead(encoder).to(device)

    pos = float((y_tr == 1).sum()); neg = float((y_tr == 0).sum())
    pos_w = torch.tensor([neg / max(pos, 1.0)], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    enc_params = [p for n, p in model.encoder.named_parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if "head" in n]
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": lr_encoder},
         {"params": head_params, "lr": lr_head}],
        weight_decay=0.01,
    )

    tr_dl = DataLoader(SmilesDataset(smiles_tr, y_tr, tok, max_len), batch_size=batch_size, shuffle=True)
    va_dl = DataLoader(SmilesDataset(smiles_va, y_va, tok, max_len), batch_size=batch_size, shuffle=False)
    te_dl = DataLoader(SmilesDataset(smiles_te, y_te, tok, max_len), batch_size=batch_size, shuffle=False)

    best_auc, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time(); total = 0.0
        for batch in tr_dl:
            ids = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            y = batch["label"].to(device)
            opt.zero_grad()
            logits = model(ids, am)
            loss = crit(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        v = evaluate(model, va_dl, device)
        print(f"  seed={seed} ep{ep:02d} loss={total/max(len(tr_dl),1):.4f} "
              f"val_auc={v['AUROC']:.4f} best={max(best_auc, v['AUROC']):.4f} ({time.time()-t0:.0f}s)",
              flush=True)
        if v["AUROC"] > best_auc:
            best_auc = v["AUROC"]
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            print(f"  seed={seed} early-stop at ep{ep}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    val_res = evaluate(model, va_dl, device)
    test_res = evaluate(model, te_dl, device)
    return float(best_auc), val_res, test_res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 13])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr-encoder", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--keep-trainable", type=int, default=6)
    ap.add_argument("--max-len", type=int, default=202)
    args = ap.parse_args()

    base = Path(__file__).resolve().parents[2]
    res = base / "results"
    res.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mf-finetune] device={device}  seeds={args.seeds}  epochs={args.epochs}  "
          f"keep_top={args.keep_trainable}", flush=True)

    graphs = load_dataset(base / "data" / "dili_clean.csv")
    smiles_all = [g.smiles for g in graphs]
    labels = np.array([int(g.y.item()) for g in graphs], dtype=np.int64)
    train_idx, val_idx, test_idx = scaffold_split(graphs, 0.6, 0.2, 0.2, seed=42)
    smiles_tr = [smiles_all[i] for i in train_idx]
    smiles_va = [smiles_all[i] for i in val_idx]
    smiles_te = [smiles_all[i] for i in test_idx]
    y_tr = labels[train_idx]; y_va = labels[val_idx]; y_te = labels[test_idx]
    print(f"[mf-finetune] split tr={len(train_idx)} va={len(val_idx)} te={len(test_idx)}", flush=True)

    val_probs_seeds, test_probs_seeds, rows = [], [], []
    t_total = time.time()
    for seed in args.seeds:
        print(f"\n[mf-finetune] === seed {seed} ===", flush=True)
        t_seed = time.time()
        best_val_auc, val_res, test_res = train_one_seed(
            seed, smiles_tr, y_tr, smiles_va, y_va, smiles_te, y_te,
            epochs=args.epochs, patience=args.patience, device=device,
            batch_size=args.batch_size, lr_encoder=args.lr_encoder, lr_head=args.lr_head,
            n_keep_trainable=args.keep_trainable, max_len=args.max_len,
        )
        elapsed = time.time() - t_seed
        print(f"[mf-finetune] seed={seed} best_val_auc={best_val_auc:.4f}  "
              f"test_auc={test_res['AUROC']:.4f}  elapsed={elapsed:.0f}s", flush=True)
        val_probs_seeds.append(val_res["y_prob"])
        test_probs_seeds.append(test_res["y_prob"])
        rows.append({"seed": seed, "val_AUROC": best_val_auc,
                     "test_AUROC": float(test_res["AUROC"]), "elapsed_sec": float(elapsed)})

    val_mean = np.mean(np.stack(val_probs_seeds), axis=0)
    test_mean = np.mean(np.stack(test_probs_seeds), axis=0)

    seed_test_cols = {f"molformerFT_s{s}": test_probs_seeds[i] for i, s in enumerate(args.seeds)}
    seed_val_cols = {f"molformerFT_s{s}": val_probs_seeds[i] for i, s in enumerate(args.seeds)}
    np.savez(res / "molformer_ft_test_probs.npz", y_true=y_te.astype(np.int64),
             molformerFT=test_mean, **seed_test_cols)
    np.savez(res / "molformer_ft_val_probs.npz", y_true=y_va.astype(np.int64),
             molformerFT=val_mean, **seed_val_cols)
    pd.DataFrame(rows).to_csv(res / "molformer_ft_metrics.csv", index=False)

    yp = (test_mean >= 0.5).astype(int)
    print(f"\n[mf-finetune] mean-of-{len(args.seeds)} test:  AUROC={roc_auc_score(y_te, test_mean):.4f}  "
          f"ACC={accuracy_score(y_te, yp):.4f}  F1={f1_score(y_te, yp, zero_division=0):.4f}  "
          f"MCC={matthews_corrcoef(y_te, yp):.4f}", flush=True)
    print(f"[mf-finetune] total elapsed {(time.time()-t_total)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
