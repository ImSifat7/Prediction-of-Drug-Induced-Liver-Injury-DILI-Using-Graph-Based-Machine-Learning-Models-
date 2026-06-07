"""ChemBERTa-77M fine-tune on TDC DILI with Test-Time Augmentation (TTA).

Trains 5 seeds. During inference, generates N_TTA random SMILES per molecule
(via RDKit canonical=False, doRandom=True) and averages the per-randomization
predictions. This is a robust, ensemble-like improvement that costs only
inference compute.

Run:
    python -m src.improved.tdc_chemberta_tta
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.improved.data_utils import standardize_smiles
    from src.improved.train_utils import set_seed
else:
    from .data_utils import standardize_smiles
    from .train_utils import set_seed

RDLogger.DisableLog("rdApp.*")

N_TTA = 10              # number of random SMILES per test molecule
SEEDS = [42, 7, 13, 21, 100]


def random_smiles(canonical: str, n: int, rng_seed: int = 0):
    """Generate n random (non-canonical) SMILES for the same molecule."""
    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        return [canonical] * n
    out = [canonical]  # always include canonical
    rng = np.random.default_rng(rng_seed)
    tries = 0
    while len(out) < n and tries < n * 4:
        s = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
        if s not in out and s:
            out.append(s)
        tries += 1
    while len(out) < n:
        out.append(canonical)
    return out


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
    return kept


def scaffold_indices(smiles_list, frac_train=0.7, frac_val=0.1, seed=42):
    from rdkit.Chem.Scaffolds import MurckoScaffold
    import random as _r
    rng = _r.Random(seed)
    scafs = {}
    for i, s in enumerate(smiles_list):
        m = Chem.MolFromSmiles(s)
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) if m else ""
        scafs.setdefault(scaf, []).append(i)
    groups = sorted(scafs.values(), key=lambda xs: (len(xs), rng.random()), reverse=True)
    n = len(smiles_list); n_tr = int(frac_train * n); n_va = int(frac_val * n)
    tr, va, te = [], [], []
    for g in groups:
        if len(tr) + len(g) <= n_tr: tr.extend(g)
        elif len(va) + len(g) <= n_va: va.extend(g)
        else: te.extend(g)
    return tr, va, te


def predict_tta(model, tok, smiles_list, device, max_len=200, batch_size=32, n_tta=N_TTA):
    """Predict with TTA: for each SMILES, average over N_TTA random forms."""
    model.eval()
    all_preds = np.zeros(len(smiles_list))
    for aug_i in range(n_tta):
        aug_smis = [random_smiles(s, n_tta, rng_seed=aug_i)[aug_i] for s in smiles_list]
        preds = []
        with torch.no_grad():
            for i in range(0, len(aug_smis), batch_size):
                chunk = aug_smis[i:i+batch_size]
                enc = tok(chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
                logits = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask).logits.view(-1)
                preds.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        all_preds += np.array(preds)
    return all_preds / n_tta


def main():
    base = Path(__file__).resolve().parents[2]
    res = base / "results"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cb-tta] device={device}", flush=True)

    pairs = load_tdc()
    smiles = [s for s, _ in pairs]
    labels = np.array([y for _, y in pairs], dtype=np.int64)
    train_idx, val_idx, test_idx = scaffold_indices(smiles)
    y_tr = labels[train_idx]; y_va = labels[val_idx]; y_te = labels[test_idx]
    print(f"[cb-tta] n={len(smiles)} tr/va/te={len(train_idx)}/{len(val_idx)}/{len(test_idx)}",
          flush=True)

    pos = (y_tr == 1).sum(); neg = (y_tr == 0).sum()
    spw = float(neg) / max(int(pos), 1)
    print(f"[cb-tta] scale_pos_weight={spw:.3f}", flush=True)

    smis_va = [smiles[i] for i in val_idx]
    smis_te = [smiles[i] for i in test_idx]

    tok = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
    va_bag = np.zeros(len(val_idx))
    te_bag = np.zeros(len(test_idx))
    seed_preds_va, seed_preds_te = {}, {}

    for sd, seed in enumerate(SEEDS):
        print(f"\n[cb-tta] seed {seed} ({sd+1}/{len(SEEDS)}) ...", flush=True)
        set_seed(seed)
        model = AutoModelForSequenceClassification.from_pretrained(
            "DeepChem/ChemBERTa-77M-MTR", num_labels=1, ignore_mismatched_sizes=True,
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
        pos_w = torch.tensor([spw], dtype=torch.float32, device=device)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        # Train with random SMILES augmentation each epoch
        tr_y = torch.tensor(y_tr, dtype=torch.float32)
        canonical_tr = [smiles[i] for i in train_idx]
        epochs = 25; patience = 5; bad = 0; best_val = -1.0
        best_p_va = None; best_p_te = None
        for ep in range(epochs):
            # Generate fresh random SMILES for this epoch
            np.random.seed(seed * 1000 + ep)
            aug_tr = [random_smiles(s, 2, rng_seed=seed * 1000 + ep + i)[1] for i, s in enumerate(canonical_tr)]
            enc_tr = tok(aug_tr, padding=True, truncation=True, max_length=200, return_tensors="pt")
            ds = TensorDataset(enc_tr.input_ids, enc_tr.attention_mask, tr_y)
            loader = DataLoader(ds, batch_size=16, shuffle=True)
            model.train()
            t0 = time.time(); ep_loss = 0; nb = 0
            for ids, mask, y in loader:
                ids, mask, y = ids.to(device), mask.to(device), y.to(device)
                logits = model(input_ids=ids, attention_mask=mask).logits.view(-1)
                loss = bce(logits, y)
                opt.zero_grad(); loss.backward(); opt.step()
                ep_loss += loss.item(); nb += 1
            # Validate with TTA every few epochs to save time
            do_eval = (ep < 3) or ((ep + 1) % 2 == 0)
            if do_eval:
                p_va = predict_tta(model, tok, smis_va, device, n_tta=5)
                va_auc = roc_auc_score(y_va, p_va)
                if va_auc > best_val:
                    best_val = va_auc; best_p_va = p_va.copy()
                    best_p_te = predict_tta(model, tok, smis_te, device, n_tta=N_TTA)
                    bad = 0
                else:
                    bad += 1
                print(f"  sd={seed} ep{ep+1:02d} loss={ep_loss/nb:.4f} "
                      f"val_tta={va_auc:.4f} best={best_val:.4f} ({time.time()-t0:.0f}s)",
                      flush=True)
                if bad >= patience:
                    print(f"  sd={seed} early-stop ep{ep+1}", flush=True); break
        # Ensure we have predictions
        if best_p_va is None:
            best_p_va = predict_tta(model, tok, smis_va, device)
            best_p_te = predict_tta(model, tok, smis_te, device)
        va_bag += best_p_va; te_bag += best_p_te
        seed_preds_va[f"CBTTA_s{seed}"] = best_p_va
        seed_preds_te[f"CBTTA_s{seed}"] = best_p_te
        print(f"  --> sd={seed} val_TTA={best_val:.4f} test_TTA={roc_auc_score(y_te, best_p_te):.4f}",
              flush=True)

        del model; torch.cuda.empty_cache() if device.type == "cuda" else None

    va_bag /= len(SEEDS); te_bag /= len(SEEDS)
    test_auc = roc_auc_score(y_te, te_bag)
    print(f"\n[cb-tta] MEAN-of-{len(SEEDS)} (TTA={N_TTA}) test AUROC = {test_auc:.4f}", flush=True)

    np.savez(res / "tdc_chemberta_tta_val_probs.npz", y_true=y_va, ChemBERTaTTA=va_bag, **seed_preds_va)
    np.savez(res / "tdc_chemberta_tta_test_probs.npz", y_true=y_te, ChemBERTaTTA=te_bag, **seed_preds_te)
    print("[cb-tta] wrote tdc_chemberta_tta_*.npz", flush=True)


if __name__ == "__main__":
    main()
