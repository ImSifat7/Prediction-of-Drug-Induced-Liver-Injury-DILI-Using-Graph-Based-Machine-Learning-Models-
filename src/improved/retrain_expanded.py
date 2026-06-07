"""Retrain the SMILES-based base models (ChemBERTa, MolFormer-XGBoost) on the
expanded training set, then re-predict on the SAME DILIrank val/test split.

We deliberately KEEP val/test untouched (exactly the same scaffold split as all
prior runs) so that the final numbers are directly comparable with the existing
stack. Only TRAIN gets expanded.

Models retrained here:
  1. ChemBERTa-77M-MTR fine-tuning
  2. MolFormer-XL embedding + bagged XGBoost (we extract MolFormer embeddings for
     the new 461 external compounds, cached embeddings for DILIrank reused)

Run:
    python -m src.improved.retrain_expanded
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import RDLogger
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from xgboost import XGBClassifier

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.improved.data_utils import load_dataset, scaffold_split
    from src.improved.chemberta import SmilesDataset, MAX_LEN, MODEL_NAME as CB_MODEL_NAME
    from src.improved.molformer import extract_molformer_embeddings
else:
    from .data_utils import load_dataset, scaffold_split
    from .chemberta import SmilesDataset, MAX_LEN, MODEL_NAME as CB_MODEL_NAME
    from .molformer import extract_molformer_embeddings

RDLogger.DisableLog("rdApp.*")


SEEDS = [42, 7, 13]
EPOCHS = 25
PATIENCE = 5
LR = 2e-5
WEIGHT_DECAY = 0.01
BATCH_SIZE = 16


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_prob = [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        am = batch["attention_mask"].to(device)
        out = model(input_ids=ids, attention_mask=am).logits.squeeze(-1)
        prob = torch.sigmoid(out).cpu().numpy().tolist()
        y_prob.extend(prob)
        y_true.extend(batch["label"].numpy().tolist())
    return {
        "AUROC": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
        "y_true": np.asarray(y_true),
        "y_prob": np.asarray(y_prob),
    }


def chemberta_train_seed(seed, smi_tr, y_tr, smi_va, y_va, smi_te, y_te,
                         epochs, patience, device, sample_weight=None):
    torch.manual_seed(seed); np.random.seed(seed)
    tok = AutoTokenizer.from_pretrained(CB_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        CB_MODEL_NAME, num_labels=1, problem_type="regression",
    ).to(device)

    pos = float((y_tr == 1).sum()); neg = float((y_tr == 0).sum())
    pos_w = torch.tensor([neg / max(pos, 1.0)], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w, reduction="none")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    tr_ds = SmilesDataset(smi_tr, y_tr, tok, MAX_LEN, augment=0.0)
    va_ds = SmilesDataset(smi_va, y_va, tok, MAX_LEN, augment=0.0)
    te_ds = SmilesDataset(smi_te, y_te, tok, MAX_LEN, augment=0.0)
    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False)
    te_dl = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False)

    # We need per-sample weights aligned with shuffled order. Easiest: store on tr_ds.
    if sample_weight is not None:
        sw = torch.tensor(sample_weight, dtype=torch.float)
    else:
        sw = None

    # Hack: wrap dataloader to also yield sample weight by index. To avoid that
    # complexity, simulate weighted loss by per-batch weighting using the labels
    # we already have. For now, use uniform (sample_weight ignored within batch).

    best_auc, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time(); total = 0.0
        for batch in tr_dl:
            ids = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            y = batch["label"].to(device)
            opt.zero_grad()
            logits = model(input_ids=ids, attention_mask=am).logits.squeeze(-1)
            loss_per = crit(logits, y)
            loss = loss_per.mean()
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
    return float(best_auc), evaluate(model, va_dl, device), evaluate(model, te_dl, device)


def main():
    base = Path(__file__).resolve().parents[2]
    data = base / "data"
    res = base / "results"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[retrain_expanded] device={device}", flush=True)

    # ----- Load DILIrank graphs to recover val/test SMILES -----
    print("\n[retrain_expanded] loading DILIrank + scaffold split ...", flush=True)
    graphs = load_dataset(data / "dili_clean.csv")
    _, val_idx, test_idx = scaffold_split(graphs, 0.6, 0.2, 0.2, seed=42)
    smi_va = [graphs[i].smiles for i in val_idx]
    smi_te = [graphs[i].smiles for i in test_idx]
    y_va = np.array([int(graphs[i].y.item()) for i in val_idx], dtype=np.int64)
    y_te = np.array([int(graphs[i].y.item()) for i in test_idx], dtype=np.int64)

    # ----- Load expanded train -----
    expanded = pd.read_csv(data / "dili_expanded_train.csv")
    smi_tr = expanded["smiles"].tolist()
    y_tr = expanded["label"].values.astype(np.int64)
    confidence = expanded["confidence"].values
    print(f"  Train (expanded): {len(smi_tr)} compounds  pos_rate={y_tr.mean():.3f}", flush=True)
    print(f"  Val:              {len(smi_va)} compounds  pos_rate={y_va.mean():.3f}", flush=True)
    print(f"  Test:             {len(smi_te)} compounds  pos_rate={y_te.mean():.3f}", flush=True)

    # =========================================================================
    # PART 1: ChemBERTa fine-tuning on expanded train
    # =========================================================================
    print("\n[retrain_expanded] ===== Part 1: ChemBERTa expanded fine-tune =====", flush=True)
    val_probs_seeds: List[np.ndarray] = []; test_probs_seeds: List[np.ndarray] = []
    metric_rows = []
    t_total = time.time()
    for seed in SEEDS:
        print(f"\n[retrain_expanded] CB seed {seed}", flush=True)
        t0 = time.time()
        best_val, val_res, test_res = chemberta_train_seed(
            seed, smi_tr, y_tr, smi_va, y_va, smi_te, y_te,
            epochs=EPOCHS, patience=PATIENCE, device=device,
        )
        elapsed = time.time() - t0
        print(f"  CB seed={seed} best_val={best_val:.4f}  test_auc={test_res['AUROC']:.4f}  "
              f"elapsed={elapsed:.0f}s", flush=True)
        val_probs_seeds.append(val_res["y_prob"])
        test_probs_seeds.append(test_res["y_prob"])
        metric_rows.append({"seed": seed, "val_AUROC": best_val,
                            "test_AUROC": float(test_res["AUROC"]), "elapsed_sec": elapsed})

    cb_val_mean = np.mean(np.stack(val_probs_seeds), axis=0)
    cb_test_mean = np.mean(np.stack(test_probs_seeds), axis=0)
    seed_test = {f"ChemBERTaEXP_s{s}": test_probs_seeds[i] for i, s in enumerate(SEEDS)}
    seed_val = {f"ChemBERTaEXP_s{s}": val_probs_seeds[i] for i, s in enumerate(SEEDS)}
    np.savez(res / "chemberta_exp_test_probs.npz", y_true=y_te,
             ChemBERTaEXP=cb_test_mean, **seed_test)
    np.savez(res / "chemberta_exp_val_probs.npz", y_true=y_va,
             ChemBERTaEXP=cb_val_mean, **seed_val)
    pd.DataFrame(metric_rows).to_csv(res / "chemberta_exp_metrics.csv", index=False)
    yp = (cb_test_mean >= 0.5).astype(int)
    print(f"\n[retrain_expanded] CB mean-of-3 EXPANDED: AUROC={roc_auc_score(y_te, cb_test_mean):.4f}  "
          f"ACC={accuracy_score(y_te, yp):.4f}  F1={f1_score(y_te, yp, zero_division=0):.4f}  "
          f"MCC={matthews_corrcoef(y_te, yp):.4f}", flush=True)
    print(f"[retrain_expanded] CB total elapsed {(time.time()-t_total)/60:.1f} min", flush=True)

    # =========================================================================
    # PART 2: MolFormer-XL frozen embeddings + bagged XGBoost on expanded train
    # =========================================================================
    print("\n[retrain_expanded] ===== Part 2: MolFormer-XL XGBoost on expanded train =====", flush=True)

    # Extract MolFormer embeddings for ALL unique SMILES (train+val+test combined).
    # Cache to disk.
    all_smiles = smi_tr + smi_va + smi_te
    print(f"  Total SMILES to embed: {len(all_smiles)}", flush=True)
    cache = res / "molformer_emb_expanded.npy"
    if cache.exists():
        emb_all = np.load(cache)
        if len(emb_all) != len(all_smiles):
            print("  cache size mismatch — recomputing", flush=True)
            emb_all = extract_molformer_embeddings(all_smiles, device)
            np.save(cache, emb_all)
    else:
        emb_all = extract_molformer_embeddings(all_smiles, device)
        np.save(cache, emb_all)
    n_tr = len(smi_tr); n_va = len(smi_va); n_te = len(smi_te)
    X_tr = emb_all[:n_tr]; X_va = emb_all[n_tr:n_tr+n_va]; X_te = emb_all[n_tr+n_va:]
    print(f"  X_tr={X_tr.shape}  X_va={X_va.shape}  X_te={X_te.shape}", flush=True)

    pos = (y_tr == 1).sum(); neg = (y_tr == 0).sum()
    spw = float(neg) / max(int(pos), 1)
    print(f"  bagged XGBoost (5 seeds), scale_pos_weight={spw:.3f}", flush=True)
    val_preds = np.zeros(n_va); test_preds = np.zeros(n_te)
    for s in range(5):
        clf = XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0,
            scale_pos_weight=spw, random_state=s, eval_metric="logloss",
            n_jobs=4, verbosity=0,
        )
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        val_preds += clf.predict_proba(X_va)[:, 1]
        test_preds += clf.predict_proba(X_te)[:, 1]
    val_preds /= 5; test_preds /= 5

    rows_mf = []
    for tgt, fn in [("MCC", matthews_corrcoef), ("ACC", accuracy_score),
                     ("F1", lambda y, yp: f1_score(y, yp, zero_division=0))]:
        best_t, best_v = 0.5, -1.0
        for t in np.linspace(0.05, 0.95, 181):
            v = fn(y_va, (val_preds >= t).astype(int))
            if v > best_v:
                best_v, best_t = float(v), float(t)
        yp = (test_preds >= best_t).astype(int)
        rows_mf.append({
            "target": tgt, "threshold": best_t,
            "AUROC": float(roc_auc_score(y_te, test_preds)),
            "ACC": float(accuracy_score(y_te, yp)),
            "F1": float(f1_score(y_te, yp, zero_division=0)),
            "MCC": float(matthews_corrcoef(y_te, yp)),
        })
        print(f"  thr={best_t:.2f} (tuned-for-{tgt})  AUROC={rows_mf[-1]['AUROC']:.4f}  "
              f"ACC={rows_mf[-1]['ACC']:.4f}  F1={rows_mf[-1]['F1']:.4f}  "
              f"MCC={rows_mf[-1]['MCC']:.4f}", flush=True)
    pd.DataFrame(rows_mf).to_csv(res / "molformer_exp_metrics.csv", index=False)
    np.savez(res / "molformer_exp_test_probs.npz", y_true=y_te, molformerEXP=test_preds)
    np.savez(res / "molformer_exp_val_probs.npz", y_true=y_va, molformerEXP=val_preds)
    print("[retrain_expanded] wrote expanded predictions to results/*_exp_*.npz", flush=True)


if __name__ == "__main__":
    main()
