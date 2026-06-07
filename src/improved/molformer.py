"""MolFormer-XL embedding extraction + XGBoost classifier.

Why MolFormer-XL (Option A from advisor discussion):
  - IBM's foundation model (ibm/MoLFormer-XL-both-10pct).
  - Pretrained on 1.1 billion molecules (ZINC + PubChem) with a SMILES-tokenizer
    + transformer encoder. 47M parameters.
  - Single biggest realistic lever for pushing DILI accuracy past current 0.72
    ceiling: brings in chemistry knowledge our 869-molecule dataset cannot teach.

Pipeline:
  1. Load tokenizer + model from HF Hub (cached after first run).
  2. For each SMILES, get the pooled output (768-dim "molecule embedding").
  3. Train XGBoost on these embeddings, scaffold-split-aware.
  4. Save predictions for the stack_final.py ensembler.

Run:
    python -m src.improved.molformer
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
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
else:
    from .data_utils import load_dataset, scaffold_split


MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"
BATCH_SIZE = 8
MAX_LEN = 256  # MolFormer was pretrained up to 202 tokens; pad to 256 for safety


def _patch_transformers_for_molformer():
    """MolFormer's dynamic modeling code was written against transformers ~4.30
    and breaks against transformers 5.x because of removed/relocated symbols.
    Inject shims so we can still load it without downgrading the whole environment.

    Specifically:
      - transformers.onnx.OnnxConfig (entire submodule removed)
      - transformers.pytorch_utils.find_pruneable_heads_and_indices (removed)
      - PreTrainedModel.get_head_mask (removed)
    """
    import sys
    import types

    # Shim 1: transformers.onnx
    if "transformers.onnx" not in sys.modules:
        onnx_stub = types.ModuleType("transformers.onnx")
        class OnnxConfig:
            pass
        onnx_stub.OnnxConfig = OnnxConfig
        sys.modules["transformers.onnx"] = onnx_stub

    # Shim 2: find_pruneable_heads_and_indices (we never prune; stub is fine)
    import transformers.pytorch_utils as pu
    if not hasattr(pu, "find_pruneable_heads_and_indices"):
        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            return set(), torch.LongTensor([])
        pu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

    # Shim 3: PreTrainedModel.get_head_mask
    from transformers import PreTrainedModel
    if not hasattr(PreTrainedModel, "get_head_mask"):
        def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            if head_mask is None:
                return [None] * num_hidden_layers
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
            return head_mask
        PreTrainedModel.get_head_mask = get_head_mask


@torch.no_grad()
def extract_molformer_embeddings(smiles_list: List[str], device, batch_size: int = BATCH_SIZE) -> np.ndarray:
    _patch_transformers_for_molformer()
    from transformers import AutoTokenizer, AutoModel

    print(f"[molformer] loading {MODEL_NAME} ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, deterministic_eval=True)
    model = model.to(device).eval()
    print(f"[molformer] loaded in {time.time()-t0:.1f}s — extracting embeddings...", flush=True)

    embs: List[np.ndarray] = []
    n = len(smiles_list)
    t_start = time.time()
    for i in range(0, n, batch_size):
        batch = smiles_list[i:i + batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        # MolFormer returns last_hidden_state of shape [B, T, 768]; use mean over
        # non-padding tokens as the molecule embedding.
        mask = enc["attention_mask"].unsqueeze(-1).float()  # [B, T, 1]
        pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        embs.append(pooled.cpu().numpy())
        if (i // batch_size) % 10 == 0:
            elapsed = time.time() - t_start
            rate = max(i, 1) / max(elapsed, 1e-6)
            eta = (n - i) / max(rate, 1e-6)
            print(f"  [{i}/{n}] {rate:.1f} mol/s  ETA={eta:.0f}s", flush=True)
    print(f"[molformer] extraction took {time.time()-t_start:.1f}s for {n} molecules", flush=True)
    return np.concatenate(embs, axis=0)


def _best_thr(y, p, metric="MCC"):
    fns = {
        "MCC": matthews_corrcoef,
        "ACC": accuracy_score,
        "F1": lambda y, yp: f1_score(y, yp, zero_division=0),
    }
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
    results_dir = base / "results"
    results_dir.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[molformer] device={device}", flush=True)

    # Load data + scaffold split (matches GNN/classical/ChemBERTa).
    graphs = load_dataset(base / "data" / "dili_clean.csv")
    smiles_all = [g.smiles for g in graphs]
    labels = np.array([int(g.y.item()) for g in graphs], dtype=np.int64)
    train_idx, val_idx, test_idx = scaffold_split(graphs, 0.6, 0.2, 0.2, seed=42)

    y_train = labels[train_idx]
    y_val = labels[val_idx]
    y_test = labels[test_idx]
    print(f"[molformer] split  tr={len(train_idx)} va={len(val_idx)} te={len(test_idx)}", flush=True)

    # Embed every molecule once; reuse cached if available.
    cache = results_dir / "molformer_embeddings.npy"
    if cache.exists():
        print(f"[molformer] loading cached embeddings from {cache.name}", flush=True)
        emb_all = np.load(cache)
        if len(emb_all) != len(smiles_all):
            print("[molformer] cache size mismatch — recomputing", flush=True)
            emb_all = extract_molformer_embeddings(smiles_all, device)
            np.save(cache, emb_all)
    else:
        emb_all = extract_molformer_embeddings(smiles_all, device)
        np.save(cache, emb_all)
    print(f"[molformer] embeddings shape: {emb_all.shape}", flush=True)

    X_train = emb_all[train_idx]
    X_val = emb_all[val_idx]
    X_test = emb_all[test_idx]

    # Train XGBoost with multi-seed bagging for stable predictions.
    pos = (y_train == 1).sum(); neg = (y_train == 0).sum()
    spw = float(neg) / max(int(pos), 1)
    print(f"[molformer] training bagged XGBoost (5 seeds)...", flush=True)
    val_probs = np.zeros(len(y_val))
    test_probs = np.zeros(len(y_test))
    n_bag = 5
    for s in range(n_bag):
        clf = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.6,
            reg_lambda=1.0, scale_pos_weight=spw,
            random_state=s, eval_metric="logloss", n_jobs=4, verbosity=0,
        )
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        val_probs += clf.predict_proba(X_val)[:, 1]
        test_probs += clf.predict_proba(X_test)[:, 1]
    val_probs /= n_bag
    test_probs /= n_bag

    rows = []
    for tgt in ("MCC", "ACC", "F1"):
        thr = _best_thr(y_val, val_probs, metric=tgt)
        m = _m(y_test, test_probs, thr)
        print(f"  thr={thr:.2f} (tuned-for-{tgt})  AUROC={m['AUROC']:.4f}  "
              f"ACC={m['ACC']:.4f}  F1={m['F1']:.4f}  MCC={m['MCC']:.4f}", flush=True)
        rows.append({"target": tgt, "threshold": thr, **m})
    pd.DataFrame(rows).to_csv(results_dir / "molformer_metrics.csv", index=False)

    np.savez(
        results_dir / "molformer_test_probs.npz",
        y_true=y_test.astype(np.int64),
        molformer=test_probs,
    )
    np.savez(
        results_dir / "molformer_val_probs.npz",
        y_true=y_val.astype(np.int64),
        molformer=val_probs,
    )
    print(f"[molformer] wrote results/molformer_metrics.csv + molformer_*_probs.npz", flush=True)


if __name__ == "__main__":
    main()
