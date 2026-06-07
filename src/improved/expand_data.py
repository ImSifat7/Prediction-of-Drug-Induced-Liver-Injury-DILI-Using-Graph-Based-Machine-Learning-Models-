"""Build expanded training set by merging DILIrank + ClinTox + SIDER (Option G).

Critical correctness requirement: external compounds must NOT leak into the
val or test scaffolds of DILIrank. We dedupe in two ways:
  1. By canonical SMILES (exact match)
  2. By Bemis-Murcko scaffold (chemical-class similarity)
External compounds that share either with a val/test molecule are dropped.

Label mapping:
  - ClinTox CT_TOX=1 -> weak DILI-positive signal (drug failed in clinical trials
    for toxicity). Used as a noisy training label.
  - SIDER Hepatobiliary disorders=1 -> weak DILI-positive (post-marketing
    hepatic side effects). Used as a noisy training label.
  - Compound is "positive" if EITHER source flags it.
  - Compound is "negative" only if BOTH sources see it AND neither flags it.
  - Compounds with only one source are kept only if that source says "positive"
    (high precision) — single-source negatives are dropped (high false-neg risk).

Output:
  data/dili_expanded_train.csv  (DILIrank train + reconciled external)
  results/expand_summary.csv     (label-source breakdown)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.improved.data_utils import standardize_smiles, load_dataset, scaffold_split
else:
    from .data_utils import standardize_smiles, load_dataset, scaffold_split

RDLogger.DisableLog("rdApp.*")


def murcko(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""


def main():
    base = Path(__file__).resolve().parents[2]
    data = base / "data"

    # 1. Load DILIrank graphs to recover the exact val/test SMILES split
    print("[expand] loading DILIrank graphs + scaffold split ...", flush=True)
    graphs = load_dataset(data / "dili_clean.csv")
    train_idx, val_idx, test_idx = scaffold_split(graphs, 0.6, 0.2, 0.2, seed=42)
    dili_train_smiles = set(graphs[i].smiles for i in train_idx)
    dili_val_smiles = set(graphs[i].smiles for i in val_idx)
    dili_test_smiles = set(graphs[i].smiles for i in test_idx)
    dili_val_test_scaffolds = set(murcko(graphs[i].smiles) for i in val_idx) | \
                               set(murcko(graphs[i].smiles) for i in test_idx)
    print(f"  DILIrank: train={len(dili_train_smiles)} val={len(dili_val_smiles)} "
          f"test={len(dili_test_smiles)} ({len(dili_val_test_scaffolds)} val+test scaffolds)",
          flush=True)

    # Get DILIrank train labels too for output reconstruction.
    dili_train_label = {graphs[i].smiles: int(graphs[i].y.item()) for i in train_idx}

    # 2. Load ClinTox + SIDER, canonicalize, build a unified per-SMILES table
    print("[expand] loading ClinTox + SIDER ...", flush=True)
    clintox = pd.read_csv(data / "clintox.csv")
    sider = pd.read_csv(data / "sider.csv")
    print(f"  ClinTox raw: {clintox.shape}  SIDER raw: {sider.shape}", flush=True)

    # ClinTox positive = CT_TOX (failed clinical trials) OR FDA_APPROVED=0 indicates risk
    clintox["canon"] = clintox["smiles"].apply(standardize_smiles)
    clintox = clintox.dropna(subset=["canon"]).drop_duplicates(subset=["canon"])
    clintox_pos = set(clintox[clintox["CT_TOX"] == 1]["canon"])
    clintox_neg = set(clintox[(clintox["CT_TOX"] == 0) & (clintox["FDA_APPROVED"] == 1)]["canon"])
    print(f"  ClinTox after canon: {len(clintox)}  pos(CT_TOX=1)={len(clintox_pos)}  "
          f"neg(CT_TOX=0,FDA=1)={len(clintox_neg)}", flush=True)

    # SIDER: Hepatobiliary disorders=1 -> DILI-positive
    sider["canon"] = sider["smiles"].apply(standardize_smiles)
    sider = sider.dropna(subset=["canon"]).drop_duplicates(subset=["canon"])
    sider_pos = set(sider[sider["Hepatobiliary disorders"] == 1]["canon"])
    sider_neg = set(sider[sider["Hepatobiliary disorders"] == 0]["canon"])
    print(f"  SIDER after canon: {len(sider)}  hepatobiliary-pos={len(sider_pos)}  "
          f"hepatobiliary-neg={len(sider_neg)}", flush=True)

    # 3. Reconcile labels per molecule
    all_external = clintox_pos | clintox_neg | sider_pos | sider_neg
    print(f"\n[expand] unique external compounds across both sources: {len(all_external)}", flush=True)

    rows = []
    for smi in all_external:
        if smi in dili_val_smiles or smi in dili_test_smiles or smi in dili_train_smiles:
            continue  # dedup vs DILIrank entirely (avoid leakage AND double-counting)
        if murcko(smi) in dili_val_test_scaffolds and murcko(smi):
            continue  # also drop anything sharing a val/test scaffold (strict)
        in_clintox_pos = smi in clintox_pos
        in_clintox_neg = smi in clintox_neg
        in_sider_pos = smi in sider_pos
        in_sider_neg = smi in sider_neg

        pos_signals = int(in_clintox_pos) + int(in_sider_pos)
        neg_signals = int(in_clintox_neg) + int(in_sider_neg)

        # Label-reconciliation rules:
        #   - Any positive signal AND no contradicting strong negative -> positive
        #   - Both sources agree negative -> negative
        #   - Only one source says negative -> drop (too noisy)
        if pos_signals >= 1:
            label = 1
            confidence = "high" if pos_signals == 2 else "medium"
        elif neg_signals == 2:
            label = 0
            confidence = "high"
        else:
            continue
        rows.append({
            "smiles": smi, "label": label, "confidence": confidence,
            "clintox_pos": in_clintox_pos, "clintox_neg": in_clintox_neg,
            "sider_pos": in_sider_pos, "sider_neg": in_sider_neg,
            "source": "external",
        })

    external = pd.DataFrame(rows)
    print(f"\n[expand] external compounds after dedup+reconcile: {len(external)}", flush=True)
    print(f"  label=1: {(external['label']==1).sum()}  label=0: {(external['label']==0).sum()}", flush=True)
    print(f"  confidence=high: {(external['confidence']=='high').sum()}  "
          f"confidence=medium: {(external['confidence']=='medium').sum()}", flush=True)

    # 4. Combine with DILIrank training set
    dili_train = pd.DataFrame([
        {"smiles": s, "label": dili_train_label[s], "confidence": "expert",
         "clintox_pos": False, "clintox_neg": False, "sider_pos": False, "sider_neg": False,
         "source": "dilirank"}
        for s in dili_train_smiles
    ])
    combined = pd.concat([dili_train, external], ignore_index=True)
    print(f"\n[expand] FINAL combined training set: {len(combined)} compounds", flush=True)
    print(f"  DILIrank-train: {(combined['source']=='dilirank').sum()}", flush=True)
    print(f"  External:       {(combined['source']=='external').sum()}", flush=True)
    print(f"  label=1: {(combined['label']==1).sum()}  label=0: {(combined['label']==0).sum()}", flush=True)
    pos_rate = (combined['label']==1).mean()
    print(f"  pos_rate: {pos_rate:.3f} (vs DILIrank baseline {(dili_train['label']==1).mean():.3f})", flush=True)

    combined.to_csv(data / "dili_expanded_train.csv", index=False)
    print(f"\n[expand] wrote data/dili_expanded_train.csv ({len(combined)} rows)", flush=True)

    summary = {
        "n_dilirank_train": len(dili_train),
        "n_external_added": len(external),
        "n_total": len(combined),
        "pos_rate_dilirank": float((dili_train["label"] == 1).mean()),
        "pos_rate_combined": float(pos_rate),
    }
    pd.DataFrame([summary]).to_csv(base / "results" / "expand_summary.csv", index=False)
    print(f"[expand] wrote results/expand_summary.csv", flush=True)


if __name__ == "__main__":
    main()
