"""Data loading, SMILES standardization, atom/bond featurization, scaffold splitting.

Why this exists (vs the baseline scripts):
- Baseline used 5 atom features and no SMILES cleanup; salts/mixtures were sent in raw.
- Baseline used a random train/test split, which leaks similar molecules across folds.
- Here we standardize SMILES, expand atom features to ~43 dims, add bond features,
  and provide a Bemis-Murcko scaffold splitter for leakage-free evaluation.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional, Tuple
import random

import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")


def standardize_smiles(smiles: str) -> Optional[str]:
    """Keep only the largest organic fragment, drop salts/mixtures, return canonical SMILES."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    if mol.GetNumHeavyAtoms() < 2 or mol.GetNumBonds() == 0:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


# --- Featurization --------------------------------------------------------

ATOM_LIST = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "B", "Si", "Se"]
HYBRID_LIST = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    Chem.rdchem.HybridizationType.UNSPECIFIED,
]
DEGREE_LIST = [0, 1, 2, 3, 4, 5]
CHARGE_LIST = [-2, -1, 0, 1, 2]
NUMH_LIST = [0, 1, 2, 3, 4]
CHIRAL_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
RING_SIZES = [3, 4, 5, 6, 7]  # ring-size membership flags


def _one_hot(value, choices):
    vec = [0] * (len(choices) + 1)
    if value in choices:
        vec[choices.index(value)] = 1
    else:
        vec[-1] = 1
    return vec


def atom_features(atom) -> List[float]:
    f: List[float] = []
    f += _one_hot(atom.GetSymbol(), ATOM_LIST)         # 13
    f += _one_hot(atom.GetDegree(), DEGREE_LIST)        # 7
    f += _one_hot(atom.GetFormalCharge(), CHARGE_LIST)  # 6
    f += _one_hot(atom.GetTotalNumHs(), NUMH_LIST)      # 6
    f += _one_hot(atom.GetHybridization(), HYBRID_LIST) # 7
    f += _one_hot(atom.GetChiralTag(), CHIRAL_LIST)     # 5  (NEW: stereochemistry)
    f += [float(atom.GetIsAromatic())]                   # 1
    f += [float(atom.IsInRing())]                        # 1
    f += [float(atom.GetNumRadicalElectrons())]          # 1  (NEW: radicals)
    f += [atom.GetMass() * 0.01]                         # 1 (scaled)
    f += [float(atom.IsInRingSize(s)) for s in RING_SIZES]  # 5 (NEW: ring sizes)
    return f  # total = 53


ATOM_FEAT_DIM = 53


def bond_features(bond) -> List[float]:
    bt = bond.GetBondType()
    return [
        float(bt == Chem.rdchem.BondType.SINGLE),
        float(bt == Chem.rdchem.BondType.DOUBLE),
        float(bt == Chem.rdchem.BondType.TRIPLE),
        float(bt == Chem.rdchem.BondType.AROMATIC),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
        float(bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE),
    ]


BOND_FEAT_DIM = 7


def smiles_to_data(smiles: str, label: int) -> Optional[Data]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() < 2 or mol.GetNumBonds() == 0:
        return None

    x = [atom_features(a) for a in mol.GetAtoms()]
    edge_index, edge_attr = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_features(b)
        edge_index.append([i, j]); edge_attr.append(bf)
        edge_index.append([j, i]); edge_attr.append(bf)

    return Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        y=torch.tensor([label], dtype=torch.float),
        smiles=smiles,
    )


# --- Scaffold split -------------------------------------------------------

def murcko_scaffold(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""


def scaffold_split(
    graphs: List[Data],
    frac_train: float = 0.6,
    frac_val: float = 0.2,
    frac_test: float = 0.2,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Bemis-Murcko scaffold split. Largest scaffold groups go to train first."""
    assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-6
    rng = random.Random(seed)

    scaffolds: dict[str, list[int]] = {}
    for idx, g in enumerate(graphs):
        s = murcko_scaffold(g.smiles)
        scaffolds.setdefault(s, []).append(idx)

    groups = sorted(scaffolds.values(), key=lambda xs: (len(xs), rng.random()), reverse=True)

    n = len(graphs)
    n_train_target = int(frac_train * n)
    n_val_target = int(frac_val * n)

    train_idx, val_idx, test_idx = [], [], []
    for grp in groups:
        if len(train_idx) + len(grp) <= n_train_target:
            train_idx.extend(grp)
        elif len(val_idx) + len(grp) <= n_val_target:
            val_idx.extend(grp)
        else:
            test_idx.extend(grp)

    return train_idx, val_idx, test_idx


# --- High-level loader ----------------------------------------------------

def load_dataset(csv_path: Path) -> List[Data]:
    """Read CSV with columns drug_name,label,smiles -> list of PyG Data graphs.

    Standardizes SMILES, drops invalid/duplicate molecules.
    """
    df = pd.read_csv(csv_path)
    if "smiles" not in df.columns or "label" not in df.columns:
        raise ValueError(f"CSV must have 'smiles' and 'label' columns; got {df.columns.tolist()}")

    n_raw = len(df)
    df["smiles_std"] = df["smiles"].apply(standardize_smiles)
    df = df.dropna(subset=["smiles_std"]).drop_duplicates(subset=["smiles_std"]).reset_index(drop=True)
    n_clean = len(df)

    graphs: List[Data] = []
    for _, row in df.iterrows():
        g = smiles_to_data(row["smiles_std"], int(row["label"]))
        if g is not None:
            graphs.append(g)

    print(f"[data] raw={n_raw}  cleaned/uniq={n_clean}  graphable={len(graphs)}")
    print(f"[data] atom_feat_dim={graphs[0].x.shape[1]}  bond_feat_dim={graphs[0].edge_attr.shape[1]}")
    return graphs


if __name__ == "__main__":
    BASE = Path(__file__).resolve().parents[2]
    g = load_dataset(BASE / "data" / "dili_clean.csv")
    pos = sum(int(d.y.item()) for d in g)
    print(f"positives={pos} negatives={len(g) - pos} pos_ratio={pos / len(g):.3f}")
