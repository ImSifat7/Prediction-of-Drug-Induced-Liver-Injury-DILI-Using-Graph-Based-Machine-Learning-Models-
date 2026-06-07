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

import math
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, FilterCatalog, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")


# --- Structural alert catalogs (loaded once) ------------------------------
# Brenk = unwanted reactive / unstable substructures (105 patterns).
# PAINS = pan-assay interference compounds (480 patterns), commonly flagged for
#         non-specific bioactivity and idiosyncratic toxicity.
# NIH   = NIH's structural alert list (180 patterns).
# These are well-known toxicophore catalogs in medicinal chemistry; their counts
# per molecule encode decades of human prior knowledge about toxicity.

def _build_filter_catalog(which):
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(which)
    return FilterCatalog.FilterCatalog(params)


_FC_BRENK = _build_filter_catalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
_FC_PAINS = _build_filter_catalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
_FC_NIH = _build_filter_catalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.NIH)


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


def _safe_gasteiger(atom) -> float:
    """Per-atom Gasteiger partial charge with NaN/inf guard.

    Why: polar atoms (carbonyl O, amine N) drive a lot of DILI-relevant chemistry
    (reactive metabolite formation, mitochondrial uncoupling). Without partial
    charges the GNN can only infer polarity indirectly through hybridization +
    element identity. RDKit's Gasteiger algorithm occasionally returns NaN for
    sulfonates / phosphates; clip to a sane range.
    """
    try:
        q = float(atom.GetDoubleProp("_GasteigerCharge"))
        if math.isnan(q) or math.isinf(q):
            return 0.0
        # Clip to [-2, 2] — empirical bound for organic chemistry.
        return max(-2.0, min(2.0, q))
    except Exception:
        return 0.0


def atom_features(atom) -> List[float]:
    f: List[float] = []
    f += _one_hot(atom.GetSymbol(), ATOM_LIST)         # 13
    f += _one_hot(atom.GetDegree(), DEGREE_LIST)        # 7
    f += _one_hot(atom.GetFormalCharge(), CHARGE_LIST)  # 6
    f += _one_hot(atom.GetTotalNumHs(), NUMH_LIST)      # 6
    f += _one_hot(atom.GetHybridization(), HYBRID_LIST) # 7
    f += _one_hot(atom.GetChiralTag(), CHIRAL_LIST)     # 5
    f += [float(atom.GetIsAromatic())]                   # 1
    f += [float(atom.IsInRing())]                        # 1
    f += [float(atom.GetNumRadicalElectrons())]          # 1
    f += [atom.GetMass() * 0.01]                         # 1 (scaled)
    f += [float(atom.IsInRingSize(s)) for s in RING_SIZES]  # 5
    f += [_safe_gasteiger(atom)]                         # 1 (NEW: Gasteiger partial charge)
    return f  # total = 54


ATOM_FEAT_DIM = 54


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


# --- Per-molecule global descriptors --------------------------------------
# Why these specific descriptors: they are the same physicochemical features
# that make classical RF (FP+Mordred) beat the GNN on this dataset — LogP +
# molecular size + hydrogen-bonding capacity. Fusing them at the readout layer
# gives the GNN access to the same global view, while the graph layers still
# learn local toxophore substructure. This is the "Hybrid GAT-XGBoost" idea
# from the advisor's plan, but inline within each model.

GLOBAL_DESC_NAMES = [
    "MolLogP",         # Crippen LogP — lipophilicity, central to DILI
    "MolWt",           # molecular weight
    "TPSA",            # topological polar surface area
    "NumHDonors",      # Lipinski H-bond donors
    "NumHAcceptors",   # Lipinski H-bond acceptors
    "NumRotatableBonds",
    "NumAromaticRings",
    "NumAliphaticRings",
    "FractionCSP3",    # sp3 carbon fraction — drug-likeness
    "HeavyAtomCount",
    # Toxicophore alert counts — decades of human medicinal-chemistry priors
    # captured as substructure counts. Each catalog flags different concerns:
    "BrenkAlerts",     # unstable / reactive substructures (~105 SMARTS)
    "PAINSAlerts",     # pan-assay interference / non-specific bioactivity (~480)
    "NIHAlerts",       # NIH's structural alert list (~180)
    "NumHeteroaromaticAtoms",  # additional toxicophore proxy: heteroatomic aromatics
]
GLOBAL_DESC_DIM = len(GLOBAL_DESC_NAMES)

_DESC_SCALES = {
    "MolLogP": 5.0, "MolWt": 500.0, "TPSA": 150.0,
    "NumHDonors": 6.0, "NumHAcceptors": 12.0, "NumRotatableBonds": 12.0,
    "NumAromaticRings": 5.0, "NumAliphaticRings": 5.0,
    "FractionCSP3": 1.0, "HeavyAtomCount": 60.0,
    "BrenkAlerts": 5.0, "PAINSAlerts": 3.0, "NIHAlerts": 5.0,
    "NumHeteroaromaticAtoms": 20.0,
}


def _safe(value, scale: float) -> float:
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return 0.0
        return v / scale
    except Exception:
        return 0.0


def _count_alerts(mol, fc) -> int:
    try:
        return len(fc.GetMatches(mol))
    except Exception:
        return 0


def _count_heteroaromatic_atoms(mol) -> int:
    return sum(1 for a in mol.GetAtoms() if a.GetIsAromatic() and a.GetSymbol() != "C")


def global_descriptors(mol) -> List[float]:
    """RDKit per-molecule descriptors + structural-alert counts, all scaled to O(1)."""
    return [
        _safe(Crippen.MolLogP(mol), _DESC_SCALES["MolLogP"]),
        _safe(Descriptors.MolWt(mol), _DESC_SCALES["MolWt"]),
        _safe(Descriptors.TPSA(mol), _DESC_SCALES["TPSA"]),
        _safe(Lipinski.NumHDonors(mol), _DESC_SCALES["NumHDonors"]),
        _safe(Lipinski.NumHAcceptors(mol), _DESC_SCALES["NumHAcceptors"]),
        _safe(Lipinski.NumRotatableBonds(mol), _DESC_SCALES["NumRotatableBonds"]),
        _safe(Lipinski.NumAromaticRings(mol), _DESC_SCALES["NumAromaticRings"]),
        _safe(Lipinski.NumAliphaticRings(mol), _DESC_SCALES["NumAliphaticRings"]),
        _safe(Descriptors.FractionCSP3(mol), _DESC_SCALES["FractionCSP3"]),
        _safe(Descriptors.HeavyAtomCount(mol), _DESC_SCALES["HeavyAtomCount"]),
        _safe(_count_alerts(mol, _FC_BRENK), _DESC_SCALES["BrenkAlerts"]),
        _safe(_count_alerts(mol, _FC_PAINS), _DESC_SCALES["PAINSAlerts"]),
        _safe(_count_alerts(mol, _FC_NIH), _DESC_SCALES["NIHAlerts"]),
        _safe(_count_heteroaromatic_atoms(mol), _DESC_SCALES["NumHeteroaromaticAtoms"]),
    ]


def smiles_to_data(smiles: str, label: int) -> Optional[Data]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() < 2 or mol.GetNumBonds() == 0:
        return None

    # Compute Gasteiger charges once — atom_features reads them via _GasteigerCharge prop.
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        pass  # _safe_gasteiger returns 0.0 if prop missing

    x = [atom_features(a) for a in mol.GetAtoms()]
    edge_index, edge_attr = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_features(b)
        edge_index.append([i, j]); edge_attr.append(bf)
        edge_index.append([j, i]); edge_attr.append(bf)

    u = torch.tensor([global_descriptors(mol)], dtype=torch.float)  # shape [1, GLOBAL_DESC_DIM]

    return Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        y=torch.tensor([label], dtype=torch.float),
        u=u,
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
    print(f"[data] atom_feat_dim={graphs[0].x.shape[1]}  bond_feat_dim={graphs[0].edge_attr.shape[1]}  "
          f"global_desc_dim={graphs[0].u.shape[1]}")
    return graphs


if __name__ == "__main__":
    BASE = Path(__file__).resolve().parents[2]
    g = load_dataset(BASE / "data" / "dili_clean.csv")
    pos = sum(int(d.y.item()) for d in g)
    print(f"positives={pos} negatives={len(g) - pos} pos_ratio={pos / len(g):.3f}")
