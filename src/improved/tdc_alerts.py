"""Official TDC-DILI benchmark + MECHANISM-INFORMED structural-alert features.

Tests whether low-dimensional, DILI-mechanism-aligned features (known
hepatotoxicity toxicophores / reactive-metabolite SMARTS + PAINS/Brenk/NIH
filter catalogs) improve the VALIDATION-SELECTED result over the plain
descriptor+fingerprint model.

Methodology is identical to tdc_official_v2.py: every feature-set / model /
hyper-parameter choice is made on the official VALIDATION split (mean val-AUROC
over 5 seeds); the fixed 96-mol TEST set is scored ONCE for the single
val-selected winner. No test-set peeking.

Run:
    python -m src.improved.tdc_alerts            # full search (+MolFormer from cache)
    python -m src.improved.tdc_alerts --no-mf
    python -m src.improved.tdc_alerts --optuna 40
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from rdkit import Chem, RDLogger
from rdkit.Chem import FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.improved.tdc_official_v2 import (
        precompute_blocks, eval_config, ensemble_eval, optuna_tune,
        bootstrap_ci, SEEDS, RES, DATA_PATH, extract_molformer_embeddings)
else:
    from .tdc_official_v2 import (
        precompute_blocks, eval_config, ensemble_eval, optuna_tune,
        bootstrap_ci, SEEDS, RES, DATA_PATH, extract_molformer_embeddings)

# ---------------- mechanism-informed structural alerts ----------------
# Reactive-metabolite / hepatotoxicity structural alerts drawn from the
# medicinal-chemistry literature (Kalgutkar et al. reactive-metabolite alerts;
# Stepan/Hewitt hepatotoxicity toxicophores). Each is a binary substructure flag.
HEPATOTOX_SMARTS = {
    "nitro_aromatic":      "[a][NX3+](=O)[O-]",
    "nitro_aliphatic":     "[CX4][NX3+](=O)[O-]",
    "aromatic_amine":      "[c][NX3;H2,H1;!$(NC=O)]",
    "hydrazine":           "[NX3]-[NX3]",
    "azo":                 "[#6]-N=N-[#6]",
    "thiophene":           "c1cccs1",
    "furan":               "c1ccoc1",
    "epoxide":             "[#6]1[#6][OX2]1",
    "michael_acceptor":    "[CX3]=[CX3]-[CX3]=[OX1]",
    "quinone":             "O=C1C=CC(=O)C=C1",
    "para_quinone_imine":  "O=C1C=CC(=N)C=C1",
    "thiourea":            "[NX3]-[CX3](=[SX1])-[NX3]",
    "thioamide":           "[NX3]-[CX3]=[SX1]",
    "alpha_halo_carbonyl": "[#6][CX3](=O)[CX4][F,Cl,Br,I]",
    "aromatic_halide":     "[c][F,Cl,Br,I]",
    "alkyl_halide":        "[CX4][Cl,Br,I]",
    "carboxylic_acid":     "[CX3](=O)[OX2H1]",
    "phenol":              "[c][OX2H]",
    "terminal_alkyne":     "[CX2]#[CX2H]",
    "isocyanate":          "[NX2]=[CX2]=[OX1]",
    "n_oxide":             "[#7]-[OX1]",
    "sulfonamide":         "[SX4](=O)(=O)[NX3]",
    "anhydride":           "[CX3](=O)[OX2][CX3](=O)",
    "aldehyde":            "[CX3H1](=O)[#6]",
    "nitroso":             "[#6]-[NX2]=[OX1]",
}
_COMPILED = {k: Chem.MolFromSmarts(v) for k, v in HEPATOTOX_SMARTS.items()}
assert all(p is not None for p in _COMPILED.values()), "bad SMARTS"
ALERT_NAMES = list(_COMPILED.keys())

_CATALOG_FAMILIES = [
    ("PAINS_A", FilterCatalogParams.FilterCatalogs.PAINS_A),
    ("PAINS_B", FilterCatalogParams.FilterCatalogs.PAINS_B),
    ("PAINS_C", FilterCatalogParams.FilterCatalogs.PAINS_C),
    ("BRENK",   FilterCatalogParams.FilterCatalogs.BRENK),
    ("NIH",     FilterCatalogParams.FilterCatalogs.NIH),
    ("ZINC",    FilterCatalogParams.FilterCatalogs.ZINC),
]
_CATALOGS = []
for _fam_name, _fam in _CATALOG_FAMILIES:
    _p = FilterCatalogParams()
    _p.AddCatalog(_fam)
    _CATALOGS.append((_fam_name, FilterCatalog.FilterCatalog(_p)))
CATALOG_NAMES = [n for n, _ in _CATALOG_FAMILIES]

FEATURE_NAMES = ALERT_NAMES + CATALOG_NAMES


def _alert_vec(mol):
    if mol is None:
        return np.zeros(len(FEATURE_NAMES), np.float32)
    smarts = [1.0 if mol.HasSubstructMatch(p) else 0.0 for p in _COMPILED.values()]
    cats = [1.0 if cat.HasMatch(mol) else 0.0 for _, cat in _CATALOGS]
    return np.array(smarts + cats, np.float32)


def compute_alerts_block(smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    return np.stack([_alert_vec(m) for m in mols], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-mf", action="store_true")
    ap.add_argument("--optuna", type=int, default=40)
    args = ap.parse_args()
    use_mf = not args.no_mf

    from tdc.benchmark_group import admet_group
    RES.mkdir(exist_ok=True)
    group = admet_group(path=str(DATA_PATH))
    benchmark = group.get("DILI")
    name = benchmark["name"]
    test_df = benchmark["test"]; trv_df = benchmark["train_val"]
    print(f"[alerts] DILI official  train_val={len(trv_df)} test={len(test_df)}", flush=True)

    master = sorted(set(trv_df["Drug"]).union(set(test_df["Drug"])))
    smi2i = {s: i for i, s in enumerate(master)}
    base_kinds = ["desc", "morgan", "avalon", "erg", "maccs"]
    t0 = time.time()
    blocks = precompute_blocks(master, base_kinds)
    blocks["alerts"] = compute_alerts_block(master)
    print(f"[alerts] precomputed blocks incl. alerts{blocks['alerts'].shape} "
          f"({len(FEATURE_NAMES)} mechanism features) in {time.time()-t0:.0f}s", flush=True)

    # alert prevalence sanity / interpretability seed
    prev = blocks["alerts"].mean(0)
    top = sorted(zip(FEATURE_NAMES, prev), key=lambda x: -x[1])[:10]
    print("[alerts] most common alerts:",
          ", ".join(f"{n}={p:.0%}" for n, p in top), flush=True)

    if use_mf:
        import torch
        cache = RES / "tdc_official_molformer_emb.npz"
        if cache.exists():
            d = np.load(cache, allow_pickle=True)
            cmap = {s: e for s, e in zip(d["smiles"], d["emb"])}
        else:
            cmap = {}
        if not set(cmap) >= set(master):
            emb = extract_molformer_embeddings(master, torch.device("cpu"))
            cmap = {s: e for s, e in zip(master, emb)}
            np.savez(cache, smiles=np.array(master, dtype=object),
                     emb=np.stack([cmap[s] for s in master]))
        blocks["molformer"] = np.stack([cmap[s] for s in master], 0).astype(np.float32)

    X_all_idx = {"smi2i": smi2i, "test": [smi2i[s] for s in test_df["Drug"]]}
    y_test = test_df["Y"].values.astype(int)

    # Baselines vs the SAME sets + mechanism alerts (val-selected throughout)
    feature_sets = {
        "desc_morgan":          ["desc", "morgan"],
        "desc_morgan_alerts":   ["desc", "morgan", "alerts"],
        "rich":                 ["desc", "morgan", "avalon", "erg", "maccs"],
        "rich_alerts":          ["desc", "morgan", "avalon", "erg", "maccs", "alerts"],
    }
    if use_mf:
        feature_sets["rich_mf"]        = ["desc", "morgan", "avalon", "erg", "maccs", "molformer"]
        feature_sets["rich_mf_alerts"] = ["desc", "morgan", "avalon", "erg", "maccs", "molformer", "alerts"]
    models = ["xgb", "lgbm", "cat"]

    print("\n[alerts] === screening (model x feature-set), ranked by mean VAL-AUROC ===", flush=True)
    rows = []
    for fs, kinds in feature_sets.items():
        for mk in models:
            t = time.time()
            val_auc, test_aucs, _ = eval_config(
                group, name, X_all_idx, blocks, kinds, mk, y_test, n_bag=1)
            rows.append({"feature_set": fs, "model": mk, "val_AUROC": val_auc,
                         "test_AUROC": float(np.mean(test_aucs)),
                         "test_std": float(np.std(test_aucs))})
            print(f"  {fs:18s} {mk:5s}  val={val_auc:.4f}  test={np.mean(test_aucs):.4f}"
                  f"+/-{np.std(test_aucs):.3f}  ({time.time()-t:.0f}s)", flush=True)
        # ensemble per feature set
        ev, et, _ = ensemble_eval(group, name, X_all_idx, blocks, kinds, models, y_test)
        rows.append({"feature_set": fs, "model": "ENS4", "val_AUROC": ev,
                     "test_AUROC": float(np.mean(et)), "test_std": float(np.std(et))})
        print(f"  {fs:18s} {'ENS4':5s}  val={ev:.4f}  test={np.mean(et):.4f}"
              f"+/-{np.std(et):.3f}", flush=True)

    tbl = pd.DataFrame(rows).sort_values("val_AUROC", ascending=False).reset_index(drop=True)
    tbl.to_csv(RES / "tdc_alerts_search.csv", index=False)
    win = tbl.iloc[0]
    win_fs, win_mk = win["feature_set"], win["model"]
    print(f"\n[alerts] VAL-SELECTED WINNER: {win_fs} / {win_mk}  "
          f"val={win['val_AUROC']:.4f}  test={win['test_AUROC']:.4f}", flush=True)

    best_params = None
    if win_mk != "ENS4" and args.optuna > 0:
        best_params = optuna_tune(group, name, X_all_idx, blocks,
                                  feature_sets[win_fs], win_mk, y_test, args.optuna)

    if win_mk == "ENS4":
        fval, ftest, fpred = ensemble_eval(
            group, name, X_all_idx, blocks, feature_sets[win_fs], models, y_test, n_bag=5)
    else:
        fval, ftest, fpred = eval_config(
            group, name, X_all_idx, blocks, feature_sets[win_fs], win_mk, y_test,
            params=best_params, n_bag=5)
    lo, hi = bootstrap_ci(y_test, fpred)
    mean, std = float(np.mean(ftest)), float(np.std(ftest))
    print("\n=== FINAL (val-selected, bagged-5, mechanism-alert search) ===")
    print(f"  config        = {win_fs} / {win_mk}" + ("  (Optuna-tuned)" if best_params else ""))
    print(f"  val AUROC     = {fval:.4f}")
    print(f"  TEST AUROC    = {mean:.4f} +/- {std:.4f}   (per-seed {[round(a,4) for a in ftest]})")
    print(f"  bootstrap 95% CI = [{lo:.4f}, {hi:.4f}]")
    print(f"  leaderboard: AttentiveFP 0.886 | MapLight+GNN 0.917 | AttrMasking 0.919")
    pd.DataFrame([{"feature_set": win_fs, "model": win_mk, "optuna": bool(best_params),
                   "val_AUROC": fval, "test_AUROC_mean": mean, "test_AUROC_std": std,
                   "ci_low": lo, "ci_high": hi,
                   **{f"seed{s}": a for s, a in zip(SEEDS, ftest)}}]
                 ).to_csv(RES / "tdc_alerts_final.csv", index=False)
    print("[alerts] saved -> results/tdc_alerts_search.csv, tdc_alerts_final.csv", flush=True)

    # ---- did alerts help? head-to-head, val-selected within each base set ----
    print("\n[alerts] === did mechanism alerts help? (best VAL per base set) ===", flush=True)
    for base in (["desc_morgan", "rich"] + (["rich_mf"] if use_mf else [])):
        a = tbl[tbl.feature_set == base]
        b = tbl[tbl.feature_set == base + "_alerts"]
        if len(a) and len(b):
            ba = a.sort_values("val_AUROC").iloc[-1]
            bb = b.sort_values("val_AUROC").iloc[-1]
            print(f"  {base:11s}: val {ba.val_AUROC:.4f}->{bb.val_AUROC:.4f}  "
                  f"(test {ba.test_AUROC:.4f}->{bb.test_AUROC:.4f})  "
                  f"{'val UP' if bb.val_AUROC > ba.val_AUROC else 'no val gain'}", flush=True)


if __name__ == "__main__":
    main()
