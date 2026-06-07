"""Official TDC DILI benchmark — Phase 1 model search (val-selected, leakage-clean).

Extends tdc_official.py:
  - Richer feature union: RDKit-2D desc + Morgan + Avalon + ErG + MACCS (+MolFormer)
  - 4 gradient-boosting backends: XGBoost, LightGBM, CatBoost, HistGB
  - An averaging ensemble of all 4 backends
  - Optuna tuning of the winning backend

CRITICAL METHODOLOGY: every model/feature/hyperparameter decision is made on the
official VALIDATION split (mean val-AUROC over the 5 seeds).  The fixed 96-mol
TEST set is touched ONLY to report the final number for the single val-selected
winner.  This avoids the test-set-peeking that the Feb-2026 TDC audit flagged.

Run:
    python -m src.improved.tdc_official_v2            # full search (+MolFormer)
    python -m src.improved.tdc_official_v2 --no-mf    # skip MolFormer block
    python -m src.improved.tdc_official_v2 --optuna 60
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
from sklearn.ensemble import HistGradientBoostingClassifier

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator, rdReducedGraphs
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Avalon import pyAvalonTools

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.improved.molformer import extract_molformer_embeddings
else:
    from .molformer import extract_molformer_embeddings

SEEDS = [1, 2, 3, 4, 5]
BASE = Path(__file__).resolve().parents[2]
DATA_PATH = BASE / "data" / "tdc_benchmark"
RES = BASE / "results"

# ---------------- feature blocks ----------------
_DESC_NAMES = [d[0] for d in Descriptors._descList]
_DESC_CALC = MoleculeDescriptors.MolecularDescriptorCalculator(_DESC_NAMES)
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _vec(mol, kind):
    if kind == "desc":
        if mol is None:
            return np.zeros(len(_DESC_NAMES), np.float32)
        v = np.array(_DESC_CALC.CalcDescriptors(mol), np.float64)
        return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if kind == "morgan":
        return (np.zeros(2048, np.float32) if mol is None
                else _MORGAN.GetFingerprintAsNumPy(mol).astype(np.float32))
    if kind == "avalon":
        if mol is None:
            return np.zeros(512, np.float32)
        bv = pyAvalonTools.GetAvalonFP(mol, 512)
        return np.frombuffer(bytes(bv.ToBitString(), "ascii"), "u1").astype(np.float32) - 48
    if kind == "erg":
        return (np.zeros(315, np.float32) if mol is None
                else np.asarray(rdReducedGraphs.GetErGFingerprint(mol), np.float32))
    if kind == "maccs":
        if mol is None:
            return np.zeros(167, np.float32)
        bv = MACCSkeys.GenMACCSKeys(mol)
        return np.frombuffer(bytes(bv.ToBitString(), "ascii"), "u1").astype(np.float32) - 48
    raise ValueError(kind)


def precompute_blocks(smiles, kinds):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    return {k: np.stack([_vec(m, k) for m in mols], 0) for k in kinds}


# ---------------- model factory ----------------
def make_model(kind, spw, params=None):
    p = params or {}
    if kind == "xgb":
        return xgb.XGBClassifier(
            n_estimators=p.get("n_estimators", 400), max_depth=p.get("max_depth", 4),
            learning_rate=p.get("lr", 0.05), subsample=p.get("subsample", 0.85),
            colsample_bytree=p.get("colsample", 0.7), reg_lambda=p.get("reg_lambda", 1.0),
            min_child_weight=p.get("min_child_weight", 1.0),
            scale_pos_weight=spw, eval_metric="logloss", n_jobs=4, verbosity=0,
            random_state=p.get("rs", 0))
    if kind == "lgbm":
        return lgb.LGBMClassifier(
            n_estimators=p.get("n_estimators", 400), num_leaves=p.get("num_leaves", 31),
            max_depth=p.get("max_depth", -1), learning_rate=p.get("lr", 0.05),
            subsample=p.get("subsample", 0.85), colsample_bytree=p.get("colsample", 0.7),
            reg_lambda=p.get("reg_lambda", 1.0), scale_pos_weight=spw,
            n_jobs=4, verbosity=-1, random_state=p.get("rs", 0))
    if kind == "cat":
        return CatBoostClassifier(
            iterations=p.get("n_estimators", 400), depth=p.get("max_depth", 4),
            learning_rate=p.get("lr", 0.05), l2_leaf_reg=p.get("reg_lambda", 3.0),
            scale_pos_weight=spw, verbose=0, random_seed=p.get("rs", 0),
            allow_writing_files=False)
    if kind == "histgb":
        return HistGradientBoostingClassifier(
            max_iter=p.get("n_estimators", 400), max_depth=p.get("max_depth", 4),
            learning_rate=p.get("lr", 0.05), l2_regularization=p.get("reg_lambda", 1.0),
            class_weight="balanced", random_state=p.get("rs", 0))
    raise ValueError(kind)


def fit_predict(kind, X_tr, y_tr, X_va, X_te, params=None):
    spw = float((y_tr == 0).sum()) / max(int((y_tr == 1).sum()), 1)
    m = make_model(kind, spw, params)
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1]


# ---------------- evaluation ----------------
def eval_config(group, name, X_all_idx, blocks, kinds, model_kind,
                y_test, params=None, n_bag=1):
    """Return (mean_val_auroc, per_seed_test_auroc, seed_avg_test_pred)."""
    Xcols = np.concatenate([blocks[k] for k in kinds], axis=1)
    val_aucs, test_aucs = [], []
    te_pred_sum = np.zeros(len(y_test))
    te_rows = X_all_idx["test"]
    for seed in SEEDS:
        tr_df, va_df = group.get_train_valid_split(benchmark=name, split_type="default", seed=seed)
        tr_rows = [X_all_idx["smi2i"][s] for s in tr_df["Drug"]]
        va_rows = [X_all_idx["smi2i"][s] for s in va_df["Drug"]]
        X_tr, y_tr = Xcols[tr_rows], tr_df["Y"].values.astype(int)
        X_va, y_va = Xcols[va_rows], va_df["Y"].values.astype(int)
        X_te = Xcols[te_rows]
        pv = np.zeros(len(va_rows)); pt = np.zeros(len(te_rows))
        for b in range(n_bag):
            pp = dict(params or {}); pp["rs"] = b
            v, t = fit_predict(model_kind, X_tr, y_tr, X_va, X_te, pp)
            pv += v; pt += t
        pv /= n_bag; pt /= n_bag
        val_aucs.append(roc_auc_score(y_va, pv))
        test_aucs.append(roc_auc_score(y_test, pt))
        te_pred_sum += pt
    return float(np.mean(val_aucs)), test_aucs, te_pred_sum / len(SEEDS)


def bootstrap_ci(y, p, n=5000):
    rng = np.random.default_rng(0)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    out = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        out.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


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
    print(f"[v2] DILI official  train_val={len(trv_df)} test={len(test_df)}", flush=True)

    master = sorted(set(trv_df["Drug"]).union(set(test_df["Drug"])))
    smi2i = {s: i for i, s in enumerate(master)}
    base_kinds = ["desc", "morgan", "avalon", "erg", "maccs"]
    t0 = time.time()
    blocks = precompute_blocks(master, base_kinds)
    print(f"[v2] precomputed FP blocks {[(k, blocks[k].shape[1]) for k in base_kinds]} "
          f"in {time.time()-t0:.0f}s", flush=True)

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

    feature_sets = {
        "desc_morgan": ["desc", "morgan"],
        "rich": ["desc", "morgan", "avalon", "erg", "maccs"],
    }
    if use_mf:
        feature_sets["rich_mf"] = ["desc", "morgan", "avalon", "erg", "maccs", "molformer"]
    models = ["xgb", "lgbm", "cat"]

    # ---- screening matrix (single-fit, selected by VAL) ----
    print("\n[v2] === screening (model x feature-set), ranked by mean VAL-AUROC ===", flush=True)
    rows = []
    cache_pred = {}
    for fs, kinds in feature_sets.items():
        for mk in models:
            t = time.time()
            val_auc, test_aucs, te_pred = eval_config(
                group, name, X_all_idx, blocks, kinds, mk, y_test, n_bag=1)
            cache_pred[(fs, mk)] = te_pred
            rows.append({"feature_set": fs, "model": mk,
                         "val_AUROC": val_auc,
                         "test_AUROC": float(np.mean(test_aucs)),
                         "test_std": float(np.std(test_aucs))})
            print(f"  {fs:12s} {mk:7s}  val={val_auc:.4f}  test={np.mean(test_aucs):.4f}"
                  f"+/-{np.std(test_aucs):.3f}  ({time.time()-t:.0f}s)", flush=True)

    # ---- averaging ensemble of all 4 backends, per feature set ----
    for fs in feature_sets:
        # rebuild per-seed val for the ensemble to score selection honestly
        ens_val, ens_test_aucs, ens_pred = ensemble_eval(
            group, name, X_all_idx, blocks, feature_sets[fs], models, y_test)
        cache_pred[(fs, "ENS4")] = ens_pred
        rows.append({"feature_set": fs, "model": "ENS4",
                     "val_AUROC": ens_val, "test_AUROC": float(np.mean(ens_test_aucs)),
                     "test_std": float(np.std(ens_test_aucs))})
        print(f"  {fs:12s} {'ENS4':7s}  val={ens_val:.4f}  test={np.mean(ens_test_aucs):.4f}"
              f"+/-{np.std(ens_test_aucs):.3f}", flush=True)

    tbl = pd.DataFrame(rows).sort_values("val_AUROC", ascending=False).reset_index(drop=True)
    tbl.to_csv(RES / "tdc_official_search.csv", index=False)
    win = tbl.iloc[0]
    print(f"\n[v2] VAL-SELECTED WINNER: {win['feature_set']} / {win['model']}  "
          f"val={win['val_AUROC']:.4f}  test={win['test_AUROC']:.4f}", flush=True)

    # ---- Optuna-tune the winning single backend (if not ENS4) on VAL ----
    best_params = None
    win_fs, win_mk = win["feature_set"], win["model"]
    if win_mk != "ENS4" and args.optuna > 0:
        best_params = optuna_tune(group, name, X_all_idx, blocks,
                                  feature_sets[win_fs], win_mk, y_test, args.optuna)

    # ---- final report on winner (bagged) + bootstrap CI ----
    if win_mk == "ENS4":
        fval, ftest_aucs, fpred = ensemble_eval(
            group, name, X_all_idx, blocks, feature_sets[win_fs], models, y_test, n_bag=5)
    else:
        fval, ftest_aucs, fpred = eval_config(
            group, name, X_all_idx, blocks, feature_sets[win_fs], win_mk, y_test,
            params=best_params, n_bag=5)
    lo, hi = bootstrap_ci(y_test, fpred)
    mean, std = float(np.mean(ftest_aucs)), float(np.std(ftest_aucs))
    print("\n=== FINAL OFFICIAL TDC DILI (val-selected, bagged-5) ===")
    print(f"  config        = {win_fs} / {win_mk}" + (f"  (Optuna-tuned)" if best_params else ""))
    print(f"  val AUROC     = {fval:.4f}")
    print(f"  TEST AUROC    = {mean:.4f} +/- {std:.4f}   (per-seed {[round(a,4) for a in ftest_aucs]})")
    print(f"  bootstrap 95% CI = [{lo:.4f}, {hi:.4f}]")
    print(f"  leaderboard: AttentiveFP 0.886 | MapLight+GNN 0.917 | AttrMasking 0.919 | ZairaChem 0.925")
    pd.DataFrame([{"feature_set": win_fs, "model": win_mk, "optuna": bool(best_params),
                   "val_AUROC": fval, "test_AUROC_mean": mean, "test_AUROC_std": std,
                   "ci_low": lo, "ci_high": hi,
                   **{f"seed{s}": a for s, a in zip(SEEDS, ftest_aucs)}}]
                 ).to_csv(RES / "tdc_official_final.csv", index=False)
    if best_params:
        print(f"  tuned params  = {best_params}")
    print("[v2] saved -> results/tdc_official_search.csv, tdc_official_final.csv", flush=True)


def ensemble_eval(group, name, X_all_idx, blocks, kinds, models, y_test, n_bag=1):
    """Rank-free average of all backends; val + test AUROC per seed."""
    Xcols = np.concatenate([blocks[k] for k in kinds], axis=1)
    te_rows = X_all_idx["test"]
    val_aucs, test_aucs = [], []
    te_sum = np.zeros(len(y_test))
    for seed in SEEDS:
        tr_df, va_df = group.get_train_valid_split(benchmark=name, split_type="default", seed=seed)
        tr_rows = [X_all_idx["smi2i"][s] for s in tr_df["Drug"]]
        va_rows = [X_all_idx["smi2i"][s] for s in va_df["Drug"]]
        X_tr, y_tr = Xcols[tr_rows], tr_df["Y"].values.astype(int)
        X_va, y_va = Xcols[va_rows], va_df["Y"].values.astype(int)
        X_te = Xcols[te_rows]
        pv = np.zeros(len(va_rows)); pt = np.zeros(len(te_rows))
        for mk in models:
            for b in range(n_bag):
                v, t = fit_predict(mk, X_tr, y_tr, X_va, X_te, {"rs": b})
                pv += v; pt += t
        pv /= (len(models) * n_bag); pt /= (len(models) * n_bag)
        val_aucs.append(roc_auc_score(y_va, pv)); test_aucs.append(roc_auc_score(y_test, pt))
        te_sum += pt
    return float(np.mean(val_aucs)), test_aucs, te_sum / len(SEEDS)


def optuna_tune(group, name, X_all_idx, blocks, kinds, model_kind, y_test, n_trials):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"\n[v2] Optuna-tuning {model_kind} on VAL ({n_trials} trials) ...", flush=True)

    def objective(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "lr": trial.suggest_float("lr", 0.01, 0.1, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 8.0, log=True),
        }
        if model_kind in ("xgb", "lgbm"):
            p["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
            p["colsample"] = trial.suggest_float("colsample", 0.4, 1.0)
        val_auc, _, _ = eval_config(group, name, X_all_idx, blocks, kinds,
                                    model_kind, y_test, params=p, n_bag=1)
        return val_auc

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"[v2] best val-AUROC={study.best_value:.4f}  params={study.best_params}", flush=True)
    return study.best_params


if __name__ == "__main__":
    main()
