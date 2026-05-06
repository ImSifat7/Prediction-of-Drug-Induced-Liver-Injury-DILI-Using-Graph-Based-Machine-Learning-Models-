"""Hyperparameter tuning per model — both Optuna (TPE) and Grid Search.

Important: tuning uses ONLY the (train, val) split. The outer test set is held out
and is never seen by Optuna or by any retraining decision. This is what guarantees
"no data leakage" between hyperparameter selection and final evaluation.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional

import optuna
import torch
from torch_geometric.data import Data

from .models import build_model
from .train_utils import (
    compute_pos_weight,
    make_loader,
    set_seed,
    train_model,
    evaluate,
)


def _train_eval_with_params(
    graphs: List[Data],
    train_idx: List[int],
    val_idx: List[int],
    model_name: str,
    params: dict,
    device,
    epochs: int,
    patience: int,
    seed: int,
) -> float:
    set_seed(seed)
    in_dim = graphs[0].x.shape[1]
    edge_dim = graphs[0].edge_attr.shape[1]
    model = build_model(model_name, in_dim, edge_dim, params)
    train_loader = make_loader(graphs, train_idx, batch_size=params["batch_size"], shuffle=True)
    val_loader = make_loader(graphs, val_idx, batch_size=params["batch_size"], shuffle=False)
    pos_w = compute_pos_weight(graphs, train_idx)
    model, _ = train_model(
        model, train_loader, val_loader,
        lr=params["lr"], weight_decay=params["weight_decay"],
        optimizer_name=params["optimizer_name"], pos_weight=pos_w,
        epochs=epochs, patience=patience, device=device,
    )
    res = evaluate(model, val_loader, device)
    return float(res["AUROC"])


# --- Optuna (TPE) ---------------------------------------------------------

def make_optuna_objective(
    graphs: List[Data],
    train_idx: List[int],
    val_idx: List[int],
    model_name: str,
    device,
    epochs: int,
    patience: int,
    seed: int,
):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128]),
            "num_layers": trial.suggest_int("num_layers", 2, 4),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "optimizer_name": trial.suggest_categorical("optimizer_name", ["adam", "adamw"]),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        }
        if model_name == "GAT":
            params["heads"] = trial.suggest_categorical("heads", [2, 4, 8])
        if model_name == "MPNN":
            params["num_steps"] = params["num_layers"]  # alias
        if model_name == "GraphSAGE":
            params["aggr"] = trial.suggest_categorical("aggr", ["mean", "max"])
        try:
            return _train_eval_with_params(
                graphs, train_idx, val_idx, model_name, params,
                device=device, epochs=epochs, patience=patience, seed=seed,
            )
        except Exception as e:
            print(f"  [trial fail] {model_name}: {e}")
            raise optuna.TrialPruned()

    return objective


def run_optuna(
    graphs: List[Data],
    train_idx: List[int],
    val_idx: List[int],
    model_name: str,
    n_trials: int = 25,
    device="cpu",
    epochs: int = 80,
    patience: int = 12,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> dict:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner,
                                study_name=f"{model_name}_tpe")
    obj = make_optuna_objective(graphs, train_idx, val_idx, model_name, device, epochs, patience, seed)
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False, gc_after_trial=True)
    out = {"method": "optuna_tpe", "best_value": float(study.best_value), "params": study.best_params}
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / f"{model_name}_optuna.json").write_text(json.dumps(out, indent=2))
    return out


# --- Grid search ----------------------------------------------------------

def grid_for(model_name: str) -> List[dict]:
    """Small explicit grid covering optimizer / lr / weight_decay axes (the focus
    of the instructor's request). Architecture knobs use sensible defaults so the
    grid stays tractable; Optuna handles the wider sweep."""
    grid = []
    base_combos = list(itertools.product(
        ["adam", "adamw", "sgd"],     # optimizer
        [1e-3, 5e-4, 1e-4],           # lr
        [0.0, 1e-4, 1e-3],            # weight decay
    ))
    for opt, lr, wd in base_combos:
        params = {
            "lr": lr,
            "weight_decay": wd,
            "hidden_dim": 64,
            "num_layers": 3,
            "dropout": 0.3,
            "optimizer_name": opt,
            "batch_size": 32,
        }
        if model_name == "GAT":
            params["heads"] = 4
        if model_name == "GraphSAGE":
            params["aggr"] = "max"
        if model_name == "MPNN":
            params["num_steps"] = 3
        grid.append(params)
    return grid


def run_grid_search(
    graphs: List[Data],
    train_idx: List[int],
    val_idx: List[int],
    model_name: str,
    device="cpu",
    epochs: int = 60,
    patience: int = 10,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> dict:
    grid = grid_for(model_name)
    best = {"value": -1.0, "params": None}
    for params in grid:
        try:
            val_auc = _train_eval_with_params(
                graphs, train_idx, val_idx, model_name, params,
                device=device, epochs=epochs, patience=patience, seed=seed,
            )
        except Exception as e:
            print(f"  [grid fail] {model_name}: {e}")
            continue
        if val_auc > best["value"]:
            best = {"value": float(val_auc), "params": params}
    out = {"method": "grid_search", "best_value": best["value"], "params": best["params"], "n_combos": len(grid)}
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / f"{model_name}_grid.json").write_text(json.dumps(out, indent=2))
    return out


def pick_best(optuna_res: dict, grid_res: dict) -> dict:
    """Choose between Optuna and Grid by validation AUROC."""
    if optuna_res["best_value"] >= grid_res["best_value"]:
        return {"method": optuna_res["method"], "params": optuna_res["params"], "best_value": optuna_res["best_value"]}
    return {"method": grid_res["method"], "params": grid_res["params"], "best_value": grid_res["best_value"]}
