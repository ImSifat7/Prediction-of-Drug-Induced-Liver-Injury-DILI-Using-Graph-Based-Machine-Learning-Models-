"""Scaffold-grouped k-fold cross-validation + class-distribution parity check.

A pure scaffold split puts whole scaffold groups into one fold (no leakage). To also
keep class ratios roughly equal across folds (statistical parity in the splits),
we greedily place each scaffold group into the fold whose current positive ratio is
most distant from the global positive ratio.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple
import random

from torch_geometric.data import Data

from .data_utils import murcko_scaffold


def scaffold_kfold(
    graphs: List[Data],
    n_splits: int = 5,
    seed: int = 42,
) -> List[Tuple[List[int], List[int]]]:
    """Group-aware k-fold: scaffolds always stay in one fold; folds class-balanced.

    Returns: list of (train_idx, test_idx) pairs.
    """
    rng = random.Random(seed)

    scaffolds: dict[str, list[int]] = defaultdict(list)
    labels = [int(g.y.item()) for g in graphs]
    for idx, g in enumerate(graphs):
        scaffolds[murcko_scaffold(g.smiles)].append(idx)

    groups = list(scaffolds.values())
    rng.shuffle(groups)
    groups.sort(key=lambda xs: len(xs), reverse=True)

    global_pos = sum(labels) / len(labels)

    fold_idx: List[List[int]] = [[] for _ in range(n_splits)]
    fold_pos: List[int] = [0] * n_splits

    for grp in groups:
        grp_pos = sum(labels[i] for i in grp)
        scores = []
        for f in range(n_splits):
            n_after = len(fold_idx[f]) + len(grp)
            pos_after = fold_pos[f] + grp_pos
            ratio_after = pos_after / max(n_after, 1)
            score = (n_after, abs(ratio_after - global_pos))
            scores.append((score, f))
        _, best = min(scores)
        fold_idx[best].extend(grp)
        fold_pos[best] += grp_pos

    splits: List[Tuple[List[int], List[int]]] = []
    for f in range(n_splits):
        test_idx = sorted(fold_idx[f])
        train_idx = sorted([j for k, fold in enumerate(fold_idx) if k != f for j in fold])
        splits.append((train_idx, test_idx))
    return splits


def class_distribution(graphs: List[Data], indices: List[int]) -> dict:
    labels = [int(graphs[i].y.item()) for i in indices]
    pos = sum(labels)
    n = len(labels)
    return {
        "n": n,
        "pos": pos,
        "neg": n - pos,
        "pos_ratio": pos / max(n, 1),
    }


def parity_table(graphs: List[Data], splits) -> List[dict]:
    """Class distribution per (fold, split) — for reporting that train/test parity holds."""
    rows = []
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        rows.append({"fold": fold_i, "split": "train", **class_distribution(graphs, train_idx)})
        rows.append({"fold": fold_i, "split": "test",  **class_distribution(graphs, test_idx)})
    return rows


def parity_summary(rows: List[dict]) -> str:
    lines = ["fold split    n   pos   neg  pos_ratio"]
    for r in rows:
        lines.append(f"{r['fold']:>4} {r['split']:<5} {r['n']:>4} {r['pos']:>4} {r['neg']:>4}    {r['pos_ratio']:.3f}")
    return "\n".join(lines)
