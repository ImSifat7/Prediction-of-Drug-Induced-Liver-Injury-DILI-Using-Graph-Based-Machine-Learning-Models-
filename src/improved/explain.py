"""GNNExplainer atom-level attribution for DILI predictions (advisor's plan, Step 5).

Produces toxophore heatmaps: for each chosen molecule, retrains the best GNN
quickly with a fixed seed, runs GNNExplainer to obtain per-atom importance
masks, and renders an RDKit 2D depiction with atoms colored by attribution.

Why this matters for the thesis:
- The plan asks for "5 drugs where the model correctly identifies known toxic
  groups (e.g., aromatic rings or specific aromatic amines)".
- We pick 5 DILI-positive molecules from the test set where the model's
  prediction is correct and confident, then visualize *which atoms* drove the
  decision. Reviewers can verify whether the highlighted region matches known
  toxicophores (aromatic amines, nitro groups, halogenated aromatics, etc.).

Run:
    python -m src.improved.explain --model GIN --top 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.improved.data_utils import load_dataset, scaffold_split
    from src.improved.models import build_model
    from src.improved.train_utils import (
        compute_pos_weight, evaluate, make_loader, set_seed, train_model,
    )
else:
    from .data_utils import load_dataset, scaffold_split
    from .models import build_model
    from .train_utils import (
        compute_pos_weight, evaluate, make_loader, set_seed, train_model,
    )


def pick_correct_positives(graphs, test_idx, probs, top: int = 5):
    """Pick DILI-positive test molecules with highest confidence + correct prediction.

    These are the "easy positives" — exactly what we want for the figure, since
    we're asking 'does the model see the right substructure when it's sure?'
    """
    rows = []
    for i, t_idx in enumerate(test_idx):
        g = graphs[t_idx]
        if int(g.y.item()) == 1 and probs[i] >= 0.6:
            rows.append((float(probs[i]), t_idx))
    rows.sort(reverse=True)
    return [t for _, t in rows[:top]]


class _ExplainerAdapter(torch.nn.Module):
    """Wrap a model that expects `forward(data)` into one accepting positional
    args (`x, edge_index, batch=..., edge_attr=..., u=...`), which is the
    signature PyG's Explainer uses internally."""

    def __init__(self, model, u_for_single_graph: torch.Tensor):
        super().__init__()
        self.model = model
        # Stored so we can re-inject `u` when Explainer drops it during masking.
        self._u = u_for_single_graph  # shape [1, desc_dim]

    def forward(self, x, edge_index, batch=None, edge_attr=None, **_):
        class _D:
            pass
        d = _D()
        d.x = x
        d.edge_index = edge_index
        d.edge_attr = edge_attr
        d.batch = batch if batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        d.u = self._u.to(x.device)
        return self.model(d)


def explain_one(model, graph, device):
    """Return per-atom importance (length = n_atoms) using GNNExplainer."""
    from torch_geometric.explain import Explainer, GNNExplainer
    model = model.to(device).eval()
    wrapped = _ExplainerAdapter(model, graph.u).to(device).eval()
    explainer = Explainer(
        model=wrapped,
        algorithm=GNNExplainer(epochs=200, lr=0.01),
        explanation_type="model",
        node_mask_type="object",
        edge_mask_type=None,
        model_config=dict(mode="binary_classification", task_level="graph", return_type="raw"),
    )
    g = graph.clone().to(device)
    batch = torch.zeros(g.x.size(0), dtype=torch.long, device=device)
    explanation = explainer(g.x, g.edge_index, batch=batch, edge_attr=g.edge_attr)
    mask = explanation.node_mask.detach().cpu().numpy().reshape(-1)
    return mask


def render_attribution(smiles: str, importance: np.ndarray, out_path: Path, title: str):
    """Render molecule with atoms colored by GNNExplainer importance."""
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    # Normalize importance to [0,1] for color intensity.
    imp = importance[: mol.GetNumAtoms()]
    if imp.max() - imp.min() > 1e-8:
        imp_n = (imp - imp.min()) / (imp.max() - imp.min())
    else:
        imp_n = np.zeros_like(imp)
    # Color top-30% atoms red, scaled by intensity.
    threshold = np.quantile(imp_n, 0.70) if len(imp_n) > 3 else 0.5
    atom_colors = {}
    highlight = []
    for i, v in enumerate(imp_n):
        if v >= threshold:
            highlight.append(i)
            # red → orange gradient based on intensity
            atom_colors[i] = (1.0, 1.0 - float(v) * 0.7, 1.0 - float(v) * 0.7)
    drawer = Draw.MolDraw2DCairo(500, 500)
    drawer.drawOptions().addAtomIndices = False
    drawer.drawOptions().legendFontSize = 16
    Draw.PrepareAndDrawMolecule(
        drawer, mol,
        highlightAtoms=highlight,
        highlightAtomColors=atom_colors,
        legend=title,
    )
    drawer.FinishDrawing()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(drawer.GetDrawingText())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="GIN", help="Which GNN to explain (default: GIN — best by tuned AUROC)")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base = Path(__file__).resolve().parents[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[explain] device={device} model={args.model}", flush=True)

    graphs = load_dataset(base / "data" / "dili_clean.csv")
    in_dim = graphs[0].x.shape[1]
    edge_dim = graphs[0].edge_attr.shape[1]
    desc_dim = graphs[0].u.shape[1] if hasattr(graphs[0], "u") and graphs[0].u is not None else 0

    train_idx, val_idx, test_idx = scaffold_split(graphs, 0.6, 0.2, 0.2, seed=42)

    cfg = json.loads((base / "configs" / f"{args.model}_best.json").read_text())["params"]
    set_seed(args.seed)
    model = build_model(args.model, in_dim, edge_dim, cfg, desc_dim=desc_dim)
    tr_loader = make_loader(graphs, train_idx, batch_size=cfg["batch_size"], shuffle=True)
    va_loader = make_loader(graphs, val_idx, batch_size=cfg["batch_size"], shuffle=False)
    te_loader = make_loader(graphs, test_idx, batch_size=64, shuffle=False)
    pos_w = compute_pos_weight(graphs, train_idx)
    print("[explain] training one seed for attribution model…", flush=True)
    model, best_val = train_model(
        model, tr_loader, va_loader,
        lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        optimizer_name=cfg["optimizer_name"], pos_weight=pos_w,
        epochs=args.epochs, patience=15, device=device,
    )
    res = evaluate(model, te_loader, device)
    print(f"[explain] held-out test AUROC={res['AUROC']:.4f}  best_val={best_val:.4f}", flush=True)

    chosen = pick_correct_positives(graphs, test_idx, res["y_prob"], top=args.top)
    print(f"[explain] selected {len(chosen)} confident-correct positive molecules", flush=True)

    out_dir = base / "results" / "explain"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, idx in enumerate(chosen):
        g = graphs[idx]
        mask = explain_one(model, g, device)
        prob = float(res["y_prob"][test_idx.index(idx)])
        title = f"{args.model}  p(DILI)={prob:.2f}  label=POS"
        out_path = out_dir / f"toxophore_{args.model}_{k+1:02d}.png"
        render_attribution(g.smiles, mask, out_path, title)
        rows.append({"rank": k + 1, "smiles": g.smiles, "prob": prob, "n_atoms": len(mask), "out": str(out_path.name)})
        print(f"  {k+1}. p={prob:.3f}  atoms={len(mask)}  -> {out_path.name}", flush=True)

    import csv
    summary_csv = out_dir / "attributions_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[explain] wrote {len(rows)} heatmaps + summary to {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
