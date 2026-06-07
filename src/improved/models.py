"""Improved GNN architectures with explicit per-model layer changes vs the baselines.

What changed for each model (defendable in the thesis):

  GCN        -> deeper (3 layers) + BatchNorm + Dropout + JumpingKnowledge readout
  GAT        -> GATv2 (instead of GAT) + multi-head + residual + BatchNorm + 3 layers
  GraphSAGE  -> max-aggregator (instead of mean) + 3 layers + residual connections
  GIN        -> GINEConv (uses bond features) + global_add_pool + MLP head
  MPNN       -> 3 message-passing steps (shared NNConv) + GRU update + Set2Set readout

Global descriptor fusion (NEW):
  All five models concatenate per-molecule RDKit descriptors (LogP, MW, TPSA, ...)
  with the pooled graph embedding before the final classifier head. This gives the
  GNN access to the same physicochemical features that make classical RF
  competitive on DILI, while the message-passing layers still learn local
  substructure. Controlled by `desc_dim` — pass 0 to disable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GATv2Conv,
    SAGEConv,
    GINEConv,
    NNConv,
    JumpingKnowledge,
    BatchNorm,
    Set2Set,
    global_mean_pool,
    global_add_pool,
)


# --- Descriptor fusion helper --------------------------------------------

class DescriptorMLP(nn.Module):
    """Small MLP that lifts the 10-dim global descriptors into the model's
    hidden space so they can be added/concatenated with the graph embedding."""

    def __init__(self, desc_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(desc_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.net(u)


def _maybe_get_u(data) -> torch.Tensor | None:
    """Return per-graph descriptor tensor [B, desc_dim] or None if absent.

    PyG's Batch collation stacks `data.u` tensors along dim 0 when each is
    [1, D], yielding [B, D] in the batched object — exactly what we want.
    """
    u = getattr(data, "u", None)
    if u is None:
        return None
    if u.dim() == 1:
        u = u.unsqueeze(0)
    return u


# --- GCN ------------------------------------------------------------------

class GCNNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 3,
                 dropout: float = 0.3, desc_dim: int = 0, **_):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.desc_dim = desc_dim
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        self.bns.append(BatchNorm(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(BatchNorm(hidden_dim))
        self.jk = JumpingKnowledge("cat", hidden_dim, num_layers)
        graph_dim = hidden_dim * num_layers
        if desc_dim > 0:
            self.desc_mlp = DescriptorMLP(desc_dim, hidden_dim)
            head_in = graph_dim + hidden_dim
        else:
            self.desc_mlp = None
            head_in = graph_dim
        self.fc = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        xs = []
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, ei)))
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)
        x = self.jk(xs)
        x = global_mean_pool(x, batch)
        if self.desc_mlp is not None:
            u = _maybe_get_u(data)
            if u is not None:
                x = torch.cat([x, self.desc_mlp(u)], dim=1)
        return self.fc(x).view(-1)


# --- GAT ------------------------------------------------------------------

class GATNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, heads: int = 4, num_layers: int = 3,
                 dropout: float = 0.3, desc_dim: int = 0, **_):
        super().__init__()
        self.dropout = dropout
        self.desc_dim = desc_dim
        per_head = max(hidden_dim // heads, 8)
        out_dim = per_head * heads
        self.in_proj = nn.Linear(in_dim, out_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GATv2Conv(out_dim, per_head, heads=heads, dropout=dropout, concat=True))
            self.bns.append(BatchNorm(out_dim))
        if desc_dim > 0:
            self.desc_mlp = DescriptorMLP(desc_dim, hidden_dim)
            head_in = out_dim + hidden_dim
        else:
            self.desc_mlp = None
            head_in = out_dim
        self.fc = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        x = self.in_proj(x)
        for conv, bn in zip(self.convs, self.bns):
            res = x
            x = F.elu(bn(conv(x, ei)))
            x = x + res  # residual
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        if self.desc_mlp is not None:
            u = _maybe_get_u(data)
            if u is not None:
                x = torch.cat([x, self.desc_mlp(u)], dim=1)
        return self.fc(x).view(-1)


# --- GraphSAGE ------------------------------------------------------------

class GraphSAGENet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 3, dropout: float = 0.3,
                 aggr: str = "max", desc_dim: int = 0, **_):
        super().__init__()
        self.dropout = dropout
        self.desc_dim = desc_dim
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim, aggr=aggr))
        self.bns.append(BatchNorm(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggr))
            self.bns.append(BatchNorm(hidden_dim))
        if desc_dim > 0:
            self.desc_mlp = DescriptorMLP(desc_dim, hidden_dim)
            head_in = hidden_dim + hidden_dim
        else:
            self.desc_mlp = None
            head_in = hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        prev = None
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, ei)))
            if prev is not None and prev.shape == x.shape:
                x = x + prev  # residual after first layer
            x = F.dropout(x, p=self.dropout, training=self.training)
            prev = x
        x = global_mean_pool(x, batch)
        if self.desc_mlp is not None:
            u = _maybe_get_u(data)
            if u is not None:
                x = torch.cat([x, self.desc_mlp(u)], dim=1)
        return self.fc(x).view(-1)


# --- GIN (with edge features via GINEConv) --------------------------------

class GINNet(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int = 64, num_layers: int = 3,
                 dropout: float = 0.3, desc_dim: int = 0, **_):
        super().__init__()
        self.dropout = dropout
        self.desc_dim = desc_dim
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=edge_dim, train_eps=True))
            self.bns.append(BatchNorm(hidden_dim))
        if desc_dim > 0:
            self.desc_mlp = DescriptorMLP(desc_dim, hidden_dim)
            head_in = hidden_dim + hidden_dim
        else:
            self.desc_mlp = None
            head_in = hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, ei, ea, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.in_proj(x)
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, ei, ea)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_add_pool(x, batch)  # GIN paper: sum readout
        if self.desc_mlp is not None:
            u = _maybe_get_u(data)
            if u is not None:
                x = torch.cat([x, self.desc_mlp(u)], dim=1)
        return self.fc(x).view(-1)


# --- MPNN -----------------------------------------------------------------

class MPNNNet(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int = 64, num_steps: int = 3,
                 dropout: float = 0.3, desc_dim: int = 0, **_):
        super().__init__()
        self.dropout = dropout
        self.num_steps = num_steps
        self.desc_dim = desc_dim
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        edge_net = nn.Sequential(
            nn.Linear(edge_dim, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim * hidden_dim),
        )
        self.conv = NNConv(hidden_dim, hidden_dim, edge_net, aggr="mean")
        self.gru = nn.GRU(hidden_dim, hidden_dim)
        self.set2set = Set2Set(hidden_dim, processing_steps=3)
        if desc_dim > 0:
            self.desc_mlp = DescriptorMLP(desc_dim, hidden_dim)
            head_in = 2 * hidden_dim + hidden_dim
        else:
            self.desc_mlp = None
            head_in = 2 * hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, ei, ea, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.in_proj(x))
        h = x.unsqueeze(0).contiguous()
        for _ in range(self.num_steps):
            m = F.relu(self.conv(x, ei, ea))
            x, h = self.gru(m.unsqueeze(0), h)
            x = x.squeeze(0)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.set2set(x, batch)
        if self.desc_mlp is not None:
            u = _maybe_get_u(data)
            if u is not None:
                x = torch.cat([x, self.desc_mlp(u)], dim=1)
        return self.fc(x).view(-1)


MODEL_REGISTRY = {
    "GCN": GCNNet,
    "GAT": GATNet,
    "GraphSAGE": GraphSAGENet,
    "GIN": GINNet,
    "MPNN": MPNNNet,
}

USES_EDGE_FEATURES = {"GIN", "MPNN"}


def build_model(name: str, in_dim: int, edge_dim: int, params: dict, desc_dim: int = 0) -> nn.Module:
    """Factory that filters per-model kwargs from a generic param dict.

    desc_dim > 0 enables global descriptor fusion at the readout layer.
    Pass desc_dim=0 (default) for backward-compat with code that doesn't supply
    per-graph descriptors on `data.u`.
    """
    cls = MODEL_REGISTRY[name]
    kwargs = {
        "in_dim": in_dim,
        "hidden_dim": params.get("hidden_dim", 64),
        "dropout": params.get("dropout", 0.3),
        "desc_dim": desc_dim,
    }
    if name == "GAT":
        kwargs["heads"] = params.get("heads", 4)
        kwargs["num_layers"] = params.get("num_layers", 3)
    elif name == "GraphSAGE":
        kwargs["num_layers"] = params.get("num_layers", 3)
        kwargs["aggr"] = params.get("aggr", "max")
    elif name == "GCN":
        kwargs["num_layers"] = params.get("num_layers", 3)
    elif name == "GIN":
        kwargs["edge_dim"] = edge_dim
        kwargs["num_layers"] = params.get("num_layers", 3)
    elif name == "MPNN":
        kwargs["edge_dim"] = edge_dim
        kwargs["num_steps"] = params.get("num_steps", 3)
    return cls(**kwargs)
