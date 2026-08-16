"""
temporal_model.py

Extends your existing VoltageHeteroGNN (pyg_dataset.py __main__: lin_in ->
conv1 -> conv2 -> out) with:

  1. A shared per-timestep encoder run across a window of W snapshots,
     followed by an LSTM head for direct multi-horizon (N-step) voltage
     forecasting per Bus.
  2. An edge-level head on ("bus","line_segment","bus") for line current
     forecasting.

IMPORTANT existing-code note: your current conv1/conv2 use plain SAGEConv,
which only takes (x_dict, edge_index_dict) -- it never sees edge_attr, so
the impedance values (r1,x1,r0,x0,c1,c0) you fold into line_segment edges
(pyg_dataset.py, CONFORMS_TO join) are stored but currently invisible to
message passing for the *voltage* prediction. This file swaps the
line_segment relation to TransformerConv(edge_dim=...), which does consume
edge_attr, so impedance actually influences the Bus embeddings now -- for
both the voltage and current heads. Other relations (supplies_power,
conforms_to, has_feeder_head, feeds) have no edge_attr in your pipeline, so
they stay plain SAGEConv, matching your existing code exactly.

This depends on your installed torch_geometric version passing
edge_attr_dict through HeteroConv.forward to relations whose conv accepts
it (true for PyG >= ~2.0.4). If your version doesn't, set
use_edge_attention=False in HeteroEncoder to fall back to your exact
current behavior (edge_attr stored but unused), and everything else in
this file still works unchanged.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, TransformerConv

LINE_REL = ("bus", "line_segment", "bus")


class HeteroEncoder(nn.Module):
    """
    Same shape as your VoltageHeteroGNN's lin_in/conv1/conv2, factored out
    so it can run once per snapshot inside a window with shared weights.
    """

    def __init__(
        self,
        in_dims: Dict[str, int],
        relations: List[Tuple[str, str, str]],
        edge_attr_dim: int,
        hidden: int = 64,
        use_edge_attention: bool = True,
    ):
        super().__init__()
        self.lin_in = nn.ModuleDict({
            ntype: nn.Linear(dim, hidden) for ntype, dim in in_dims.items()
        })

        def make_conv():
            convs = {}
            for rel in relations:
                if use_edge_attention and rel == LINE_REL:
                    convs[rel] = TransformerConv((-1, -1), hidden, edge_dim=edge_attr_dim)
                else:
                    convs[rel] = SAGEConv((-1, -1), hidden)
            return HeteroConv(convs, aggr="sum")

        self.conv1 = make_conv()
        self.conv2 = make_conv()
        self.hidden = hidden
        self.use_edge_attention = use_edge_attention

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        x_dict = {k: F.relu(self.lin_in[k](v)) for k, v in x_dict.items()}

        kwargs = {"edge_attr_dict": edge_attr_dict} if self.use_edge_attention else {}
        out1 = self.conv1(x_dict, edge_index_dict, **kwargs)
        x_dict = {**x_dict, **{k: F.relu(v) for k, v in out1.items()}}

        out2 = self.conv2(x_dict, edge_index_dict, **kwargs)
        x_dict = {**x_dict, **out2}

        return x_dict


class TemporalVoltageHeteroGNN(nn.Module):
    def __init__(
        self,
        in_dims: Dict[str, int],
        relations: List[Tuple[str, str, str]],
        edge_attr_dim: int = 10,   # length(1) + phase one-hot(3) + r1,x1,r0,x0,c1,c0(6) = 10
        hidden: int = 64,
        horizon: int = 15,
        lstm_layers: int = 1,
        use_edge_attention: bool = True,
    ):
        super().__init__()
        self.encoder = HeteroEncoder(in_dims, relations, edge_attr_dim, hidden, use_edge_attention)
        self.hidden = hidden
        self.horizon = horizon

        self.lstm = nn.LSTM(hidden, hidden, num_layers=lstm_layers, batch_first=True)

        # Direct multi-horizon: last LSTM hidden state -> N future Vpu values per bus
        self.voltage_head = nn.Linear(hidden, horizon)

        # Edge head: concat(h_src, h_dst, edge_attr) -> N future currents per line
        self.current_head = nn.Sequential(
            nn.Linear(2 * hidden + edge_attr_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, horizon),
        )

    def forward(self, graph_batch, B: int, W: int):
        """
        graph_batch: PyG Batch of B*W flattened HeteroData snapshots, as
                     produced by temporal_dataset.collate_windows.
        """
        x_dict = self.encoder(
            graph_batch.x_dict, graph_batch.edge_index_dict, graph_batch.edge_attr_dict
        )
        bus_emb_flat = x_dict["bus"]                       # (B*W*num_bus, hidden)
        num_bus = bus_emb_flat.size(0) // (B * W)
        bus_emb = bus_emb_flat.view(B, W, num_bus, self.hidden)

        lstm_in = bus_emb.permute(0, 2, 1, 3).reshape(B * num_bus, W, self.hidden)
        _, (h_n, _) = self.lstm(lstm_in)
        last_hidden = h_n[-1]                                # (B*num_bus, hidden)

        voltage_pred = self.voltage_head(last_hidden)          # (B*num_bus, horizon)
        voltage_pred = voltage_pred.view(B, num_bus, self.horizon).permute(0, 2, 1)
        # -> (B, horizon, num_bus)

        # Line current head, using the *last* snapshot's line_segment
        # topology/edge_attr in each window (fixed topology, so any
        # snapshot's edge_index for this relation would do -- we use the
        # batch's own per-graph edge_index for exact index alignment).
        last_graphs = graph_batch.to_data_list()[W - 1 :: W]   # every W-th graph = last step of each window
        edge_index = last_graphs[0][LINE_REL].edge_index          # (2, num_lines), same topology for all
        edge_attr = torch.stack([g[LINE_REL].edge_attr for g in last_graphs])  # (B, num_lines, edge_attr_dim)

        bus_hidden = last_hidden.view(B, num_bus, self.hidden)
        src, dst = edge_index
        h_src = bus_hidden[:, src, :]                            # (B, num_lines, hidden)
        h_dst = bus_hidden[:, dst, :]                             # (B, num_lines, hidden)

        edge_in = torch.cat([h_src, h_dst, edge_attr], dim=-1)
        current_pred = self.current_head(edge_in)                  # (B, num_lines, horizon)
        current_pred = current_pred.permute(0, 2, 1)                 # -> (B, horizon, num_lines)

        return voltage_pred, current_pred


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Same convention as your existing label_mask / __main__ training loop
    (`loss = F.mse_loss(pred[mask], batch["bus"].y[mask])`): select via
    boolean indexing rather than multiplying by a 0/1 mask. Unlabeled
    targets are stored as NaN (see pyg_dataset.py's torch.where pattern),
    and NaN * 0 is still NaN in IEEE float -- multiplying would silently
    poison the loss instead of excluding those entries."""
    mask = mask.bool()
    diff2 = (pred[mask] - target[mask]) ** 2
    if diff2.numel() == 0:
        return pred.new_zeros(())
    return diff2.mean()