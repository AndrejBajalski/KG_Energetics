"""
temporal_dataset.py

Two additions on top of your existing pyg_dataset.py:

1. FeederHeteroSnapshotDatasetWithCurrent -- subclasses
   FeederHeteroSnapshotDataset to also load line_currents.csv and attach
   per-timestep current labels + a label_mask to the
   ("bus", "line_segment", "bus") edge store, the same way voltage labels
   are attached to the "bus" node store.

2. TemporalFeederWindowDataset -- wraps any FeederHeteroSnapshotDataset
   (with or without current labels) into (window of W snapshots -> N-step
   horizon) training examples for the LSTM head, split-aware so a window's
   *target* never crosses a train/val/test boundary. Input history is
   allowed to reach before the boundary -- that's normal forecasting, not
   leakage.

Line-name alignment: line_currents.csv columns are OpenDSS Line names
(dss.Lines.AllNames(), from generate_labels.py), which are exactly the
"Name" column of Lines.csv -- the same value stored as the LINE_SEGMENT
relationship's `name` property in load_data.py
(`MERGE (b1)-[r:LINE_SEGMENT {name: toString(row.Name)}]->(b2)`). So we
just need to also SELECT r.name in the line_rows query and keep it in the
same order as line_rows_valid, which is exactly what edge_attr/edge_index
for ("bus","line_segment","bus") are built from.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import HeteroData, Batch

from pyg_dataset import FeederHeteroSnapshotDataset, _phase_onehot


class FeederHeteroSnapshotDatasetWithCurrent(FeederHeteroSnapshotDataset):
    """
    Same as FeederHeteroSnapshotDataset, plus per-timestep line current
    labels attached to the line_segment edge store as `.y` / `.label_mask`,
    mirroring the Bus voltage convention exactly.
    """

    def __init__(self, *args, line_currents_path, **kwargs):
        # Need line names aligned with edge order, which the parent's
        # _fetch_static_graph doesn't expose -- re-run that query ourselves
        # inside an overridden _fetch_static_graph (below) so self.line_names
        # is built from the *same* query result as edge_index/edge_attr,
        # rather than risking a second query returning rows in a different
        # order.
        self._line_currents_path = Path(line_currents_path)
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    def _fetch_static_graph(self, session, include_upstream):
        # Delegate to the parent for everything (nodes, all edge types,
        # edge_attr) -- then do ONE extra query, using the identical MATCH
        # pattern plus r.name, to recover line names in the same row order
        # line_rows_valid was filtered from. Cypher without ORDER BY isn't
        # guaranteed stable across separate queries, so instead of trusting
        # a second query to align with the first, we rebuild the same
        # src/dst-valid filter here explicitly against idx_of, exactly as
        # the parent does for line_rows_valid.
        super()._fetch_static_graph(session, include_upstream)

        line_name_rows = session.run("""
            MATCH (b1:Bus)-[r:LINE_SEGMENT]->(b2:Bus)
            RETURN b1.id AS src, b2.id AS dst, coalesce(r.name, '') AS name
        """).data()

        idx_of = self._idx_of
        valid = [r for r in line_name_rows
                 if r["src"] in idx_of["bus"] and r["dst"] in idx_of["bus"]]
        self.line_names = [r["name"] for r in valid]

        n_edges = self.edge_index[("bus", "line_segment", "bus")].size(1)
        if len(self.line_names) != n_edges:
            raise RuntimeError(
                f"line_names ({len(self.line_names)}) doesn't match "
                f"line_segment edge count ({n_edges}) -- Cypher row order "
                f"assumption broken, don't trust the alignment silently."
            )

    # ------------------------------------------------------------------
    def _load_line_currents(self):
        raw = pd.read_csv(self._line_currents_path, index_col=0)

        # OpenDSS lowercases object names in dss.Lines.AllNames()
        # (generate_labels.py), while Neo4j keeps the original CSV casing
        # (load_data.py's toString(row.Name)) -- same line, different case.
        # Match case-insensitively, but keep self.line_names' original
        # casing as the canonical label everywhere downstream.
        raw.columns = [c.upper() for c in raw.columns]
        line_names_upper = [n.upper() for n in self.line_names]

        missing = [n for n, nu in zip(self.line_names, line_names_upper) if nu not in raw.columns]
        if missing:
            print(f"NOTE: {len(missing)} line_segment edges have no current "
                  f"label and will be excluded from that loss: "
                  f"{missing[:5]}{'...' if len(missing) > 5 else ''}")

        self._line_label_mask = torch.tensor(
            [nu in raw.columns for nu in line_names_upper], dtype=torch.bool
        )
        self.line_current_labels = raw.reindex(columns=line_names_upper)
        self.line_current_labels.columns = self.line_names  # restore original casing

    # ------------------------------------------------------------------
    def fit_normalization(self, train_indices):
        super().fit_normalization(train_indices)

        if not hasattr(self, "line_current_labels"):
            self._load_line_currents()

        i_train = torch.tensor(
            self.line_current_labels.iloc[list(train_indices)].values, dtype=torch.float
        )
        # NaNs from unlabeled columns must not pollute the mean/std --
        # mask them out the same way bus voltage handles missing columns.
        valid_cols = self._line_label_mask
        i_train_valid = i_train[:, valid_cols]
        i_mean = i_train_valid.mean()
        i_std = i_train_valid.std().clamp(min=1e-6)
        self._current_norm_stats = (i_mean, i_std)

    # ------------------------------------------------------------------
    def get(self, idx):
        data = super().get(idx)

        if not hasattr(self, "line_current_labels"):
            self._load_line_currents()

        i_t = torch.tensor(self.line_current_labels.iloc[idx].values, dtype=torch.float)
        y = torch.where(self._line_label_mask, i_t, torch.tensor(float("nan")))

        if getattr(self, "_current_norm_stats", None) is not None:
            mean, std = self._current_norm_stats
            y = torch.where(self._line_label_mask, (y - mean) / std, y)

        rel = ("bus", "line_segment", "bus")
        data[rel].y = y
        data[rel].label_mask = self._line_label_mask.clone()
        return data


# ==============================================================================
# Sliding-window wrapper for LSTM training
# ==============================================================================

class TemporalFeederWindowDataset(TorchDataset):
    """
    Wraps a (with-current) FeederHeteroSnapshotDataset into
    (window of W past snapshots -> N future bus/line targets) examples.

    base_ds must already have fit_normalization() called (time_split() on
    the base dataset does this automatically -- pass base_ds.time_split()'s
    train indices to fit, then build one TemporalFeederWindowDataset per
    split with the matching split_range).
    """

    def __init__(
        self,
        base_ds: FeederHeteroSnapshotDatasetWithCurrent,
        window: int = 30,
        horizon: int = 15,
        split_range: tuple[int, int] | None = None,
    ):
        self.base_ds = base_ds
        self.window = window
        self.horizon = horizon

        if not hasattr(base_ds, "line_current_labels"):
            base_ds._load_line_currents()

        total = base_ds.n_timesteps
        lo, hi = split_range if split_range is not None else (0, total)

        self.valid_t: List[int] = []
        for t in range(window - 1, total - horizon):
            first_target, last_target = t + 1, t + horizon
            if lo <= first_target and last_target < hi:
                self.valid_t.append(t)

        # Pre-materialize target tensors (small: n_timesteps x n_bus / n_line)
        bus_vals = base_ds.voltage_labels.reindex(columns=base_ds.node_id["bus"]).values
        self._bus_all = torch.tensor(bus_vals, dtype=torch.float)          # (T, num_bus)
        self._bus_mask_all = base_ds._bus_label_mask                        # (num_bus,)

        line_vals = base_ds.line_current_labels.values
        self._line_all = torch.tensor(line_vals, dtype=torch.float)          # (T, num_line)
        self._line_mask_all = base_ds._line_label_mask                        # (num_line,)

        # Match the normalization the model will be trained against --
        # fit_normalization() on base_ds must already have been called
        # (it's what computes _current_norm_stats). NaN entries (unlabeled
        # lines) stay NaN under this arithmetic, which is fine: masked_mse
        # excludes them via boolean indexing, never touches the value.
        if getattr(base_ds, "_current_norm_stats", None) is not None:
            mean, std = base_ds._current_norm_stats
            self._line_all = (self._line_all - mean) / std

    def __len__(self):
        return len(self.valid_t)

    def __getitem__(self, idx):
        t = self.valid_t[idx]
        history = [self.base_ds.get(s) for s in range(t - self.window + 1, t + 1)]
        target_idx = list(range(t + 1, t + 1 + self.horizon))

        bus_targets = self._bus_all[target_idx]                              # (N, num_bus)
        bus_mask = self._bus_mask_all.unsqueeze(0).expand(self.horizon, -1)   # (N, num_bus)
        line_targets = self._line_all[target_idx]                              # (N, num_line)
        line_mask = self._line_mask_all.unsqueeze(0).expand(self.horizon, -1)   # (N, num_line)

        return history, bus_targets, bus_mask, line_targets, line_mask


def collate_windows(batch):
    """
    Flattens (B windows x W snapshots) into one PyG Batch of B*W graphs so
    the shared encoder runs in a single forward pass. Relies on every
    snapshot sharing the same fixed topology (true here -- only Load.x and
    the y/label_mask targets vary per timestep), so Bus-node ordering is
    identical across all graphs in the batch.
    """
    B = len(batch)
    W = len(batch[0][0])

    flat_graphs: List[HeteroData] = []
    for history, *_ in batch:
        flat_graphs.extend(history)

    graph_batch = Batch.from_data_list(flat_graphs)

    bus_targets = torch.stack([b for _, b, _, _, _ in batch])   # (B, N, num_bus)
    bus_mask = torch.stack([m for _, _, m, _, _ in batch])       # (B, N, num_bus)
    line_targets = torch.stack([l for _, _, _, l, _ in batch])     # (B, N, num_line)
    line_mask = torch.stack([m for _, _, _, _, m in batch])         # (B, N, num_line)

    return {
        "graph_batch": graph_batch,
        "B": B,
        "W": W,
        "bus_targets": bus_targets,
        "bus_mask": bus_mask,
        "line_targets": line_targets,
        "line_mask": line_mask,
    }