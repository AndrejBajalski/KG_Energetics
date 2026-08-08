"""
pyg_dataset.py (HeteroData version)

Builds a heterogeneous graph snapshot dataset from the full Neo4j schema:
node types {Bus, Load, LineCode, SubstationTransformer, Source} and edge
types {LINE_SEGMENT, SUPPLIES_POWER, CONFORMS_TO, HAS_FEEDER_HEAD, FEEDS}.

Unlike the earlier homogeneous version, this:
  - keeps each relationship type as its own edge_index/edge_attr, so a
    HeteroConv model learns separate weights per relation instead of
    treating "line segment" and "supplies power" edges identically
  - folds LineCode impedance (r1/x1/r0/x0/c1/c0) directly into
    LINE_SEGMENT edge_attr via the CONFORMS_TO link, so the model sees
    actual electrical impedance, not just line length
  - includes the Source -> Transformer -> Bus chain, so distance-from-
    source is represented structurally, not just inferred
  - reads the precomputed Bus.hopsFromSource / Bus.cumulativeDownstreamKW
    properties as static node features (cheap Cypher aggregates computed
    once at ingestion time, not recomputed per training run)

UPSTREAM_OF is deliberately NOT included as a message-passing edge type
by default -- it's a transitive closure (every bus links to every
downstream bus), so it's O(n^2) edges and would dominate message passing
with redundant long-range shortcuts. Pass include_upstream=True to add it
as an extra relation if you want the GNN to have direct long-range hops
without stacking many layers.

Requires: torch, torch_geometric, neo4j, pandas
"""

import torch, os
from dotenv import load_dotenv
from pathlib import Path
from neo4j import GraphDatabase
from torch_geometric.data import HeteroData, Dataset

BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

def _phase_onehot(phase_str):
    mapping = {"A": 0, "B": 1, "C": 2}
    vec = [0, 0, 0]
    for c in str(phase_str).upper():
        if c in mapping:
            vec[mapping[c]] = 1
    return vec


class FeederHeteroSnapshotDataset(Dataset):
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password,
                 load_profiles_path, voltage_labels_path,
                 include_upstream: bool = False, n_timesteps: int = None):
        super().__init__()
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        with driver.session() as session:
            self._fetch_static_graph(session, include_upstream)
        driver.close()

        # --- Time-varying data ---
        import pandas as pd
        self.load_profiles = pd.read_csv(load_profiles_path, index_col=0)
        raw_voltages = pd.read_csv(voltage_labels_path, index_col=0)
        self.voltage_labels = self._collapse_phase_columns_to_bus(raw_voltages)

        load_names = self.node_id["load"]
        missing = set(load_names) - set(self.load_profiles.columns)
        if missing:
            raise ValueError(f"Load profiles missing columns for: {missing}")
        self.load_profiles = self.load_profiles[load_names]

        bus_names = self.node_id["bus"]
        self._bus_label_mask = torch.tensor(
            [b in self.voltage_labels.columns for b in bus_names], dtype=torch.bool
        )
        missing_bus = [b for b in bus_names if b not in self.voltage_labels.columns]
        if missing_bus:
            print(f"NOTE: {len(missing_bus)} Bus nodes have no voltage label and will "
                  f"be excluded from the loss: {missing_bus[:5]}{'...' if len(missing_bus) > 5 else ''}")
        self.voltage_labels = self.voltage_labels.reindex(columns=bus_names)

        self.n_timesteps = n_timesteps or len(self.load_profiles)

    # ------------------------------------------------------------------
    @staticmethod
    def _collapse_phase_columns_to_bus(df):
        """bus_voltages.csv has one column per bus-phase terminal ('1.1',
        '1.2', '1.3', ...). Collapse to one column per Bus via min Vpu
        across present phases (a single sagging phase should count)."""
        bus_of = {col: col.split(".")[0] for col in df.columns}
        out = {}
        for bus_name in sorted(set(bus_of.values()), key=lambda b: (len(b), b)):
            cols = [c for c, b in bus_of.items() if b == bus_name]
            out[bus_name] = df[cols].min(axis=1)
        import pandas as pd
        return pd.DataFrame(out)

    # ------------------------------------------------------------------
    def _fetch_static_graph(self, session, include_upstream):
        # ---- Nodes, one query per type ----
        bus_rows = session.run("""
            MATCH (b:Bus)
            RETURN b.id AS id, coalesce(b.x, 0.0) AS x, coalesce(b.y, 0.0) AS y,
                   coalesce(b.hopsFromSource, -1) AS hops,
                   coalesce(b.cumulativeDownstreamKW, 0.0) AS cumKW
        """).data()

        load_rows = session.run("""
            MATCH (l:Load)
            RETURN l.id AS id, coalesce(l.kV, 0.0) AS kV,
                   coalesce(l.powerFactor, 0.0) AS pf,
                   coalesce(l.model, 0) AS model,
                   coalesce(l.targetPhase, '') AS phase,
                   coalesce(l.connectionType, '') AS conn
        """).data()

        linecode_rows = session.run("""
            MATCH (lc:LineCode)
            RETURN lc.name AS id, coalesce(lc.nphases, 0) AS nphases,
                   coalesce(lc.r1, 0.0) AS r1, coalesce(lc.x1, 0.0) AS x1,
                   coalesce(lc.r0, 0.0) AS r0, coalesce(lc.x0, 0.0) AS x0,
                   coalesce(lc.c1, 0.0) AS c1, coalesce(lc.c0, 0.0) AS c0
        """).data()

        xfmr_rows = session.run("""
            MATCH (t:SubstationTransformer)
            RETURN t.id AS id, coalesce(t.phases, 0) AS phases,
                   coalesce(t.ratedMVA, 0.0) AS mva,
                   coalesce(t.primaryKV, 0.0) AS kvp, coalesce(t.secondaryKV, 0.0) AS kvs,
                   coalesce(t.xhl, 0.0) AS xhl, coalesce(t.resistancePct, 0.0) AS rpct
        """).data()

        source_rows = session.run("""
            MATCH (s:Source)
            RETURN s.id AS id, coalesce(s.nominalKV, 0.0) AS kv,
                   coalesce(s.pu, 1.0) AS pu, coalesce(s.isc3, 0.0) AS isc3,
                   coalesce(s.isc1, 0.0) AS isc1
        """).data()

        self.node_id = {
            "bus": [r["id"] for r in bus_rows],
            "load": [r["id"] for r in load_rows],
            "linecode": [r["id"] for r in linecode_rows],
            "transformer": [r["id"] for r in xfmr_rows],
            "source": [r["id"] for r in source_rows],
        }
        idx_of = {t: {name: i for i, name in enumerate(ids)} for t, ids in self.node_id.items()}
        self._idx_of = idx_of  # kept for edge-building below

        self.static_x = {
            "bus": torch.tensor(
                [[r["x"], r["y"], r["hops"], r["cumKW"]] for r in bus_rows], dtype=torch.float
            ),
            "load": torch.tensor(
                [[r["kV"], r["pf"], r["model"]] + _phase_onehot(r["phase"]) +
                 [1.0 if r["conn"].lower() == "wye" else 0.0,
                  1.0 if r["conn"].lower() == "delta" else 0.0]
                 for r in load_rows], dtype=torch.float
            ),
            "linecode": torch.tensor(
                [[r["nphases"], r["r1"], r["x1"], r["r0"], r["x0"], r["c1"], r["c0"]]
                 for r in linecode_rows], dtype=torch.float
            ),
            "transformer": torch.tensor(
                [[r["phases"], r["mva"], r["kvp"], r["kvs"], r["xhl"], r["rpct"]]
                 for r in xfmr_rows], dtype=torch.float
            ),
            "source": torch.tensor(
                [[r["kv"], r["pu"], r["isc3"], r["isc1"]] for r in source_rows], dtype=torch.float
            ),
        }

        # ---- Edges, one query per relationship type ----
        # LINE_SEGMENT with LineCode impedance folded in via CONFORMS_TO
        # (CONFORMS_TO is attached to the line's target bus, per ingestion schema)
        line_rows = session.run("""
            MATCH (b1:Bus)-[r:LINE_SEGMENT]->(b2:Bus)
            OPTIONAL MATCH (b2)-[:CONFORMS_TO]->(lc:LineCode)
            RETURN b1.id AS src, b2.id AS dst,
                   coalesce(r.length, 0.0) AS length, coalesce(r.phases, '') AS phases,
                   coalesce(lc.r1, 0.0) AS r1, coalesce(lc.x1, 0.0) AS x1,
                   coalesce(lc.r0, 0.0) AS r0, coalesce(lc.x0, 0.0) AS x0,
                   coalesce(lc.c1, 0.0) AS c1, coalesce(lc.c0, 0.0) AS c0
        """).data()

        supply_rows = session.run("""
            MATCH (b:Bus)-[:SUPPLIES_POWER]->(l:Load)
            RETURN b.id AS src, l.id AS dst
        """).data()

        conforms_rows = session.run("""
            MATCH (b:Bus)-[:CONFORMS_TO]->(lc:LineCode)
            RETURN b.id AS src, lc.name AS dst
        """).data()

        feeder_head_rows = session.run("""
            MATCH (t:SubstationTransformer)-[:HAS_FEEDER_HEAD]->(b:Bus)
            RETURN t.id AS src, b.id AS dst
        """).data()

        feeds_rows = session.run("""
            MATCH (s:Source)-[:FEEDS]->(t:SubstationTransformer)
            RETURN s.id AS src, t.id AS dst
        """).data()

        def build_edge_index(rows, src_type, dst_type):
            valid = [r for r in rows if r["src"] in idx_of[src_type] and r["dst"] in idx_of[dst_type]]
            src = [idx_of[src_type][r["src"]] for r in valid]
            dst = [idx_of[dst_type][r["dst"]] for r in valid]
            return torch.tensor([src, dst], dtype=torch.long)

        self.edge_index = {
            ("bus", "line_segment", "bus"): build_edge_index(line_rows, "bus", "bus"),
            ("bus", "supplies_power", "load"): build_edge_index(supply_rows, "bus", "load"),
            ("bus", "conforms_to", "linecode"): build_edge_index(conforms_rows, "bus", "linecode"),
            ("transformer", "has_feeder_head", "bus"): build_edge_index(feeder_head_rows, "transformer", "bus"),
            ("source", "feeds", "transformer"): build_edge_index(feeds_rows, "source", "transformer"),
        }

        line_rows_valid = [r for r in line_rows if r["src"] in idx_of["bus"] and r["dst"] in idx_of["bus"]]
        self.edge_attr = {
            ("bus", "line_segment", "bus"): torch.tensor(
                [[r["length"]] + _phase_onehot(r["phases"]) +
                 [r["r1"], r["x1"], r["r0"], r["x0"], r["c1"], r["c0"]]
                 for r in line_rows_valid],
                dtype=torch.float
            ),
        }

        if include_upstream:
            upstream_rows = session.run("""
                MATCH (b1:Bus)-[:UPSTREAM_OF]->(b2:Bus)
                RETURN b1.id AS src, b2.id AS dst
            """).data()
            self.edge_index[("bus", "upstream_of", "bus")] = build_edge_index(upstream_rows, "bus", "bus")

    # ------------------------------------------------------------------
    def fit_normalization(self, train_indices):
        """Compute per-node-type feature mean/std for z-score normalization,
        using ONLY the training timesteps (never val/test, to avoid leaking
        their distribution into training). Must be called before get() if
        you want normalized features -- time_split() calls this
        automatically by default.

        Coordinates, cumulative downstream kW, and load kW sit several
        orders of magnitude apart from impedance/phase features; without
        this, the first linear layer sees wildly different feature scales
        and training is unstable (this is what caused MSE ~17 instead of
        the ~0.001-0.01 you'd expect for Vpu regression).
        """
        self.norm_stats = {}

        for ntype, feat in self.static_x.items():
            if ntype == "load":
                continue  # handled below -- has a time-varying column
            mean = feat.mean(dim=0)
            std = feat.std(dim=0).clamp(min=1e-6)
            self.norm_stats[ntype] = (mean, std)

        # Load: static columns are constant across time, but the appended
        # kW column varies -- compute its mean/std from TRAIN timesteps only.
        static_load = self.static_x["load"]
        static_mean = static_load.mean(dim=0)
        static_std = static_load.std(dim=0).clamp(min=1e-6)

        kw_train = torch.tensor(
            self.load_profiles.iloc[list(train_indices)].values, dtype=torch.float
        )
        kw_mean = kw_train.mean().unsqueeze(0)
        kw_std = kw_train.std().clamp(min=1e-6).unsqueeze(0)

        self.norm_stats["load"] = (
            torch.cat([static_mean, kw_mean]),
            torch.cat([static_std, kw_std]),
        )

    # ------------------------------------------------------------------
    def len(self):
        return self.n_timesteps

    def get(self, idx):
        data = HeteroData()

        for ntype, feat in self.static_x.items():
            data[ntype].x = feat.clone()

        # Overwrite Load features with this timestep's kW appended as an extra column
        kw_t = torch.tensor(self.load_profiles.iloc[idx].values, dtype=torch.float).unsqueeze(1)
        data["load"].x = torch.cat([self.static_x["load"], kw_t], dim=1)

        # Bus voltage labels for this timestep
        v_t = torch.tensor(self.voltage_labels.iloc[idx].values, dtype=torch.float)
        y = torch.where(self._bus_label_mask, v_t, torch.tensor(float("nan")))
        data["bus"].y = y
        data["bus"].label_mask = self._bus_label_mask.clone()

        for rel, ei in self.edge_index.items():
            data[rel].edge_index = ei
        for rel, ea in self.edge_attr.items():
            data[rel].edge_attr = ea

        # Apply z-score normalization if fit_normalization() has been called.
        if getattr(self, "norm_stats", None) is not None:
            for ntype in data.node_types:
                mean, std = self.norm_stats[ntype]
                data[ntype].x = (data[ntype].x - mean) / std

        data["bus"].t = idx
        return data

    # ------------------------------------------------------------------
    def time_split(self, train_frac=0.7, val_frac=0.15, fit_norm=True):
        n = self.n_timesteps
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train_idx = list(range(0, n_train))
        val_idx = list(range(n_train, n_train + n_val))
        test_idx = list(range(n_train + n_val, n))

        if fit_norm:
            self.fit_normalization(train_idx)

        return (torch.utils.data.Subset(self, train_idx),
                torch.utils.data.Subset(self, val_idx),
                torch.utils.data.Subset(self, test_idx))


# Example training loop skeleton -----------------------------------------
if __name__ == "__main__":
    import torch.nn.functional as F
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import HeteroConv, SAGEConv

    class VoltageHeteroGNN(torch.nn.Module):
        def __init__(self, metadata, in_dims, hidden=64):
            super().__init__()
            self.lin_in = torch.nn.ModuleDict({
                ntype: torch.nn.Linear(dim, hidden) for ntype, dim in in_dims.items()
            })
            self.conv1 = HeteroConv({
                rel: SAGEConv((-1, -1), hidden) for rel in metadata[1]
            }, aggr="sum")
            self.conv2 = HeteroConv({
                rel: SAGEConv((-1, -1), hidden) for rel in metadata[1]
            }, aggr="sum")
            self.out = torch.nn.Linear(hidden, 1)

        def forward(self, x_dict, edge_index_dict):
            x_dict = {k: F.relu(self.lin_in[k](v)) for k, v in x_dict.items()}
            x_dict = self.conv1(x_dict, edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
            x_dict = self.conv2(x_dict, edge_index_dict)
            return self.out(x_dict["bus"]).squeeze(-1)


    ds = FeederHeteroSnapshotDataset(
        neo4j_uri=os.environ.get("NEO4J_URI"),
        neo4j_user=os.environ.get("NEO4J_USERNAME"),
        neo4j_password=os.environ.get("NEO4J_PASSWORD"),
        load_profiles_path=BASE_DIR / "data" / "Computed" / "load_profiles.csv",
        voltage_labels_path=BASE_DIR / "data" / "Computed" / "bus_voltages.csv"
    )
    train_ds, val_ds, test_ds = ds.time_split()
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    sample = ds[0]
    in_dims = {ntype: sample[ntype].x.size(1) for ntype in sample.node_types}
    model = VoltageHeteroGNN(sample.metadata(), in_dims)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(50):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            opt.zero_grad()
            pred = model(batch.x_dict, batch.edge_index_dict)
            mask = batch["bus"].label_mask
            loss = F.mse_loss(pred[mask], batch["bus"].y[mask])
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        print(f"epoch {epoch}: train MSE {total_loss / len(train_ds):.5f}")