"""
hetero_gnn_dataset.py (HeteroData version - Multi-Task Extension)

Builds a heterogeneous graph snapshot dataset from the full Neo4j schema:
node types {Bus, Load, LineCode, SubstationTransformer, Source} and edge
types {LINE_SEGMENT, SUPPLIES_POWER, CONFORMS_TO, HAS_FEEDER_HEAD, FEEDS}.

Extended to load both Bus Voltage (node labels) and Line Current (edge labels)
to enable multi-task target predictions natively.

Requires: torch, torch_geometric, neo4j, pandas
"""

import torch, os
from collections import deque
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase
from torch_geometric.data import HeteroData, Dataset
from torch_geometric.transforms import ToUndirected
import pandas as pd


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
                 current_labels_path=None,
                 include_upstream: bool = False, n_timesteps: int = None):

        super().__init__()

        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        with driver.session() as session:
            self._fetch_static_graph(session, include_upstream)
        driver.close()

        # --- Time-varying data ---
        self.load_profiles = pd.read_csv(load_profiles_path, index_col=0)
        raw_voltages = pd.read_csv(voltage_labels_path, index_col=0)
        self.voltage_labels = self._collapse_phase_columns_to_bus(raw_voltages)

        load_names = self.node_id["load"]
        missing = set(load_names) - set(self.load_profiles.columns)
        if missing:
            raise ValueError(f"Load profiles missing columns for: {missing}")
        self.load_profiles = self.load_profiles[load_names]

        # Map Voltage Labels to Bus Node Order
        bus_names = self.node_id["bus"]
        self._bus_label_mask = torch.tensor(
            [b in self.voltage_labels.columns for b in bus_names], dtype=torch.bool
        )
        missing_bus = [b for b in bus_names if b not in self.voltage_labels.columns]
        if missing_bus:
            print(f"NOTE: {len(missing_bus)} Bus nodes have no voltage label and will "
                  f"be excluded from the loss: {missing_bus[:5]}{'...' if len(missing_bus) > 5 else ''}")
        self.voltage_labels = self.voltage_labels.reindex(columns=bus_names)

        # Map Current Labels to Line Segment Edge Order
        self.has_current_labels = False
        if current_labels_path is not None and os.path.exists(current_labels_path):
            self.current_labels = pd.read_csv(current_labels_path, index_col=0)

            line_edge_names = self.line_names[("bus", "line_segment", "bus")]
            # Standardize column naming if OpenDSS formats them differently
            self.current_labels.columns = [str(c).lower().strip() for c in self.current_labels.columns]
            line_edge_names_lower = [str(n).lower().strip() for n in line_edge_names]

            self._line_label_mask = torch.tensor(
                [ln in self.current_labels.columns for ln in line_edge_names_lower], dtype=torch.bool
            )

            # Diagnostic: unlike bus voltage, this coverage check didn't
            # exist before. If most lines show up here, current_labels.csv
            # column names don't match Neo4j's line-segment r.name values
            # (e.g. an OpenDSS "Line." prefix) and the loss is silently
            # being computed on only a handful of lines.
            missing_lines = [
                orig for orig, present in zip(line_edge_names, self._line_label_mask.tolist())
                if not present
            ]
            if missing_lines:
                print(f"NOTE: {len(missing_lines)}/{len(line_edge_names)} line segments have no "
                      f"current label and will be excluded from the loss: {missing_lines[:5]}"
                      f"{'...' if len(missing_lines) > 5 else ''}")

            self.current_labels = self.current_labels.reindex(columns=line_edge_names_lower)
            self.has_current_labels = True

        self.n_timesteps = n_timesteps or len(self.load_profiles)

        # Per-item pandas .iloc on a ~2700-column frame is slow and runs
        # 1000+ times per epoch. The frames never change after this point,
        # so convert once to contiguous float32 arrays and index those.
        self._load_np = self.load_profiles.to_numpy(dtype="float32")
        self._voltage_np = self.voltage_labels.to_numpy(dtype="float32")
        self._current_np = (
            self.current_labels.to_numpy(dtype="float32")
            if self.has_current_labels else None
        )

        # Every relation ingested from Neo4j points from source toward load
        # (Source -> Transformer -> Bus -> Bus (line_segment) -> Load), and
        # HeteroConv/SAGEConv only propagates along edge direction. That
        # means a Load's real-time kW — the only per-timestep-varying input
        # in the whole graph — has no directed path back to any Bus, and no
        # Bus/line has a path to anything downstream of it at any number of
        # hops. merge=False keeps same-type (bus, bus) relations as two
        # separate directed relations (line_segment + rev_line_segment)
        # rather than symmetrizing them, so the forward line_segment
        # relation used by the current-prediction head — and its y /
        # label_mask, which are only sized for the forward direction — is
        # left untouched.
        self._to_undirected = ToUndirected(merge=False)

        # Cached template graph, built lazily on first get() and rebuilt
        # whenever normalization is re-fit. See _get_template().
        self._template = None

    # ------------------------------------------------------------------
    def _get_template(self):
        """Everything in a snapshot that does NOT vary with the timestep.

        The topology, every static node feature and every edge attribute
        are identical across all timesteps — only a Load's kW column and
        the two label vectors change. Previously get() rebuilt all of it
        per item: cloning every static tensor, re-running normalization
        over all node types and edge attrs, and re-running ToUndirected
        (which sorts/coalesces every relation) 1000+ times per epoch.

        Now that work happens once and the result is reused. get() builds
        a fresh HeteroData that points at these same tensors; PyG's
        collation concatenates into new storage, so sharing is safe (the
        original code already shared self.edge_attr this way). Numerically
        this is identical to the previous behaviour.
        """
        if self._template is not None:
            return self._template

        data = HeteroData()
        for ntype, feat in self.static_x.items():
            data[ntype].x = feat.clone()

        # Placeholder kW column so the load feature width matches what
        # get() will produce; the values are overwritten every item.
        n_load = self.static_x["load"].size(0)
        data["load"].x = torch.cat(
            [self.static_x["load"], torch.zeros(n_load, 1)], dim=1
        )

        for rel, ei in self.edge_index.items():
            data[rel].edge_index = ei
        for rel, ea in self.edge_attr.items():
            data[rel].edge_attr = ea

        if getattr(self, "norm_stats", None) is not None:
            for ntype in data.node_types:
                mean, std = self.norm_stats[ntype]
                data[ntype].x = (data[ntype].x - mean) / std
            for rel in data.edge_types:
                if rel in self.edge_attr:
                    mean, std = self.norm_stats[rel]
                    data[rel].edge_attr = (data[rel].edge_attr - mean) / std

        data = self._to_undirected(data)
        self._template = data
        return data

    # ------------------------------------------------------------------
    @staticmethod
    def _collapse_phase_columns_to_bus(df):
        bus_of = {col: col.split(".")[0] for col in df.columns}
        out = {}
        for bus_name in sorted(set(bus_of.values()), key=lambda b: (len(b), b)):
            cols = [c for c, b in bus_of.items() if b == bus_name]
            out[bus_name] = df[cols].min(axis=1)
        return pd.DataFrame(out)

    # ------------------------------------------------------------------
    def _fetch_static_graph(self, session, include_upstream):
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
        self._idx_of = idx_of

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

        # ---- Edges Mapping ----
        line_rows = session.run("""
            MATCH (b1:Bus)-[r:LINE_SEGMENT]->(b2:Bus)
            OPTIONAL MATCH (b2)-[:CONFORMS_TO]->(lc:LineCode)
            RETURN b1.id AS src, b2.id AS dst, r.name AS name,
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

        # ---- Downstream-load membership matrix ----
        # M[b, l] = 1 iff Load l sits at or below Bus b in the feeder.
        # Line current is essentially P_downstream(t) / (sqrt(3) * V), so
        # M @ kw_t gives each bus the exact real-time demand it carries.
        # The GNN cannot reconstruct this on its own: with 55 loads across
        # ~900 buses and 2 conv layers, most buses have no load within
        # reach, so their inputs were byte-identical at every timestep and
        # the best possible prediction was each line's mean (which is what
        # the model had converged to).
        #
        # Built by walking UP from each load's bus to all of its ancestors,
        # using only LINE_SEGMENT direction — no dependency on UPSTREAM_OF
        # existing or on what exactly it means. 55 loads * ~900 buses, so
        # this is instant and happens once.
        parents = {}
        for r in line_rows_valid:
            parents.setdefault(idx_of["bus"][r["dst"]], []).append(idx_of["bus"][r["src"]])

        M = torch.zeros(len(self.node_id["bus"]), len(self.node_id["load"]))
        for r in supply_rows:
            if r["src"] not in idx_of["bus"] or r["dst"] not in idx_of["load"]:
                continue
            l_i = idx_of["load"][r["dst"]]
            stack, seen = [idx_of["bus"][r["src"]]], set()
            while stack:
                u = stack.pop()
                if u in seen:
                    continue
                seen.add(u)
                stack.extend(parents.get(u, ()))
            for u in seen:
                M[u, l_i] = 1.0
        self.downstream_load_matrix = M

        # ---- Cumulative source-to-bus series impedance (static) ----
        # LineCode impedance is per-unit-length, so a segment's series R, X is
        # length*r1, length*x1. Summing these along the path from the source
        # to a bus gives its ELECTRICAL distance from the substation — the
        # quantity voltage drop scales with, and a far better position feature
        # than the raw hop count already in the bus vector. O(n): one walk down
        # the radial tree. This is topology only (no current), so it is an
        # ingredient the model combines with real-time demand, not the answer.
        #
        # Paired with the downstream-demand feature, both factors of the
        # dominant voltage-drop term (path impedance x demand) now live as
        # per-bus features, so the MLP voltage head can form their product
        # node-locally without needing extra message-passing reach.
        n_bus = len(self.node_id["bus"])
        seg_R, seg_X, children = {}, {}, {}
        for r in line_rows_valid:
            p = idx_of["bus"][r["src"]]
            c = idx_of["bus"][r["dst"]]
            children.setdefault(p, []).append(c)
            seg_R[c] = r["length"] * r["r1"]
            seg_X[c] = r["length"] * r["x1"]
        cumR, cumX = [0.0] * n_bus, [0.0] * n_bus
        # Roots = buses with no incoming line segment (fed by the transformer).
        queue = deque(i for i in range(n_bus) if i not in seg_R)
        visited = set(queue)
        while queue:
            u = queue.popleft()
            for c in children.get(u, ()):
                if c in visited:
                    continue
                cumR[c] = cumR[u] + seg_R.get(c, 0.0)
                cumX[c] = cumX[u] + seg_X.get(c, 0.0)
                visited.add(c)
                queue.append(c)
        cum_impedance = torch.tensor(list(zip(cumR, cumX)), dtype=torch.float)
        # Append to the static Bus features (x, y, hops, cumKW -> + cumR, cumX).
        self.static_x["bus"] = torch.cat([self.static_x["bus"], cum_impedance], dim=1)

        # Track line names aligned to edge_index for targets
        self.line_names = {
            ("bus", "line_segment", "bus"): [r["name"] for r in line_rows_valid]
        }

        # Edge features: length, phase one-hot, per-unit-length impedance
        # (r1/x1/r0/x0/c1/c0), and the ACTUAL series impedance of the segment
        # length*r1, length*x1. LineCode gives impedance per unit length, so
        # a segment's real impedance is length x that — the quantity that
        # drives its voltage drop (V_drop ~ I*(R+jX)). Providing r1/x1 and
        # length separately isn't enough for a linear edge encoder, which
        # can't form their product; supplying length*r1 and length*x1
        # directly hands the model the physical R and X per segment.
        self.edge_attr = {
            ("bus", "line_segment", "bus"): torch.tensor(
                [[r["length"]] + _phase_onehot(r["phases"]) +
                 [r["r1"], r["x1"], r["r0"], r["x0"], r["c1"], r["c0"],
                  r["length"] * r["r1"], r["length"] * r["x1"]]
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
        self.norm_stats = {}
        # Stats changed, so the cached normalized template is stale.
        self._template = None

        # Node Normalization
        for ntype, feat in self.static_x.items():
            if ntype == "load":
                continue
            mean = feat.mean(dim=0)
            std = feat.std(dim=0, unbiased=False).clamp(min=1e-6)
            self.norm_stats[ntype] = (mean, std)

        # Edge Normalization
        for rel, attr in self.edge_attr.items():
            mean = attr.mean(dim=0)
            std = attr.std(dim=0, unbiased=False).clamp(min=1e-6)
            self.norm_stats[rel] = (mean, std)

        static_load = self.static_x["load"]
        static_mean = static_load.mean(dim=0)
        static_std = static_load.std(dim=0, unbiased=False).clamp(min=1e-6)

        kw_train = torch.tensor(
            self.load_profiles.iloc[list(train_indices)].values, dtype=torch.float
        )
        kw_mean = kw_train.mean().unsqueeze(0)
        kw_std = kw_train.std().clamp(min=1e-6).unsqueeze(0)

        self.norm_stats["load"] = (
            torch.cat([static_mean, kw_mean]),
            torch.cat([static_std, kw_std]),
        )

        # Downstream-kW bus feature stats, fit on the training split.
        # Two columns, because they carry different things:
        #   abs  — one global mean/std, preserving proportionality between
        #          buses (a trunk really does carry 50x a lateral).
        #   dev  — per-bus mean/std, i.e. this bus's deviation from its own
        #          typical loading. The current target is per-line
        #          normalized, so the deviation is the directly matched
        #          input; without it, a lateral's tiny absolute swing has
        #          to drive a full-scale normalized output.
        # std clamped at 1e-3 kW so buses with no downstream load stay ~0
        # instead of amplifying float noise.
        dkw_train = kw_train @ self.downstream_load_matrix.t()  # [T_train, n_bus]
        self.norm_stats[("dkw_abs",)] = (
            dkw_train.mean().unsqueeze(0),
            dkw_train.std().clamp(min=1e-3).unsqueeze(0),
        )
        self.norm_stats[("dkw_dev",)] = (
            dkw_train.mean(dim=0),
            dkw_train.std(dim=0).clamp(min=1e-3),
        )

        # Line current TARGET normalization — per line, not global.
        # Current magnitude in a radial feeder varies by orders of
        # magnitude depending on a line's position (trunk vs. lateral),
        # unlike per-unit voltage which sits near 1.0 everywhere. A single
        # global mean/std makes the model chase the few high-current trunk
        # lines while effectively ignoring the rest, so each line instead
        # gets its own training-set statistics (its magnitude is fairly
        # consistent across timesteps since it's driven by topology).
        if getattr(self, "has_current_labels", False):
            train_currents = torch.tensor(
                self.current_labels.iloc[list(train_indices)].values, dtype=torch.float
            )  # [T_train, n_lines], NaN where a line has no matching label column
            valid = ~torch.isnan(train_currents)
            i_mean = torch.nan_to_num(train_currents.nanmean(dim=0), nan=0.0)
            centered = torch.where(valid, train_currents - i_mean, torch.zeros_like(train_currents))
            i_var = (centered ** 2).sum(dim=0) / valid.sum(dim=0).clamp(min=1)
            i_std = i_var.sqrt().clamp(min=1e-6)
            self.norm_stats[("current_target",)] = (i_mean, i_std)

        # Bus voltage TARGET normalization — per bus, same treatment the line
        # currents already get. Raw per-unit voltage sits in a very narrow band
        # (~0.98-1.05 across this feeder), so its MSE is numerically tiny —
        # ~1e-4 — while the per-line-normalized current loss is ~1e-1. In the
        # joint objective (loss_v + LAMBDA_CURRENT * loss_i) that put the two
        # terms ~60:1 apart, so essentially all gradient into the shared conv
        # layers was shaped by the current task and the voltage head was left
        # to whatever fell out. Normalizing each bus by its own training-split
        # mean/std puts both targets on a comparable scale, so LAMBDA_CURRENT
        # means what it looks like and the voltage head competes for capacity.
        #
        # Per-bus (not global) for the same reason as current: it makes the
        # target the bus's deviation from its own typical voltage, which is
        # the time-varying part the model actually has to learn.
        train_voltages = torch.tensor(
            self.voltage_labels.iloc[list(train_indices)].values, dtype=torch.float
        )  # [T_train, n_buses], NaN where a bus has no matching label column
        v_valid = ~torch.isnan(train_voltages)
        v_mean = torch.nan_to_num(train_voltages.nanmean(dim=0), nan=0.0)
        v_centered = torch.where(v_valid, train_voltages - v_mean, torch.zeros_like(train_voltages))
        v_var = (v_centered ** 2).sum(dim=0) / v_valid.sum(dim=0).clamp(min=1)
        # Floor is defensive only: the smallest real per-bus std in this
        # feeder is ~1.5e-4, so this never binds on labelled buses. It just
        # stops an unlabelled/all-NaN column (std 0) producing a divide-by-zero.
        v_std = v_var.sqrt().clamp(min=1e-6)
        self.norm_stats[("voltage_target",)] = (v_mean, v_std)

    # ------------------------------------------------------------------
    def len(self):
        return self.n_timesteps

    def get(self, idx):
        # Static structure (topology, static features, edge attrs, reverse
        # relations, normalization) is built once and reused — see
        # _get_template(). Only the three time-varying pieces below are
        # constructed per item.
        tpl = self._get_template()

        data = HeteroData()
        for ntype in tpl.node_types:
            data[ntype].x = tpl[ntype].x
        for rel in tpl.edge_types:
            data[rel].edge_index = tpl[rel].edge_index
            if "edge_attr" in tpl[rel]:
                data[rel].edge_attr = tpl[rel].edge_attr

        # Overwrite Load features: static columns come from the template
        # (already normalized), the kW column is normalized here with the
        # same stats the old code applied to it.
        kw_raw = torch.from_numpy(self._load_np[idx])
        kw_t = kw_raw.unsqueeze(1)
        if getattr(self, "norm_stats", None) is not None:
            kw_mean, kw_std = self.norm_stats["load"]
            kw_t = (kw_t - kw_mean[-1]) / kw_std[-1]
        data["load"].x = torch.cat([tpl["load"].x[:, :-1], kw_t], dim=1)

        # Real-time downstream demand per bus — the physical driver of line
        # current, appended as two extra Bus features. See the matrix build
        # in _fetch_static_graph and the stats in fit_normalization.
        dkw = self.downstream_load_matrix @ kw_raw  # [n_bus]
        if getattr(self, "norm_stats", None) is not None:
            a_mean, a_std = self.norm_stats[("dkw_abs",)]
            d_mean, d_std = self.norm_stats[("dkw_dev",)]
            dkw_cols = torch.stack([(dkw - a_mean) / a_std, (dkw - d_mean) / d_std], dim=1)
        else:
            dkw_cols = torch.stack([dkw, dkw], dim=1)
        data["bus"].x = torch.cat([tpl["bus"].x, dkw_cols], dim=1)

        # Bus voltage labels
        v_t = torch.from_numpy(self._voltage_np[idx])
        y_v = torch.where(self._bus_label_mask, v_t, torch.tensor(float("nan")))
        # Per-bus normalization, fit on the training split (see
        # fit_normalization). Falls back to raw Vpu if norm_stats hasn't been
        # fit yet (e.g. before time_split() is called).
        if getattr(self, "norm_stats", None) is not None and ("voltage_target",) in self.norm_stats:
            v_mean, v_std = self.norm_stats[("voltage_target",)]
            y_v = (y_v - v_mean) / v_std
        data["bus"].y = y_v
        data["bus"].label_mask = self._bus_label_mask.clone()

        # Line current labels
        line_rel = ("bus", "line_segment", "bus")
        if getattr(self, "has_current_labels", False):
            i_t = torch.from_numpy(self._current_np[idx])
            y_i = torch.where(self._line_label_mask, i_t, torch.tensor(float("nan")))
            # Per-line normalization, fit on the training split (see
            # fit_normalization). Falls back to raw Amps if norm_stats
            # hasn't been fit yet (e.g. before time_split() is called).
            if getattr(self, "norm_stats", None) is not None and ("current_target",) in self.norm_stats:
                i_mean, i_std = self.norm_stats[("current_target",)]
                y_i = (y_i - i_mean) / i_std
            data[line_rel].y = y_i
            data[line_rel].label_mask = self._line_label_mask.clone()

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