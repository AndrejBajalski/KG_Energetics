# KG_Energetics: Knowledge-Graph-Backed GNN for LV Feeder Voltage Prediction

## 1. Problem Statement

Distribution-level voltage prediction on a low-voltage (LV) feeder is fundamentally a
graph problem: the voltage at any bus is a function of the network's topology
(how far it sits from the source, what impedance lies along the path) and the
time-varying load drawn by every consumer downstream. Traditional power-flow
solvers (OpenDSS, pandapower) compute this exactly by iterating on Kirchhoff's laws,
but they are computationally expensive to run repeatedly at scale — e.g. for
real-time monitoring, contingency screening across thousands of scenarios, or
fast what-if analysis.

This project builds a pipeline that:

1. Represents the physical feeder (buses, lines, transformers, loads, cable
   specifications) as a **Neo4j knowledge graph**, preserving the electrical
   topology as an explicit, queryable graph structure.
2. Generates **ground-truth labels** (bus voltages, line currents) by running
   an actual OpenDSS power-flow simulation across a full day of per-minute
   load profiles.
3. Trains a **heterogeneous Graph Neural Network (GNN)** that learns to predict
   bus voltage directly from graph structure and load conditions — a fast,
   learned approximation of what the physics solver computes exactly, with the
   long-term goal of enabling near-instant voltage estimation without a full
   power-flow solve for every scenario.

The graph itself is not just a data store here — it is the model's inductive
bias. Every relationship ingested into Neo4j (line segments, cable
conformance, feeder hierarchy, load connections) becomes a distinct edge type
the GNN passes messages along, so the fidelity of the knowledge graph directly
determines what the model is capable of learning.

---

## 2. Source Data

The project uses the **IEEE European Low Voltage Test Feeder**, a standard
benchmark distribution network distributed in OpenDSS-native CSV format.

| File | Contents | Key columns |
|---|---|---|
| `Buscoords.csv` | Bus locations (arbitrary local x/y, not GPS) | `Busname, x, y` |
| `LineCodes.csv` | Cable impedance profiles per conductor class | `Name, nphases, R1, X1, R0, X0, C1, C0, Units` |
| `Lines.csv` | Physical line segments connecting two buses | `Name, Bus1, Bus2, LineCode, Length, Units, Phases` |
| `Loads.csv` | Individual consumer connections | `Name, Bus, phases, numPhases, Connection, Model, kV, kW, PF, Yearly` |
| `LoadShapes.csv` | Metadata for per-load time-series profiles | `Name, npts, minterval, File, useactual` |
| `Source.csv` | Substation source parameters (INI-style, not tabular) | `Voltage, pu, ISC3, ISC1` |
| `Transformer.csv` | Substation transformer | `Name, bus1, bus2, phases, kV_pri, kV_sec, Conn_pri, Conn_sec, MVA, %XHL, % resistance` |
| `Load_Profiles/Load_profile_N.csv` | Raw per-minute demand curve, one file per LoadShape | `time, mult` (1440 rows = 1 day at 1-min resolution) |

A few properties of this dataset shaped the whole pipeline:

- **Per-phase, not per-bus.** The feeder is intentionally unbalanced — loads
  are single-phase, scattered across A/B/C — so OpenDSS reports voltage per
  **node** (bus-phase terminal), not per bus. Downstream code has to
  explicitly aggregate `"1.1"`, `"1.2"`, `"1.3"` back to one value for bus `"1"`.
- **`Source.csv` is not tabular** — it's an INI-style `[Source]` block, parsed
  with `configparser`, not `pandas.read_csv`.
- **Raw load profile files are inconsistently delimited** (tab, whitespace),
  requiring a robust reader (`pd.read_csv(..., sep=None, engine="python")`)
  rather than a fixed delimiter.

---

## 3. Data Ingestion — Neo4j Knowledge Graph (`load_data.py`)

### 3.1 Graph Schema

```
(:Source)-[:FEEDS]->(:SubstationTransformer)-[:HAS_FEEDER_HEAD]->(:Bus)
                                                                      │
                                                            [:LINE_SEGMENT]
                                                                      │
                                                                      ▼
(:Bus)-[:SUPPLIES_POWER]->(:Load)          (:Bus)-[:CONFORMS_TO]->(:LineCode)
(:Bus)-[:UPSTREAM_OF]->(:Bus)   (derived, transitive closure of LINE_SEGMENT)
```

| Node label | Identifying property | Key attributes |
|---|---|---|
| `Bus` | `id` | `x`, `y`, `hopsFromSource`, `cumulativeDownstreamKW` |
| `Load` | `id` | `numPhases`, `targetPhase`, `baseKW`, `kV`, `model`, `powerFactor`, `connectionType`, `profileURI` |
| `LineCode` | `name` | `nphases`, `r1`, `x1`, `r0`, `x0`, `c1`, `c0`, `units` |
| `SubstationTransformer` | `id` | `phases`, `ratedMVA`, `primaryKV`, `secondaryKV`, `primaryConn`, `secondaryConn`, `xhl`, `resistancePct` |
| `Source` | `id` (`'SourceBus'`) | `nominalKV`, `pu`, `isc3`, `isc1` |

### 3.2 Ingestion Logic

`load_data.py` reads each CSV, cleans column whitespace, and uses idempotent
`MERGE` (not `CREATE`) for every node and relationship — re-running the script
never duplicates data, it only updates properties. This matters for a
knowledge graph that gets iteratively corrected: several real bugs were found
and fixed during ingestion, most notably a case-sensitivity mismatch on
`LineCode` columns (`r1` vs. `R1`) and a Transformer schema mismatch
(`kV_primary` vs. the real `kV_pri`) that had been silently writing `null`
onto every affected node without raising an error — a reminder that Cypher's
`toFloat()`/`toString()` on a missing key returns `null` rather than failing.

### 3.3 Derived Relationships and Features

Two graph-native features are precomputed once at ingestion time rather than
recomputed at training time:

- **`UPSTREAM_OF`** — the full transitive closure of `LINE_SEGMENT` from the
  feeder head. Every bus gets a direct edge to every downstream bus, giving
  long-range reachability that a shallow GNN (2–3 message-passing layers)
  couldn't otherwise capture in one pass. Deliberately **excluded from the
  default GNN edge set** (see §5) because it is O(n²) and would dominate
  message passing with redundant shortcuts — kept available as an opt-in.
- **`Bus.hopsFromSource`** and **`Bus.cumulativeDownstreamKW`** — computed
  once via Cypher aggregation over `UPSTREAM_OF`/`LINE_SEGMENT` and
  `SUPPLIES_POWER`, then stored as plain node properties. These become
  static GNN input features, sparing the model from having to learn
  "distance from source" and "downstream demand" purely from raw topology.

An earlier version of this step attempted `MERGE (r1)-[:NEXT_SEGMENT]->(r2)`
to connect two `LINE_SEGMENT` *relationships* directly — this is invalid in
Neo4j's property graph model (relationships can only connect nodes, never
other relationships) and was removed; line-to-line adjacency for the GNN is
instead computed directly from `edge_index` in Python, where it belongs as a
training-time construct rather than a stored graph fact.

---

## 4. Label Generation — OpenDSS Simulation (`generate_labels.py`)

Ground-truth labels cannot be derived from graph structure alone — they
require solving Kirchhoff's laws. `generate_labels.py` rebuilds the circuit
from the same CSVs directly as OpenDSS text commands (`New Line...`,
`New Transformer...`, `New Load...`), sets up per-load `LoadShape` objects
using the actual per-minute profile files, and calibrates nominal voltage
levels with:

```
Set VoltageBases=[11, 0.416, 0.23]
CalcVoltageBases
```

`CalcVoltageBases` traces the circuit topology from the source through every
transformer winding to assign each bus its nominal kV — required before any
per-unit (Vpu) quantity can be computed correctly.

The simulation runs in `Daily` mode at 1-minute resolution (`Set stepsize=1m`,
`number=1440`), solving the power flow once per minute and reading back:

- **`bus_voltages.csv`** — one column per bus-phase node (`Circuit.AllBusMagPu`),
  per-unit voltage magnitude.
- **`line_currents.csv`** — one column per line, max magnitude across its
  phase conductors (`CktElement.CurrentsMagAng`).

These two files are the GNN's regression targets.

---

## 5. GNN Data Pipeline (`pyg_dataset.py`)

### 5.1 Snapshot Framing

Rather than modeling time explicitly inside the GNN, the graph is treated as
**1440 independent snapshots** sharing one fixed topology: static graph
structure (edge_index, edge_attr, node types) is built once from Neo4j, and
each snapshot `t` only varies in (a) the Load nodes' current kW draw and
(b) the Bus nodes' solved voltage label. This is the standard
"static-topology, time-varying-features" framing for spatio-temporal graph
problems, and it means every minute of the day becomes one training example.

### 5.2 Heterogeneous Graph Construction

The dataset builds a `torch_geometric.data.HeteroData` object per snapshot,
preserving the KG's actual relationship types rather than flattening
everything into one homogeneous edge set:

| Relation | Edge features (`edge_attr`) |
|---|---|
| `(bus, line_segment, bus)` | `length`, phase one-hot (A/B/C), **`r1, x1, r0, x0, c1, c0`** folded in from `LineCode` via `CONFORMS_TO` |
| `(bus, supplies_power, load)` | — |
| `(bus, conforms_to, linecode)` | — |
| `(transformer, has_feeder_head, bus)` | — |
| `(source, feeds, transformer)` | — |
| `(bus, upstream_of, bus)` *(opt-in)* | — |

Folding LineCode impedance directly into `line_segment` edges (via a single
Cypher query joining `LINE_SEGMENT` and `CONFORMS_TO`) was a deliberate fix —
an earlier version only carried line *length*, meaning the model could not
distinguish a thick low-impedance cable from a thin high-impedance one of the
same length, despite impedance being the dominant physical driver of voltage
drop.

### 5.3 Node Features

| Node type | Static features | Dynamic (per-timestep) |
|---|---|---|
| `Bus` | `x, y, hopsFromSource, cumulativeDownstreamKW` | — |
| `Load` | `kV, powerFactor, model,` phase one-hot, connection one-hot (wye/delta) | current-minute `kW` (appended column) |
| `LineCode` | `nphases, r1, x1, r0, x0, c1, c0` | — |
| `SubstationTransformer` | `phases, ratedMVA, primaryKV, secondaryKV, xhl, resistancePct` | — |
| `Source` | `nominalKV, pu, isc3, isc1` | — |

### 5.4 Labels and Masking

`bus_voltages.csv` columns are per bus-*phase* (`"1.1"`, `"1.2"`, `"1.3"`),
not per bus, so they are collapsed to one value per `Bus` node by taking the
**minimum** Vpu across present phases — the conservative choice, since a
single sagging phase should register as a violation on that bus even if the
other phases look fine. Any `Bus` node with no matching voltage column is
tracked in a `label_mask` and excluded from the training loss entirely,
rather than being silently trained against a filled zero/NaN value.

### 5.5 Chronological Splitting and Normalization

`time_split()` divides the 1440 minutes into train/val/test **by time order**
(e.g. first 70% / next 15% / final 15%), never shuffled — consecutive minutes
are highly correlated, so a random split would leak future information into
training and produce an overly optimistic validation score.

Feature normalization (`fit_normalization`) is fit **exclusively on the
training split** and applied identically to val/test, avoiding any
distributional leakage. This step was necessary in practice: without it,
training MSE plateaued around **17.7** — the raw feature set mixes bus
coordinates (order 10⁵), cumulative downstream kW (order 10²), and line
impedance (order 10⁰), and an unnormalized first linear layer produces
unstable activations regardless of what the model is otherwise learning.
Z-score normalization per node type brought training MSE down to
**~0.00007** within ~20 epochs (RMSE ≈ 0.008 Vpu, roughly 0.8% error).

One subtlety in the normalization step: `Source` and `SubstationTransformer`
each have exactly one node instance in this feeder, so computing `std()`
with the default unbiased (n−1) estimator over a single sample divides by
zero and returns `NaN` — which a `.clamp(min=1e-6)` does **not** fix, since
clamp only bounds real numbers. The fix was switching to
`std(dim=0, unbiased=False)`, giving `0` (correctly clamped to `1e-6`)
instead of `NaN` for single-instance node types.

---

## 6. Model Architecture — `VoltageHeteroGNN`

```
x_dict (per node type)
   │
   ▼
Linear(in_dim[type] → hidden)  +  ReLU        # per-type input projection
   │
   ▼
HeteroConv({ relation: SAGEConv(hidden, hidden) for each relation }, aggr="sum")
   │
   ▼
ReLU
   │
   ▼
HeteroConv({ relation: SAGEConv(hidden, hidden) for each relation }, aggr="sum")
   │
   ▼
Linear(hidden → 1)     # applied to Bus embeddings only
   │
   ▼
predicted Vpu per Bus
```

**Per-type input projection.** Node types arrive with different feature
dimensionalities (`Bus`: 4, `Load`: 9 after the dynamic kW column, `Source`:
4, etc.). A separate `nn.Linear` per type maps everything into a common
`hidden` dimension before any message passing occurs, which is required for
`HeteroConv` to combine embeddings across relations.

**Two rounds of heterogeneous message passing.** `HeteroConv` wraps one
`GraphSAGE` convolution per relation type and aggregates the results per
destination node type (`aggr="sum"`) — this means, for example, that a
`Bus` node's updated embedding after layer 1 is a function of messages
arriving from both `line_segment` neighbors (other Buses) *and*
`has_feeder_head` neighbors (its Transformer), combined additively, with
**separate learned weight matrices per relation** rather than one shared
transformation for every edge regardless of its physical meaning.

**Handling a source-only node type.** `Source` never appears as a
*destination* in any relation (it only ever sends messages via `feeds`) —
`HeteroConv` therefore drops it from its output dictionary after every
layer, since it never gets an update rule applied. Left unhandled, this
causes the second `HeteroConv` layer to look up a now-missing `"source"` key
when building `feeds` messages for `Transformer`, crashing deep inside
PyG's aggregation code with an opaque `AttributeError` on a `None` tensor.
The fix merges each layer's output back into the running `x_dict`
(`x_dict = {**x_dict, **layer_output}`), so `Source` simply retains its
post-projection embedding unchanged across both layers — which is also the
electrically correct behavior, since the source is the root of the feeder
and has nothing genuinely "upstream" of it to aggregate from.

**Output head.** Only `Bus` embeddings are passed through the final linear
layer, since bus voltage is the sole prediction target; `Load`, `LineCode`,
`Transformer`, and `Source` embeddings exist purely to inform Bus
predictions through message passing and are not directly supervised.

---

## 7. Training Procedure

- **Loss:** MSE, computed only over `label_mask`-selected `Bus` nodes per
  snapshot (excludes any bus with no matching voltage-label column).
- **Optimizer:** Adam, learning rate `1e-3`.
- **Batching:** `torch_geometric.loader.DataLoader` batches multiple
  independent graph snapshots together (standard PyG mini-batching via
  block-diagonal adjacency), `batch_size=32`.
- **Split:** chronological 70/15/15 (train/val/test), normalization fit on
  train only.
- **Result:** training MSE converges from ~0.10 (epoch 0) to ~0.00007 by
  epoch ~20 and plateaus — indicating the model has largely fit the training
  distribution. Validation-loss tracking (`val_ds`) is the next required
  step to confirm this generalizes to held-out time rather than memorizing
  the repeated static topology.

---

## 8. Known Limitations and Next Steps

- **No temporal component yet.** Each snapshot is treated as i.i.d.; a
  GRU/LSTM or attention layer over a window of past snapshots (e.g. PyG
  Temporal's `A3TGCN`/`DCRNN`) would let the model forecast ahead rather
  than only reconstruct the current minute.
- **Validation/test evaluation not yet wired into the main training loop** —
  currently only training MSE is tracked live; held-out evaluation needs to
  be added to distinguish genuine learning from overfitting to a
  structurally-repetitive graph.
- **`UPSTREAM_OF` is unused by default** — worth an ablation to check whether
  including it as a long-range shortcut relation improves accuracy on deep
  feeder branches enough to justify its edge-count cost.
- **Line current labels (`line_currents.csv`) are generated but not yet
  used** as a second/auxiliary prediction target.
- **`CONFORMS_TO` is attached to the line's target bus, not the line segment
  itself** — a minor schema quirk (a consequence of Neo4j not allowing
  relationship-to-relationship links) that's compensated for at query time
  by joining through the bus, but worth keeping in mind if querying the
  graph directly outside this pipeline.
