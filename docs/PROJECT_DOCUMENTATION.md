# KG_Energetics: Knowledge-Graph-Backed GNN for LV Feeder State Estimation

## 1. Problem Statement

Distribution-level state estimation on a low-voltage (LV) feeder is fundamentally a
graph problem: the voltage at any bus, and the current on any line, is a function
of the network's topology (how far it sits from the source, what impedance lies
along the path) and the time-varying load drawn by every consumer downstream.
Traditional power-flow solvers (OpenDSS, pandapower) compute this exactly by
iterating on Kirchhoff's laws, but they are computationally expensive to run
repeatedly at scale — e.g. for real-time monitoring, contingency screening across
thousands of scenarios, or fast what-if analysis.

This project builds a pipeline that:

1. Represents the physical feeder (buses, lines, transformers, loads, cable
   specifications) as a **Neo4j knowledge graph**, preserving the electrical
   topology as an explicit, queryable graph structure.
2. Generates **ground-truth labels** (bus voltages, line currents) by running
   an actual OpenDSS power-flow simulation across a full day of per-minute
   load profiles.
3. Trains a **heterogeneous, multi-task Graph Neural Network (GNN)** that learns
   to predict both bus voltage (a node-level target) and line current (an
   edge-level target) directly from graph structure and load conditions — a
   fast, learned approximation of what the physics solver computes exactly, with
   the long-term goal of enabling near-instant feeder state estimation without a
   full power-flow solve for every scenario.

> **Status note.** The model is a joint voltage+current GNN. Its headline outcome
> is a **contrast**: the **line-current** head works well (test R² ≈ 0.83), while
> the **bus-voltage** head sits at ≈ 0 and no feature or architecture change moved
> it. This is a real finding — current is a *local, near-linear* quantity a GNN
> learns easily; voltage is a *global, nonlinear* one with tiny variance that it
> does not. See §7 for the results progression and §8 for the full interpretation
> (including why R² is a harsh lens for voltage and why reporting pu RMSE may be
> the right call).

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
  default GNN edge set** (see §5, §8) because it is O(n²) and would dominate
  message passing with redundant shortcuts — kept available as an opt-in. Note
  that the same ancestor relationship is now also used, separately, to build the
  analytic downstream-demand feature (§5.3), which delivers the long-range demand
  signal without adding those edges to the message-passing graph.
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

## 5. GNN Data Pipeline (`hetero_gnn_dataset.py`)

### 5.1 Snapshot Framing

Rather than modeling time explicitly inside the GNN, the graph is treated as
**1440 independent snapshots** sharing one fixed topology: static graph
structure (edge_index, edge_attr, node types) is built once from Neo4j, and
each snapshot `t` only varies in (a) the Load nodes' current kW draw,
(b) the Bus nodes' derived downstream real-time demand (§5.3), and (c) the two
solved labels — bus voltage and line current. This is the standard
"static-topology, time-varying-features" framing for spatio-temporal graph
problems, and it means every minute of the day becomes one training example.

### 5.2 Heterogeneous Graph Construction

The dataset builds a `torch_geometric.data.HeteroData` object per snapshot,
preserving the KG's actual relationship types rather than flattening
everything into one homogeneous edge set:

| Relation | Edge features (`edge_attr`) |
|---|---|
| `(bus, line_segment, bus)` | `length`, phase one-hot (A/B/C), `r1, x1, r0, x0, c1, c0` (per-unit-length impedance from `LineCode` via `CONFORMS_TO`), **`length·r1, length·x1`** (actual per-segment series R, X) — 12 dims |
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

The last two edge features, **`length·r1` and `length·x1`, are precomputed
products** giving each segment's *actual* series resistance and reactance.
`LineCode` stores impedance *per unit length*, so a segment's real impedance is
`length × r1`; supplying `length` and `r1` separately isn't enough for a linear
edge encoder, which cannot form their product. This is a legitimate physical
constant of the segment (not the voltage answer), so precomputing it is fair.

**These edge features are only useful if message passing actually reads them —
which required changing the convolution (see §6).** `SAGEConv`, the original
choice, *ignores* `edge_attr` entirely, so for a long time the impedance,
length, and per-line phases reached only the current head (which concatenates
`edge_attr` explicitly) and never influenced any node embedding. The voltage
head was therefore structurally blind to the quantities voltage drop depends on.
The `line_segment` relations now use an edge-aware convolution (`GATv2Conv` with
`edge_dim`) so these features flow into the bus embeddings.

**Reverse relations are added at snapshot-build time** via
`ToUndirected(merge=False)`. Every relation ingested from Neo4j points strictly
from source toward load (`Source → Transformer → Bus → Bus (line_segment) →
Load`), and `HeteroConv`/`SAGEConv` only propagates *along* edge direction.
Without reverse edges, a Load's real-time kW — the only per-timestep-varying
input in the entire graph — has no directed path to reach *any* Bus, at any
number of hops, so every snapshot would present byte-identical inputs to the
buses and the model could not learn anything time-varying. `ToUndirected`
therefore adds `rev_supplies_power`, `rev_line_segment`, etc. so demand can
flow back upstream. `merge=False` is deliberate: it keeps the same-type
`(bus, line_segment, bus)` relation as two *separate* directed relations
(forward + `rev_line_segment`) rather than symmetrizing them into one, so the
forward relation the current head reads — along with its `y` and `label_mask`,
which are sized for the forward direction only — is left untouched.

### 5.3 Node Features

| Node type | Static features | Dynamic (per-timestep) |
|---|---|---|
| `Bus` | `x, y, hopsFromSource, cumulativeDownstreamKW`, **`cumR, cumX`** (cumulative source-path impedance) | **downstream real-time kW (2 columns: `abs` + `dev`)** |
| `Load` | `kV, powerFactor, model,` phase one-hot, connection one-hot (wye/delta) | current-minute `kW` (appended column) |
| `LineCode` | `nphases, r1, x1, r0, x0, c1, c0` | — |
| `SubstationTransformer` | `phases, ratedMVA, primaryKV, secondaryKV, xhl, resistancePct` | — |
| `Source` | `nominalKV, pu, isc3, isc1` | — |

#### The downstream real-time demand feature (why the current head works)

The single most important change for the current-prediction task was giving each
`Bus` an explicit, per-timestep **downstream real-time demand** feature. Line
current is essentially `P_downstream(t) / (√3 · V)`, so the demand carried
through a line is the direct physical driver of its current. The GNN could not
reconstruct this on its own: there are only 55 loads across ~900 buses, and with
2 message-passing layers a bus can only "see" loads within 2 hops. The
overwhelming majority of buses have no load in reach, so their inputs were
identical at every minute of the day and the best the current head could do was
predict each line's mean — which is exactly where it had converged (train
MSE ≈ 47, matching the trivial mean-predictor baseline; see §8).

The feature is computed analytically rather than learned:

- **Membership matrix `M` (built once, at ingestion).** `M[b, l] = 1` iff
  Load `l` sits at or below Bus `b` in the feeder. It is built by walking *up*
  the `LINE_SEGMENT` tree from each load's bus to all of its ancestors — using
  only `LINE_SEGMENT` direction, so it does **not** depend on `UPSTREAM_OF`
  existing or on its exact semantics. At 55 loads × ~900 buses this is instant.
- **Per snapshot,** the bus-level demand vector is one small matrix–vector
  product, `dkw_t = M @ kw_t`, giving each bus the exact real-time kW it
  carries. No extra edges, no extra message passing — training stays fast.

It is appended as **two** normalized columns rather than one, because they carry
complementary information:

- **`abs`** — one global mean/std, which preserves proportionality *between*
  buses (a trunk really does carry ~50× a lateral).
- **`dev`** — per-bus mean/std, i.e. this bus's deviation from its *own* typical
  loading. Because the current *target* is per-line normalized (§5.5), the
  deviation is the input that directly matches it; without it, a lateral's tiny
  absolute swing would have to drive a full-scale normalized output.

A worthwhile sanity check when regenerating the graph: `M @ (nominal load kW)`
should closely match the `cumulativeDownstreamKW` already stored on each Bus
node. A disagreement means the membership matrix is wrong and nothing downstream
of it can be trusted.

#### Cumulative source-path impedance (a feature for the voltage head)

By analogy to the demand feature, each `Bus` also gets its **cumulative series
impedance from the source** — `cumR = Σ length·r1` and `cumX = Σ length·x1`
summed over the segments on the path from the substation down to that bus.
Voltage drop scales with this *electrical distance*, which is a far better
position feature than the raw `hopsFromSource` count (which weights a short thick
cable and a long thin one equally). It is computed once with a single O(n)
breadth-first walk down the radial tree from the source, adds no edges, and is
appended to the static Bus vector (so it flows through normalization and the
template cache automatically).

Crucially this is **static topology, not the voltage answer** — it contains no
current and no time dependence. Paired with the per-timestep downstream-demand
feature, both factors of the dominant voltage-drop term (`path impedance ×
demand`) now live as *per-bus* features, so the voltage head can in principle
form their product node-locally without needing extra message-passing reach.
Whether the head *can* form that product depends on its architecture — see the
MLP head in §6, and the honest outcome in §8 (it did not, in the end, unlock
voltage).

### 5.4 Labels and Masking

**Bus voltage (node target).** `bus_voltages.csv` columns are per bus-*phase*
(`"1.1"`, `"1.2"`, `"1.3"`), not per bus, so they are collapsed to one value
per `Bus` node by taking the **minimum** Vpu across present phases — the
conservative choice, since a single sagging phase should register as a violation
on that bus even if the other phases look fine. Any `Bus` node with no matching
voltage column is tracked in a `label_mask` and excluded from the training loss
entirely, rather than being silently trained against a filled zero/NaN value.

**Line current (edge target).** `line_currents.csv` is loaded and aligned to the
forward `(bus, line_segment, bus)` edge order, giving the model a second,
edge-level regression target. Two practical wrinkles were handled here:

- **Name normalization.** OpenDSS and Neo4j disagree on line-name casing/prefixes
  (`LINE1` vs. `line1`), so both sides are lower-cased and stripped before
  matching, and a coverage diagnostic prints how many of the graph's line
  segments actually matched a label column. (In the current dataset this is a
  clean 905/905; the check exists so a silent mismatch — where the loss is
  computed on only a handful of lines — is caught immediately rather than
  masquerading as poor accuracy.) Lines with no matching column are tracked in a
  per-edge `label_mask` and excluded from the loss, mirroring the bus scheme.
- **De-energized stubs.** ~205 of the 905 lines carry ~1e-10 A (open/unused
  segments). They stay in the label set — their per-line target normalization
  (§5.5) clamps their std so they contribute nothing destabilizing — but they
  are worth remembering, since they are roughly a quarter of the loss terms and
  contribute essentially no signal.

### 5.5 Chronological Splitting and Normalization

`time_split()` divides the 1440 minutes into train/val/test **by time order**
(e.g. first 70% / next 15% / final 15%), never shuffled — consecutive minutes
are highly correlated, so a random split would leak future information into
training and produce an overly optimistic validation score.

Feature normalization (`fit_normalization`) is fit **exclusively on the
training split** and applied identically to val/test, avoiding any
distributional leakage. This step was necessary in practice: without it,
training loss plateaued badly — the raw feature set mixes bus coordinates
(order 10⁵), cumulative downstream kW (order 10²), and line impedance (order
10⁰), and an unnormalized first linear layer produces unstable activations
regardless of what the model is otherwise learning. Z-score normalization per
node type (and per edge relation) resolved it.

One subtlety in the normalization step: `Source` and `SubstationTransformer`
each have exactly one node instance in this feeder, so computing `std()`
with the default unbiased (n−1) estimator over a single sample divides by
zero and returns `NaN` — which a `.clamp(min=1e-6)` does **not** fix, since
clamp only bounds real numbers. The fix was switching to
`std(dim=0, unbiased=False)`, giving `0` (correctly clamped to `1e-6`)
instead of `NaN` for single-instance node types.

**Line-current targets are normalized *per line*, not globally.** Current
magnitude in a radial feeder spans orders of magnitude by position — a trunk
line carries the whole feeder while a lateral carries one house — unlike
per-unit voltage, which sits near 1.0 everywhere. A single global mean/std would
make the model chase the few high-current trunk lines and effectively ignore the
rest, so each line gets its own training-split mean/std (computed with a
NaN-aware reduction so unmatched columns don't poison the statistics). Its
magnitude is fairly consistent across time because it is driven by topology, so
per-line statistics are stable. Predictions are un-normalized back to real Amps
for logging (so the reported "MSE I" is in Amps², directly comparable to the
trivial baseline in §8), while the *loss* is computed in normalized space.

**Bus-voltage targets are normalized *per bus*, added later for the same
reason.** Raw per-unit voltage sits in a tiny band (~0.98–1.05), so its MSE is
numerically minuscule (~1e-4) while the per-line-normalized current loss is
~1e-1. In the joint objective `loss_v + LAMBDA_CURRENT · loss_i` that put the two
terms ~60:1 apart at the operating point (and ~10⁴:1 at initialization), so
essentially all gradient into the shared conv layers was shaped by the current
task and the voltage head was *starved* (§8). Normalizing each bus by its own
training-split mean/std puts both targets on a comparable scale, so
`LAMBDA_CURRENT` genuinely controls their relative weight. As with current,
voltage predictions are un-normalized to real Vpu for logging and R².

**Performance: the static graph is built once, not per snapshot.** Topology,
static node features, edge attributes, normalization, and the reverse-edge
construction are all identical across all 1440 snapshots — only the load-kW
column, the downstream-demand feature, and the two label vectors change. An
earlier version rebuilt the entire `HeteroData` (cloning every static tensor,
re-running normalization over all node/edge types, and re-running `ToUndirected`)
on *every* `get()` call, i.e. ~1000+ times per epoch. This is now cached as a
template graph built on first access and invalidated whenever normalization is
re-fit; each `get()` reuses it and only attaches the time-varying pieces. The
label frames are also converted from pandas to contiguous NumPy arrays once at
init, since per-item `.iloc` on the ~2700-column voltage frame was itself a
measurable cost. Neither change affects the numbers — they are pure throughput.

---

## 6. Model Architecture — `FeederMultiTaskGNN` (`hetero_gnn_model.py`)

```
x_dict (per node type)
   │
   ▼
Linear(in_dim[type] → hidden)  +  ReLU        # per-type input projection
   │
   ▼
HeteroConv({                                                       # layer 1
   line_segment / rev_line_segment : GATv2Conv(hidden, edge_dim)   #  edge-aware
   all other relations             : SAGEConv(hidden)              #  no edge_attr
}, aggr="sum")  +  ReLU
   │
   ▼
HeteroConv({ … same per-relation mix … }, aggr="sum")             # layer 2
   │
   ├──► MLP(hidden → hidden → 1) on Bus embeddings ──► predicted Vpu (normalized) per Bus
   │
   └─ for each (bus, line_segment, bus) edge:
        concat[ h[src] , h[dst] , edge_attr ] ──► MLP(2·hidden + edge_attr_dim → hidden → 1) ──► predicted Amps (normalized) per line
```

**Per-type input projection.** Node types arrive with different feature
dimensionalities (`Bus`: 8 — 4 base + `cumR`/`cumX` + 2 downstream-demand
columns; `Load`: 9 after the dynamic kW column; `Source`: 4; etc.). A separate
`nn.Linear` per type maps everything into a common `hidden` dimension before any
message passing occurs, which is required for `HeteroConv` to combine embeddings
across relations. Feature widths are read from a sample at construction time, so
these projections resize automatically when features are added.

**Two rounds of heterogeneous message passing, with edge-aware convolutions
on the line segments.** `HeteroConv` wraps one convolution per relation type and
aggregates the results per destination node type (`aggr="sum"`), with **separate
learned weight matrices per relation**. The convolution *type* is chosen per
relation:

- **`line_segment` and `rev_line_segment` use `GATv2Conv` with `edge_dim`**, an
  attention-based conv that *consumes* `edge_attr`. This is what lets the length,
  phase, and impedance features (§5.2) actually flow into the bus embeddings.
  `HeteroConv` routes `edge_attr_dict` per-relation, so only these convs receive
  it.
- **Every other relation keeps `SAGEConv`** (it carries no `edge_attr`).

The original design used `SAGEConv` for *all* relations — and since `SAGEConv`
ignores edge features, the carefully-assembled per-segment impedance never
reached any node embedding. Switching the line relations to an edge-aware conv
was the fix; see §8 for its measured effect (it clearly helped **current**;
voltage was unmoved).

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

**Two output heads.** The model is multi-task:

- **Voltage head (node-level).** `Bus` embeddings pass through an **MLP**
  (`Linear → ReLU → Linear`) to predict (normalized) per-unit voltage per bus. It
  was upgraded from a bare `Linear` because a single linear layer can only form
  weighted *sums* of the embedding, whereas the dominant voltage-drop term is a
  *product* (`path impedance × demand`); a hidden ReLU layer gives the head the
  capacity to approximate that product from the two per-bus ingredients now in
  the embedding. (In practice this added capacity did not help — and appears to
  have hurt, by feeding voltage's overfitting; see §8.)
- **Current head (edge-level).** For each forward `(bus, line_segment, bus)`
  edge, the source and destination bus embeddings are concatenated with the
  line's `edge_attr` (length, phases, impedance) and fed through a small MLP
  (`2·hidden + edge_attr_dim → hidden → 1`) to predict the line's current. It
  reads **only the forward** relation, never the `rev_line_segment` one that
  `ToUndirected` adds, so its output stays aligned with the forward-sized `y`
  and `label_mask`. `Load`, `LineCode`, `Transformer`, and `Source` embeddings
  are not directly supervised — they exist purely to inform the two heads
  through message passing.

**On message-passing depth.** A third convolution layer was trialed (giving
3 hops of reach) and then reverted: it was never shown to help, and — crucially
— the reason the current head was failing turned out not to be reach at all but
the *absence of any time-varying bus feature* (§5.3). Once the analytic
downstream-demand feature was added, 2 layers were sufficient for the current
head to learn. Depth remains a lever worth revisiting, but only against a
measured baseline rather than on the assumption that "more hops" is the fix.

---

## 7. Training Procedure

- **Loss:** joint MSE, `loss = loss_v + LAMBDA_CURRENT · loss_i`
  (`LAMBDA_CURRENT = 1.0`). **Both** targets are now per-entity normalized by the
  dataset (per-bus voltage, per-line current), so the two terms sit on a
  comparable scale and `LAMBDA_CURRENT` genuinely weights them (at 1.0 they are
  roughly equal). Each term is computed over its `label_mask`-selected entities.
- **Optimizer:** Adam, learning rate `1e-3`.
- **Batching:** `torch_geometric.loader.DataLoader` batches multiple
  independent graph snapshots together (standard PyG mini-batching via
  block-diagonal adjacency), `batch_size=32`.
- **Split:** chronological 70/15/15 (train/val/test), normalization fit on
  train only.
- **Early stopping + test evaluation.** Both heads are evaluated on the held-out
  val split each epoch. Training stops after `PATIENCE` epochs with no
  improvement in the combined normalized val loss, and the **best epoch's
  weights are restored** (validation current is noisy, so the final epoch is
  often not the best). The untouched **test** split — which never influenced
  training or model selection — is evaluated once at the end for the honest
  generalization number.
- **Metrics.** MSE is logged in real units (Vpu², Amps²) for comparison against
  the trivial baselines. The headline metric is **per-entity R²** — computed
  against *each bus's / each line's own mean* (equivalent to
  `sklearn.r2_score(multioutput="variance_weighted")`), **not** a single global
  mean. A global mean inflates R² by crediting the model for merely knowing that
  a trunk line carries far more current than a lateral; the per-entity reference
  measures only the time-varying part the model actually has to predict. `R² ≤ 0`
  means "no better than predicting that entity's own mean."

### Trivial baselines (the numbers that actually matter)

Because per-unit voltage barely moves and ~23% of lines carry ~0 A, raw MSE can
look small even when the model has learned nothing. Every run should therefore be
compared against a **trivial predictor** — predicting each bus's / each line's
training-split mean and ignoring the timestep entirely. Computed from the current
`data/Computed` CSVs (1440 timesteps, 70/15/15 split, 907 buses, 905 lines,
55 loads):

| Target | Trivial baseline (train) | Trivial baseline (val) |
|---|---|---|
| Line current | 47.23 A² | 78.25 A² |
| Bus voltage (phase-min) | 0.000042 Vpu² | — |

### Results (test-set R², best-epoch weights, 2 layers)

The table tracks **test-set per-entity R²** across the sequence of changes, which
is the clearest way to see what actually moved the needle:

| Configuration | test R² current | test R² voltage |
|---|---|---|
| SAGEConv, no edge features, `LAMBDA_CURRENT=0.5` | 0.73 | **0.17** |
| + edge-aware `GATv2` conv + impedance/`length·r` features, `LAMBDA=1.0` | **0.83** | 0.05 |
| + cumulative-impedance bus feature + MLP voltage head | 0.84 | −0.25 |

Reading this honestly:

- **Current is a genuine success (R² ≈ 0.83).** It climbed steadily as the model
  gained access to impedance. A line's current is a *local, near-linear* function
  of its downstream demand, and the GNN learns it well.
- **Voltage never worked.** Across every feature and architecture change its test
  R² stayed near zero and eventually went slightly negative — i.e. no better than
  predicting each bus's mean. Adding capacity (the MLP head, extra features) made
  it *worse*, because the voltage head **overfits** (train R² ≈ +0.5 vs test R²
  negative); more capacity gave it more ways to memorize the training timesteps.

The structural reason for the split is developed in §8. In short: **current is
local and near-linear (learnable); voltage is global and nonlinear with tiny
variance (not learnable by this class of model on this feeder).**

---

## 8. Findings, Limitations, and Next Steps

### The headline result: current is learnable, voltage is not

The clearest outcome of this project is a **contrast**, and it is a genuine
finding rather than a failure to fix:

- **Line current — R² ≈ 0.83 on held-out test.** A line's current is dominated by
  the real-time demand *directly downstream* of it (`I ≈ P_down / (√3·V)`) — a
  **local, near-linear** quantity. Once the model was given downstream demand
  (§5.3) and per-segment impedance (§5.2, §6), it learned current well.
- **Bus voltage — R² ≈ 0, and no intervention moved it.** Voltage at a bus is the
  impedance-weighted sum of currents over the *entire path back to the source* —
  a **global, nonlinear** quantity — and on this feeder it varies within a tiny
  band (~0.98–1.05 pu, ~0.7% std). Every change we tried (per-bus normalization,
  edge-aware impedance, cumulative-impedance feature, an MLP head) left test R²
  near or below zero.

Why this is the *expected* result, not a bug: current's driver is local and
roughly linear, so a 2-hop GNN captures it. Voltage's driver is a long-range
nonlinear integral, and its signal is tiny, so the mean is already an excellent
predictor and there is very little time-varying structure to learn. This
local-vs-global split is arguably the most interesting thing the project
demonstrates about what a heterogeneous GNN can and cannot learn on a feeder.

### On the voltage metric — R² is the wrong lens here

A caution for anyone reading the voltage numbers: **R² is scale-invariant but
punishing for low-variance targets.** Because voltage barely moves, "predict the
mean" is already accurate to ~0.5%, so beating it (positive R²) is very hard even
when the *absolute* error is small. The voltage RMSE is on the order of **~0.005
pu** — which, for a *fast surrogate for OpenDSS*, may be perfectly acceptable.
The recommendation is to **report voltage as pu RMSE / MAE, not R²**: judged on
absolute accuracy the voltage head may already be adequate for a surrogate use
case, even though its R² looks like failure. If the goal is instead violation
*detection*, note that this benign benchmark feeder never leaves nominal, so
there may be nothing to detect here — the interesting version of that problem
lives on a *stressed* feeder (high DER/EV) where voltage actually swings.

### Voltage overfits — capacity is the wrong lever

Diagnostically, the voltage head shows a large train/test gap (train R² ≈ +0.5,
test R² negative): it fits the training timesteps and fails to generalize. Every
*capacity-adding* change (the MLP head, extra features) made test R² worse. The
correct lever for an overfitting head is **regularization** (weight decay,
dropout) or *reducing* capacity (revert the MLP head to a linear one), not more
of it. This was recognized late; earlier changes pushed the wrong direction.

### Standing limitations and smaller items

- **De-energized lines dilute the current loss.** ~205 of 905 lines carry
  ~1e-10 A; they are harmless (their per-line std is clamped) but make up ~¼ of
  the loss terms while carrying no signal. Excluding them from `label_mask` would
  make the loss and reported metrics reflect only real, energized lines.
- **No temporal component.** Each snapshot is treated as i.i.d.; a GRU/LSTM or
  attention layer over a window of past snapshots (e.g. PyG Temporal's
  `A3TGCN`/`DCRNN`) would let the model *forecast* rather than only reconstruct
  the current minute. Two separate models explore this direction and are
  documented in **§9**: a spatio-temporal GNN (`temporal_model.py`) and a
  graph-free LSTM baseline (`lstm_model.py`).
- **`UPSTREAM_OF` is unused by default** — an `INCLUDE_UPSTREAM` flag exists for
  ablation, but it is O(n²) edges (~20–30k vs. 905 line segments, doubled by
  `ToUndirected`) and too expensive to justify. The cheaper long-range options
  for voltage (moderate depth + residual connections; the static
  cumulative-impedance feature already added) avoid that cost.
- **`CONFORMS_TO` is attached to the line's target bus, not the line segment
  itself** — a minor schema quirk (Neo4j does not allow relationship-to-
  relationship links) compensated for at query time by joining through the bus,
  but worth remembering when querying the graph directly.

### Suggested next steps

1. **Re-report voltage as pu RMSE/MAE** and decide whether it is already adequate
   for the intended use — this may resolve the "voltage problem" without any
   modeling change.
2. If pursuing voltage further, **regularize rather than add capacity**: revert
   the MLP voltage head to linear, add weight decay/dropout, and only then
   consider moderate depth (4–6 layers) with residual/jumping-knowledge
   connections (naive depth oversmooths on a ~20–30-hop feeder).
3. **Lean into the current result.** Current at R² 0.83 is a solid demonstration
   that a KG-backed heterogeneous GNN can act as a fast state estimator for the
   quantities that are locally determined.

---

## 9. Forecasting Variants — `temporal_model.py` and `lstm_model.py`

Everything in §1–§8 describes a **same-instant** model: `hetero_gnn_dataset.py`
pairs the loads at minute `t` with the labels at minute `t` (`_load_np[idx]` with
`_voltage_np[idx]` / `_current_np[idx]`), so `FeederMultiTaskGNN` is a fast
surrogate for the power-flow solve, **not** a forecaster. It has no lead time and
no temporal component.

Two separate scripts attack the forecasting problem instead: predict the next
**15 minutes** from the previous **30 minutes** of observations. Both are direct
multi-horizon models — all 15 future minutes come out of one forward pass rather
than by feeding a one-step prediction back in, which avoids compounding
recursive error.

### 9.1 `TemporalVoltageHeteroGNN` (`temporal_model.py`, trained by `train_temporal.py`)

The graph model extended over time. The encoder is the same `lin_in → conv1 →
conv2` shape as §6, factored out as `HeteroEncoder` so it can be applied to every
snapshot in a window **with shared weights**.

```
window of W=30 snapshots
   │   (shared HeteroEncoder applied per snapshot)
   ▼
bus embeddings  (B, W, n_bus, hidden)
   │   permute → (B·n_bus, W, hidden)
   ▼
LSTM(hidden, hidden, num_layers=1)  →  last hidden state h_T
   │
   ├──► Linear(hidden → horizon) ──────────► (B, 15, n_bus)  future Vpu
   │
   └──► concat[ h[src], h[dst], edge_attr ] ──► MLP(→ horizon)
            └─ + last observed current ───────► (B, 15, n_lines) future Amps
```

Points worth recording:

- **Edge-aware conv, different class.** `line_segment` uses `TransformerConv`
  with `edge_dim` here rather than the `GATv2Conv` of §6. Both consume
  `edge_attr`; the choice is incidental, not a measured improvement.
- **It bypasses `HeteroConv` deliberately.** `_apply_layer` loops over relations
  manually, because some installed `torch_geometric` versions pass an `edge_attr`
  kwarg to *every* relation's conv uniformly — including `SAGEConv`, which does
  not accept it (`SAGEConv.forward() got an unexpected keyword argument
  'edge_attr'`). The manual loop is version-proof.
- **The current head predicts a delta from persistence.** It outputs a correction
  that is *added to the last observed current* in the window, so the model starts
  at persistence for free (delta = 0 at initialization) and training is spent on
  the deviation — a far easier target than absolute amperes. The earlier version
  predicted the absolute value from bus embeddings and static `edge_attr` alone,
  so it had to reconstruct "what was the current a moment ago" indirectly before
  it could even match persistence.
- **`masked_mse` selects by boolean indexing**, never by multiplying with a 0/1
  mask: unlabeled targets are stored as `NaN`, and `NaN * 0` is still `NaN`, so
  multiplying would silently poison the loss.
- **Configuration:** `WINDOW=30`, `HORIZON=15`, `HIDDEN=64`, `BATCH_SIZE=8`,
  `LAMBDA_CURRENT=0.3`, `EPOCHS=20`, same chronological 70/15/15 split.

> **No results are recorded for this model.** `train_temporal.py` writes to a
> `results/` directory that does not exist in the repository, so this variant has
> not been evaluated. Any claim comparing it to the LSTM below is currently
> unsupported — running it is the obvious next step.

### 9.2 `CSVForecastLSTM` (`lstm_model.py`)

The deliberate control: **no graph, no Neo4j, no topology at all**. It reads the
same three CSVs as plain sequences, and exists to answer whether the knowledge
graph buys anything a purely temporal model cannot get on its own.

```
load history    (55)  ─ levels + Δ ─► Linear+GELU ─► 64 ┐
voltage history (906) ─ PCA→16 ─ levels + Δ ─► ... ─► 96 ├─► fusion 256→192
current history (905) ─ PCA→64 ─ levels + Δ ─► ... ─► 96 ┘   GELU + LayerNorm
                                                                  │
                                                     LSTM ×2, hidden 192, dropout 0.1
                                                                  │
                            last step + horizon embedding (15×192) → Linear+GELU+Dropout
                                                                  │
                          voltage/current delta heads (latent) → project back via basisᵀ
                                                                  │
                                    prediction = trend baseline + delta
```

- **PCA on the targets, fit on train only.** 906 buses and 905 lines are heavily
  correlated (one electrical network), so each is projected onto a
  training-fitted basis — **16** components for voltage (99.99% of standardized
  training variation), **64** for current (99.95%) — stored as registered
  buffers. The LSTM forecasts in that latent space and the prediction is
  projected back out, which cuts the output count and encourages spatially
  coherent predictions.
- **Levels *and* one-step differences** are fed to every encoder, making
  short-term direction and rate of change explicit rather than something the
  recurrence must infer.
- **Residual around a fitted trend baseline.** The baseline extrapolates the
  slope of the last 30 minutes with training-fitted coefficients; the network only
  predicts a correction on top. **Both delta heads are zero-initialized**, so
  training begins at exactly the baseline and can only improve from there — the
  same design instinct as the temporal GNN's persistence residual.
- **Configuration:** 721 040 trainable parameters, AdamW, lr `1e-4`, weight decay
  `1e-5`, batch 32, gradient clipping 1.0, `lambda_current=0.3`, LR halved on an
  8-epoch validation plateau. 963 / 202 / 203 train/val/test windows.

**Results** (held-out test, best epoch 6, from `lstm-data-results/`):

| Method | Voltage RMSE (Vpu) | Current RMSE (A) |
|---|---:|---:|
| Persistence (repeat last observation) | 0.004396 | 5.113 |
| Training-fitted trend baseline | **0.004034** | **4.700** |
| CSVForecastLSTM | 0.004050 | 4.914 |

LSTM voltage MAE 0.002989 Vpu (max abs 0.018693); current MAE 2.143 A (max abs
61.161 A). Against persistence it improves RMSE by 7.9% (voltage) and 3.9%
(current), and maximum error by 21.4% and 19.7%.

Read honestly: **the LSTM does not beat the fitted trend baseline on average.**
Its voltage RMSE is marginally worse and its current RMSE is ~4.6% worse; its
genuine advantage is in the *extreme* errors. A more complex model did not
automatically outperform a well-constructed statistical baseline, and one
simulated day is not enough evidence to claim it would.

### 9.3 A caution on comparing the three models

The numbers in §7 and §9.2 **are not comparable**:

- `FeederMultiTaskGNN` is scored as **per-entity R²** on *same-instant
  reconstruction* — no lead time.
- `CSVForecastLSTM` is scored as **RMSE/MAE** on a *15-minute forecast*.
- `TemporalVoltageHeteroGNN` is **unscored**.

Different tasks, different metrics, different target definitions. A like-for-like
statement about whether the knowledge graph helps requires running the temporal
GNN on the LSTM's exact split, horizon and metric. Until then the defensible
claim is narrow: the graph model reconstructs *current* well at R² ≈ 0.83, and a
graph-free LSTM forecasts 15 minutes ahead at roughly the accuracy of a fitted
trend extrapolation.

One convergence is worth noting: both forecasting models independently arrived at
the same structure — **predict a residual around a baseline that already encodes
persistence**. That is a strong hint that at one-minute resolution on this feeder
the signal is dominated by persistence, and that the interesting modeling question
is which model best captures the *deviation* from it.
