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

> **Status note.** The model has moved from a voltage-only proof-of-concept to a
> joint voltage+current model. The current head only began learning once each bus
> was given an explicit *downstream real-time demand* feature (§5.3); with message
> passing alone it could not do better than predicting each line's mean. This
> document reflects that multi-task state. See §8 for the outstanding issues
> (voltage head starved by loss weighting; current head overfits held-out time).

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

## 5. GNN Data Pipeline (`pyg_dataset.py`)

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
| `Bus` | `x, y, hopsFromSource, cumulativeDownstreamKW` | **downstream real-time kW (2 columns: `abs` + `dev`)** |
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

## 6. Model Architecture — `FeederMultiTaskGNN`

```
x_dict (per node type)
   │
   ▼
Linear(in_dim[type] → hidden)  +  ReLU        # per-type input projection
   │
   ▼
HeteroConv({ relation: SAGEConv(hidden, hidden) for each relation }, aggr="sum")  +  ReLU
   │
   ▼
HeteroConv({ relation: SAGEConv(hidden, hidden) for each relation }, aggr="sum")
   │
   ├─────────────────────────────► Linear(hidden → 1) on Bus embeddings ──► predicted Vpu per Bus
   │
   └─ for each (bus, line_segment, bus) edge:
        concat[ h[src] , h[dst] , edge_attr ] ──► MLP(2·hidden + edge_attr_dim → hidden → 1) ──► predicted Amps per line
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

**Two output heads.** The model is multi-task:

- **Voltage head (node-level).** `Bus` embeddings pass through a `Linear(hidden →
  1)` to predict per-unit voltage per bus.
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
  (`LAMBDA_CURRENT = 0.5`). The voltage term is computed over `label_mask`-
  selected `Bus` nodes; the current term over `label_mask`-selected forward
  line-segment edges, in **normalized** space (the target is already per-line
  normalized by the dataset). Both masks exclude entries with no matching label
  column.
- **Optimizer:** Adam, learning rate `1e-3`.
- **Batching:** `torch_geometric.loader.DataLoader` batches multiple
  independent graph snapshots together (standard PyG mini-batching via
  block-diagonal adjacency), `batch_size=32`.
- **Split:** chronological 70/15/15 (train/val/test), normalization fit on
  train only.
- **Validation loop:** wired in — both heads are evaluated on the held-out
  split each epoch. Current is logged in real Amps² (un-normalized) so it is
  directly comparable to the trivial baseline; voltage is logged in Vpu².

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

### Current results (multi-task, downstream-demand feature, 2 layers, 20 epochs)

| Head | Train MSE (final) | Val MSE (best / final) | vs. baseline |
|---|---|---|---|
| Current (A²) | ~4.0 | ~8.9 / ~13–17 | **beats baseline ~5–8×** on val |
| Voltage (Vpu²) | ~0.0003 | ~0.001 | still ~7× *worse* than its own baseline |

The current head is genuinely learning time-varying behavior for the first time
(from a pure mean-predictor to ~4 A² train). Two caveats are visible in the logs
and carried into §8: validation current is **noisy and not trending down** (best
val current occurs mid-run, not at the final epoch), and the voltage head is
being **starved** by the loss weighting.

---

## 8. Known Limitations and Next Steps

- **Voltage head is starved by the loss weighting.** With `loss_v ≈ 3e-4` and
  `LAMBDA_CURRENT · loss_i` now a real, non-trivial quantity, the current term
  dominates the gradient by roughly three orders of magnitude, and the voltage
  head coasts on "per-unit voltage barely moves" rather than learning — it sits
  ~7× worse than its own trivial baseline. The fix is to put the two targets on
  a comparable scale (e.g. per-bus normalize the voltage target the same way the
  current target already is) and/or retune `LAMBDA_CURRENT`. *Deferred for now
  by decision — recorded here so it isn't lost.*
- **Current head overfits held-out time.** Train current slides steadily while
  validation current bounces (≈8.9–18.8 A²) with no downward trend after the
  first few epochs, and the best validation epoch is not the last. Keeping
  best-val weights (early stopping / checkpointing) and light regularization
  (weight decay, dropout) are the obvious next levers. Still a large win over
  baseline, but generalization is the current ceiling.
- **De-energized lines dilute the current loss.** ~205 of 905 lines carry
  ~1e-10 A; they are harmless (their per-line std is clamped) but make up ~¼ of
  the loss terms while carrying no signal. Excluding them from `label_mask` would
  make both the loss and the reported MSE reflect only real, energized lines.
- **No temporal component yet.** Each snapshot is treated as i.i.d.; a GRU/LSTM
  or attention layer over a window of past snapshots (e.g. PyG Temporal's
  `A3TGCN`/`DCRNN`) would let the model forecast ahead rather than only
  reconstruct the current minute.
- **`UPSTREAM_OF` is unused by default** — an `INCLUDE_UPSTREAM` flag exists for
  ablation. It is the single largest compute cost when enabled (~20–30k edges vs.
  905 line-segment edges, doubled by `ToUndirected` and walked by every conv
  layer). Note that the analytic downstream-demand feature (§5.3) now delivers
  the long-range demand signal `UPSTREAM_OF` was meant to help propagate, so its
  marginal value should be re-measured rather than assumed.
- **`CONFORMS_TO` is attached to the line's target bus, not the line segment
  itself** — a minor schema quirk (a consequence of Neo4j not allowing
  relationship-to-relationship links) that's compensated for at query time by
  joining through the bus, but worth keeping in mind if querying the graph
  directly outside this pipeline.
