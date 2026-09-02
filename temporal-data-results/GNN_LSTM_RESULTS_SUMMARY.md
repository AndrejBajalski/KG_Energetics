# KG-GNN + LSTM temporal forecasting — progress report

> **Status note:** the checkpointing/early-stopping setup has now been run
> 5 times with different random seeds, giving a stable enough picture to
> report a range rather than a single number. Earlier sections of this
> report (persistence comparison, training behavior) still describe the
> original single 20-epoch run for context, but the
> [headline result](#headline-result-checkpointed-test-evaluation-n5-runs)
> below supersedes those numbers as the current best evidence.

## Headline result: checkpointed test evaluation (n=5 runs)

Five independent runs (differing only in random seed — model init and
training-batch order) of the early-stopping setup (patience 8 epochs),
each evaluated once on the held-out test split using that run's own
best-validation checkpoint:

| Run | Best epoch | Test I-MSE | Test I-R² | vs. persistence (0.985847) |
|---|---:|---:|---:|---:|
| 1 | 4 | 0.7864 | 0.3815 | 20.2% |
| 2 | 5 | 0.8458 | 0.3348 | 14.2% |
| 3 | 3 | 0.7693 | 0.3949 | 22.0% |
| 4 | 2 | 0.7965 | 0.3736 | 19.2% |
| 5 | 1 | 0.8017 | 0.3695 | 18.7% |
| **Mean ± stdev** | — | **0.7999 ± 0.028** | **0.3709 ± 0.022** | **18.85% ± 2.88pp** |

Every run beat persistence, with no overlap between the improvement range
(14.2–22.0%) and zero — this looks like a genuine, repeatable effect
rather than a favorable single seed. Voltage MSE was not aggregated across
runs (still to do; see next steps) but stayed in the same healthy
~0.0005–0.0009 per-unit² range as the original single run across all 5.

One notable pattern: the best epoch varies a lot across runs (1, 2, 3, 4,
5) and is consistently early. Current overfits fast and the specific point
at which it peaks isn't very predictable run-to-run — early stopping is
doing real, necessary work on every single run, not occasionally catching
an unlucky one.


## Executive summary

This strategy reuses the project's existing Neo4j knowledge graph (Bus,
Load, LineCode, SubstationTransformer, Source nodes, with topology-derived
features such as `hopsFromSource` and `cumulativeDownstreamKW`) as the
encoder backbone, rather than treating the forecasting problem as a flat
CSV time series. A shared heterogeneous GNN encoder runs once per minute
across a 30-minute window; the resulting per-bus embedding sequence feeds
an LSTM, which forecasts the next 15 minutes of bus voltage directly, and
separately drives an edge-level head that forecasts per-line current.

The most consequential finding so far is about the current head, not the
final numbers: an early version that predicted **absolute** current values
could not beat a trivial persistence baseline (predict "current stays the
same") at all — it plateaued at 0.72–0.78 (normalized units) against
persistence's 0.72. Two changes were needed before it started actually
beating persistence:

1. Reformulating the head to predict a **residual/delta** from the last
   observed current rather than the absolute value, so the model starts
   training from persistence-level performance instead of having to
   rediscover it from scratch.
2. Making the `line_segment` relation's convolution **impedance-aware**
   (`TransformerConv(edge_dim=...)` in place of plain `SAGEConv`), since
   plain `SAGEConv` never consumes `edge_attr` — the impedance values
   (`r1, x1, r0, x0, c1, c0`) folded into each line edge were present in
   the graph but invisible to message passing until this change.

After both changes, the model beat the persistence baseline on validation
current error for the first several epochs, before overfitting set in (see
[Training behavior](#training-behavior)). With checkpointing and early
stopping now in place, the best-validation checkpoint (epoch 4) beats
persistence by **20.2%** on current MSE on the held-out test split — the
first clean, fully apples-to-apples win for this strategy (see
[Headline result](#headline-result-checkpointed-test-evaluation)).

## Model / strategy

- **Encoder:** per-node-type linear projection, then two rounds of
  message passing. Most relations (`supplies_power`, `conforms_to`,
  `has_feeder_head`, `feeds`) use `SAGEConv`; `line_segment` uses
  `TransformerConv` with `edge_dim` set to the line edge-feature width
  (length + phase one-hot + 6 impedance values), so impedance can
  influence the Bus embeddings.
- **Temporal head:** the encoder runs with shared weights across all 30
  minutes in a window, producing a per-bus embedding sequence, which an
  `LSTM` consumes. Voltage is forecast directly (all 15 future values
  from the LSTM's final hidden state in one shot — "direct multi-horizon,"
  not autoregressive).
- **Current head:** edge-level MLP on `(bus, line_segment, bus)`, taking
  `concat(h_src, h_dst, edge_attr)` and predicting a **delta** added to
  the last observed (normalized) current for that line.
- **Normalization:** voltage labels are left raw (per-unit, already
  tightly clustered near 1.0, no normalization needed). Current labels
  are z-scored **per line**, not globally — a line near the substation
  can carry currents an order of magnitude larger than a tail line, so a
  single global mean/std would force very different-scale signals into
  the same normalized space. The per-line std floor is set relative to
  the other lines' typical variance (5th percentile), not a fixed
  epsilon, to avoid blowing up near-flat lines' normalized targets.

## Evaluation setup

| Item | Value |
|---|---:|
| Available data | One simulated day, 1,440 one-minute timesteps (same European LV feeder / OpenDSS simulation as the CSV-only model) |
| Historical window | 30 minutes |
| Forecast horizon | 15 minutes |
| Line current targets | 905 lines |
| Split | Chronological 70% / 15% / 15% (train / val / test) |
| Hidden size | 64 |
| Batch size | 8 |
| Current loss weight (λ) | 0.3 |
| Early stopping patience | 8 epochs of no val improvement *(newly added, not yet validated with a completed run)* |

All normalization (per-node-type feature stats, per-line current stats) is
fit on the training split only, matching the project's existing convention
for feature normalization.

## Comparison with the persistence baseline

Current error is still reported in **normalized (per-line z-scored)
units**, not Amps — converting back to physical units is an open item (see
[Recommended next steps](#recommended-next-steps)). Voltage error is
already in physical units (per-unit), since it was never normalized.

| Metric | Persistence (repeat last value) | GNN+LSTM |
|---|---:|---:|
| Current train MSE (normalized) | 0.7214 | 0.4599–0.7217 across epochs* |
| Current val MSE (normalized) | 1.3214 | 1.1694 (best epoch) – 1.7839 (worst epoch)* |
| Current val R² | — (not applicable to a fixed baseline) | 0.32 (best epoch) – −0.04 (worst epoch)* |

\* *The model beat persistence on validation current error for roughly the
first 6–14 epochs of the 20-epoch run, then degraded past it as
overfitting took hold — see below. No stable single number exists yet
without the checkpointing fix.*

There is no voltage persistence baseline computed yet for direct
comparison, unlike the current metric above — worth adding for symmetry.

## Where the model struggles

Not yet analyzed. The CSV-only model's report identifies specific
difficult buses and lines (e.g. buses in the 600/800 series, `line27`,
`line31`, `line161`); no equivalent per-bus/per-line error breakdown has
been run for this strategy yet. This is a clear and fairly cheap next
step, and cross-referencing the two models' hard cases against each other
would be a useful sanity check (see next steps).

## Training behavior

In the most recent completed 20-epoch run:

- **Voltage** stayed stable and healthy across the whole run — val MSE
  ranged roughly 0.0004–0.001 (per-unit²) throughout, with no clear
  overfitting trend.
- **Current** showed a textbook overfitting curve. Train current R² rose
  steadily from 0.05 to 0.40 over 20 epochs (train MSE 0.72 → 0.46), but
  validation R² *peaked at epoch 1* (0.32) and *declined* from there,
  briefly turning negative around epochs 16–17, while val MSE correspondingly
  rose from its best point (1.169 at epoch 1) back up above the persistence
  baseline (1.784 at epoch 16). The best validation epoch for current was
  effectively epoch 1 of 20 — far earlier than the run's fixed length,
  which is exactly the scenario the newly-added checkpointing and early
  stopping are meant to catch.

This is consistent with the project's small effective dataset: one
simulated day yields roughly 900 overlapping training windows, and because
the split is chronological, train/val/test each cover *disjoint hours of
the day* — the model never sees val's or test's specific hours during
training at all. Current, which tracks load and therefore varies
meaningfully by time of day, is far more exposed to this than voltage,
which stays near 1.0 pu regardless of hour due to regulation.

### A data-engineering note worth keeping on record

Line names required case-insensitive matching between Neo4j (original CSV
casing, e.g. `LINE1`) and OpenDSS's `dss.Lines.AllNames()` output (which
lowercases object names, e.g. `line1`). Silently mismatched names produced
an entirely-empty mask rather than an error, which briefly looked like a
converged near-zero loss before being traced back to `masked_mse`'s
empty-mask fallback path. Worth remembering for any future script that
joins Neo4j-sourced names against OpenDSS-sourced names.

## Interpretation and limitations

1. **Voltage forecasting is solid and consistent across every run so
   far.** Current forecasting only recently started beating persistence at
   all, and only for part of a run.
2. **The single-day chronological split is a likely major contributor to
   current's overfitting**, since train/val/test never share the same
   time-of-day — the same limitation the CSV-only model's report
   identifies independently.
3. **The residual/delta reformulation and the impedance-aware conv were
   both introduced in the same change**, so their individual contributions
   haven't been cleanly separated yet.
4. **R² and the persistence baseline are different reference points** —
   R² compares against the pooled mean, persistence compares against the
   last observed value. A positive R² does not by itself imply beating
   persistence, and both should be read together.
5. **No physical-unit (Amp) error reporting exists yet for current** —
   everything above is in normalized per-line z-score space. This means
   there's currently no way to directly compare this model's current
   accuracy against the CSV-only LSTM's 2.143 A MAE / 4.914 A RMSE table.
6. **No per-bus/per-line error breakdown has been performed.**
7. **This snapshot predates the checkpointing/early-stopping run** — the
   numbers above come from a fixed 20-epoch run that did not select the
   best epoch, so they understate what the same architecture can already
   achieve with a properly early-stopped run.

## Recommended next steps

1. ~~Let the checkpointing + early-stopping run complete~~ — **done**, see
   [Headline result](#headline-result-checkpointed-test-evaluation). Next:
   repeat across multiple seeds/runs to check how consistent the 20.2%
   improvement is, since it currently rests on a single run.
2. Denormalize current predictions back to Amps using the saved per-line
   mean/std from `fit_normalization`, so results are directly comparable
   to the CSV-only LSTM's MAE/RMSE-in-Amps table.
3. Add a voltage persistence (and/or trend) baseline on the test split,
   for the same reason the CSV-only report treats persistence/trend as
   mandatory context rather than optional — current now has this, voltage
   still doesn't.
4. Generate multiple simulated days and split by whole day rather than by
   adjacent minutes within one day, so train/val/test each cover a full
   daily cycle — addressing the time-of-day confound identified above.
5. Cleanly ablate the residual/delta current formulation against the
   impedance-aware `TransformerConv` change, to attribute the current
   head's improvement between the two rather than crediting both jointly.
6. Add a per-bus and per-line error breakdown, and cross-check any
   consistently hard buses/lines against the CSV-only LSTM's findings
   (buses `639/633/626/616/813`, lines `27/31/161`, etc.) and against
   Neo4j topology (`hopsFromSource`, `cumulativeDownstreamKW`) — agreement
   between two independently-built models on where the feeder is hardest
   to predict would be a meaningful signal either way.
7. Consider a Huber/MAE training objective or peak-weighted loss for
   current, mirroring the CSV-only report's suggestion, given current's
   error is likely dominated by a small number of high-current lines and
   rare abrupt events.

## Result artifacts

- `scripts/temporal_dataset.py` — windowing dataset + per-line current
  label loading
- `scripts/temporal_model.py` — encoder, LSTM head, residual current head
- `scripts/train_temporal.py` — training loop, checkpointing, early
  stopping, test evaluation
- `scripts/persistence_baseline.py` — the persistence baseline used above
- `results/train_run_<timestamp>.txt` — full per-epoch log for a given run
- `results/best_model_<timestamp>.pt` — best-val-checkpoint model weights
  for a given run
