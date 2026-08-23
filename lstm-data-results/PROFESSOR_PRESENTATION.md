# Short-Term Voltage and Line-Current Forecasting with an LSTM

## Presentation purpose

This document is written as a slide-by-slide presentation script. The main
bullet points can be placed directly on slides, while the speaker notes provide
the explanation to give orally.

---

## Slide 1 — Title

### Short-Term Forecasting of Low-Voltage Feeder Conditions

**A CSV-based LSTM approach for predicting bus voltage and line current**

- IEEE European Low Voltage Test Feeder
- 30-minute observation window
- 15-minute direct forecast horizon
- Bus-voltage and line-current prediction

### Speaker notes

This work investigates whether a recurrent neural network can forecast the
short-term electrical state of a low-voltage distribution feeder. The model
uses recent load, voltage, and current observations to predict voltage at every
bus and current on every line for the following 15 minutes.

The objective is not to replace detailed power-flow simulation in every
setting. It is to test whether a trained temporal model can provide a fast
approximation of the feeder state for monitoring, planning, or early-warning
applications.

---

## Slide 2 — Motivation

### Why forecast the feeder state?

- Distribution feeders contain many buses and line segments.
- Voltage and current change as customer demand changes.
- Repeated power-flow calculations can become expensive in real-time systems.
- Short-term forecasts can support:
  - Voltage-quality monitoring
  - Congestion awareness
  - Operational planning
  - Anomaly and event detection

### Speaker notes

The physical condition of a distribution network depends strongly on changing
load demand. Operators are interested in whether voltage remains close to its
nominal value and whether lines carry unusually high current.

Traditional power-flow tools provide physically grounded answers, but they
must solve the network equations repeatedly. A learned forecasting model can
potentially provide results much faster after training. This is particularly
useful when forecasts are required every minute for hundreds of buses and
lines.

---

## Slide 3 — Research question

### Main question

> Can the next 15 minutes of bus voltage and line current be forecast from the
> previous 30 minutes of feeder observations using an LSTM?

### Inputs

- Active load demand in kW
- Bus voltage magnitude in per-unit
- Line current magnitude in amperes

### Outputs

- Voltage at 906 buses
- Current on 905 lines
- Predictions for forecast minutes 1 through 15

### Speaker notes

The task is multivariate and high-dimensional. At every forecast origin, the
model receives 30 time steps containing 55 load measurements, 906 bus-voltage
measurements, and 905 line-current measurements.

It then produces 15 future values for every one of the 906 buses and 905
lines. The model therefore makes more than 27,000 scalar predictions for each
input window.

---

## Slide 4 — Data source

### Simulated feeder data

- Network: IEEE European Low Voltage Test Feeder
- Simulation engine: OpenDSS
- Resolution: one minute
- Duration: one simulated day
- Total timesteps: 1,440

### Generated datasets

| Dataset | Dimensions | Meaning |
|---|---:|---|
| Load profiles | 1,440 × 55 | Active demand for each load, in kW |
| Bus voltages | 1,440 × 2,721 | Per-phase voltage magnitudes, in Vpu |
| Line currents | 1,440 × 905 | Maximum current magnitude per line, in A |

### Speaker notes

The data is synthetic rather than measured in the field. OpenDSS reconstructs
the feeder and solves one power flow per minute using the supplied daily load
profiles.

This gives a controlled ground truth: every voltage and current target is the
result of a physical power-flow calculation. However, conclusions are limited
to the simulated network and operating scenario until the model is evaluated
on additional days or real measurements.

---

## Slide 5 — Meaning of the targets

### Bus voltage

- Stored as per-unit voltage magnitude.
- `1.0 Vpu` represents nominal voltage.
- Each bus may contain one or more phase measurements.
- The minimum available phase voltage is used as the bus target.

### Line current

- Stored in amperes.
- One value per line and minute.
- Represents the maximum current magnitude returned for that line.

### Speaker notes

Per-unit voltage expresses voltage relative to the nominal voltage base. This
makes results comparable across different voltage levels. For example, an
error of 0.004 Vpu represents approximately 0.4% of nominal voltage.

The minimum phase voltage is selected because it provides a conservative view
of voltage quality. If one phase experiences a voltage drop, averaging all
phases could hide the problem.

The current target is also conservative because it retains the maximum
conductor magnitude for each line. It does not retain separate phase currents.

---

## Slide 6 — Forecasting formulation

### Sliding-window prediction

```text
Previous 30 minutes
    loads + voltages + currents
              │
              ▼
         LSTM forecaster
              │
              ▼
Next 15 minutes
    bus voltages + line currents
```

- Direct multi-horizon prediction
- All 15 future minutes are predicted together
- No future load values are provided
- No Neo4j or graph structure is used by this model

### Speaker notes

The model uses a rolling window. A training example consists of 30 minutes of
past observations and 15 minutes of future targets.

This is a direct forecasting model. It predicts every horizon in one forward
pass instead of predicting minute 1 and recursively feeding that prediction
back to obtain minute 2. Direct prediction avoids the accumulation of
recursive errors, although uncertainty still increases with forecast horizon.

The model is deliberately CSV-only. It provides a temporal baseline that can
later be compared with graph-based models that also use feeder topology.

---

## Slide 7 — Data preparation

### Processing steps

1. Align all three CSV files by timestep.
2. Collapse phase voltages to minimum voltage per bus.
3. Split data chronologically.
4. Fit all transformations on training data only.
5. Standardize every input and target feature.
6. Create 30-minute input and 15-minute target windows.

### Chronological split

| Split | Fraction | Forecast windows |
|---|---:|---:|
| Training | 70% | 963 |
| Validation | 15% | 202 |
| Test | 15% | 203 |

### Speaker notes

The split preserves time order. Randomly mixing adjacent minutes would cause
temporal leakage because neighboring observations are highly correlated.

The validation set controls checkpoint selection and early stopping, while
the final interval is used for evaluation. Normalization statistics, PCA
bases, and trend coefficients are fitted only from the training interval.

Validation and test histories may include observations immediately before the
target boundary. This is valid forecasting behavior because those observations
would already be available at prediction time.

---

## Slide 8 — Why dimensionality reduction was needed

### High-dimensional output problem

- 906 voltage targets
- 905 current targets
- 15 forecast horizons
- More than 27,000 outputs per example

### Training-only PCA representation

| Target | Original dimensions | PCA components | Training variation retained |
|---|---:|---:|---:|
| Voltage | 906 | 16 | 99.99% |
| Current | 905 | 64 | 99.95% |

### Speaker notes

Neighboring buses and lines do not behave independently. Their measurements
are strongly correlated because they belong to one electrical network and are
affected by shared loads.

Principal component analysis captures these common spatial patterns. Voltage
is especially low-rank: only 16 components retain almost all standardized
training variation across 906 buses. Current requires more components because
line loading varies more strongly across the feeder.

PCA reduces the number of independently learned outputs and encourages
spatially coherent predictions. It also lowers the risk of memorizing a single
day of data.

---

## Slide 9 — Model architecture

### LSTM forecasting pipeline

```text
Load history ───────► load encoder ──────┐
Voltage history ────► PCA + encoder ─────┼─► feature fusion
Current history ────► PCA + encoder ─────┘        │
                                                   ▼
                                           2-layer LSTM
                                                   │
                                      horizon-conditioned decoder
                                                   │
                                  voltage and current residuals
                                                   │
                                      add fitted trend baseline
```

### Final configuration

- Two LSTM layers
- Hidden dimension: 192
- Dropout: 0.1
- Trainable parameters: 721,040
- Optimizer: AdamW
- Initial learning rate: `1 × 10⁻⁴`

### Speaker notes

The three input groups are encoded separately. This prevents the small
55-dimensional load vector from being overwhelmed by the much larger voltage
and current vectors.

Both measurement levels and one-step differences are supplied to the
encoders. The differences expose short-term direction and rate of change.

The LSTM processes the fused 30-minute sequence. A horizon-conditioned decoder
then produces separate corrections for forecast minutes 1 through 15.

Rather than predicting the complete electrical state from zero, the network
predicts residual corrections around a trend baseline fitted from training
data. This makes the forecast stable before learning and lets the LSTM focus
on deviations from recent behavior.

---

## Slide 10 — Training procedure

### Training controls

- Objective: weighted voltage and current mean squared error
- Current-loss weight: 0.3
- Batch size: 32
- Gradient clipping: 1.0
- Learning-rate reduction on validation plateau
- Maximum epochs: 100
- Early-stopping patience: 20 epochs

### Training outcome

- Best validation checkpoint: epoch 6
- Training stopped: epoch 26
- Best normalized validation loss: 1.1312

### Speaker notes

Mean squared error was selected because large errors are operationally
important and should receive more weight than small deviations.

The current loss is weighted by 0.3 so that its standardized error does not
dominate voltage learning. Gradient clipping stabilizes the recurrent network.

The best checkpoint occurred early, at epoch 6. After that, training error
continued to fall but validation performance did not improve. Early stopping
therefore prevented continued overfitting and restored the best checkpoint.

---

## Slide 11 — Overall test results

### Final held-out performance

| Target | MAE | RMSE | Maximum absolute error |
|---|---:|---:|---:|
| Bus voltage | **0.002989 Vpu** | **0.004050 Vpu** | **0.018693 Vpu** |
| Line current | **2.143 A** | **4.914 A** | **61.161 A** |

### Interpretation

- Typical voltage error is approximately 0.3% of nominal voltage.
- Voltage RMSE is approximately 0.405% of nominal voltage.
- Typical current error is approximately 2.1 A.
- Large current events remain the most difficult cases.

### Speaker notes

The voltage results are relatively strong. An RMSE of 0.00405 Vpu corresponds
to approximately 0.405% of nominal voltage. The mean absolute error is below
0.003 Vpu.

Current is more difficult because feeder lines operate at very different
scales. Many lateral lines carry small currents, while upstream trunk lines
can carry much larger values. Abrupt load changes also create current events
that are difficult to infer from past observations alone.

---

## Slide 12 — Performance by forecast horizon

| Lead time | Voltage RMSE | Current RMSE |
|---:|---:|---:|
| 1 minute | 0.002320 Vpu | 3.030 A |
| 3 minutes | 0.003584 Vpu | 4.484 A |
| 5 minutes | 0.003829 Vpu | 4.722 A |
| 10 minutes | 0.004360 Vpu | 5.180 A |
| 15 minutes | 0.004691 Vpu | 5.618 A |

### Main pattern

- Voltage RMSE grows by 102% from minute 1 to minute 15.
- Current RMSE grows by 85% from minute 1 to minute 15.
- Accuracy is strongest for very-short-term operation.

### Speaker notes

The results show the expected relationship between lead time and uncertainty.
One-minute predictions are substantially more accurate than 15-minute
predictions.

The largest deterioration occurs in the first few minutes. After approximately
five minutes, error continues to increase but at a more gradual rate.

This suggests that the model is best suited to immediate operational
forecasting. Fifteen-minute predictions remain informative at feeder level,
but individual line events should be interpreted more cautiously.

---

## Slide 13 — Baseline comparison

### Why compare against simple forecasts?

A temporal model is only useful if it improves on repeating or extrapolating
recent observations.

| Method | Voltage RMSE | Current RMSE |
|---|---:|---:|
| Repeat the last observation | 0.004396 Vpu | 5.113 A |
| Training-fitted trend | **0.004034 Vpu** | **4.700 A** |
| LSTM | 0.004050 Vpu | 4.914 A |

### LSTM versus persistence

- 7.9% lower voltage RMSE
- 3.9% lower current RMSE
- 21.4% lower maximum voltage error
- 19.7% lower maximum current error

### Speaker notes

Persistence is a strong baseline for smooth one-minute electrical data. The
LSTM improves RMSE and maximum error relative to persistence, demonstrating
that it learns more than simply copying the last value.

The fitted trend baseline is even stronger on average in this one-day test.
Its voltage RMSE is slightly lower than the LSTM's, and its current RMSE is
about 4.6% lower. However, the LSTM produces smaller maximum errors.

This is an important scientific finding. A more complex neural network does
not automatically outperform a well-designed statistical baseline. More
diverse training data is required before claiming that the LSTM is generally
superior.

---

## Slide 14 — Error analysis

### Difficult voltage locations

- Highest per-bus RMSE: approximately 0.0051–0.0053 Vpu
- Concentrated around buses:
  - `639`, `633`, `626`, `616`
  - `813`, `804`

### Difficult current locations

- `line27` and `line31`: approximately 15.53 A RMSE
- Several additional high-current lines: approximately 15.21 A RMSE

### Speaker notes

The feeder-wide average hides significant spatial variation. A subset of
buses in the 600- and 800-series is harder to predict. Their clustering may
indicate a shared feeder branch, but this should be confirmed using the network
topology.

Current error is concentrated on lines carrying larger loads. This is expected
because their absolute changes are larger and multiple downstream loads affect
them simultaneously.

Per-bus and per-line evaluation is therefore necessary. Aggregate metrics
alone are not sufficient for operational decisions.

---

## Slide 15 — Largest observed forecast error

### Current event on `line161`

At target timestep 1366 and forecast minute 11:

```text
Predicted current:  3.76 A
Actual current:    64.92 A
Absolute error:    61.16 A
```

### Interpretation

- The event appears abrupt relative to recent observations.
- No future load forecast was available to the model.
- Rare current spikes dominate maximum-error statistics.
- Event-focused training may be required.

### Speaker notes

The largest error is not a small systematic bias. It is a missed current spike
on line 161. The model predicted a low-current state while OpenDSS produced
approximately 65 A.

This event illustrates the limitation of purely historical forecasting. If a
large load change is not visible in the previous 30 minutes, the model has no
direct information that it is about to occur.

Future work should inspect the underlying load profiles and feeder topology at
this timestep. Providing forecast load demand as an exogenous input may also
improve anticipation of these events.

---

## Slide 16 — What the model achieved

### Technical achievements

- Forecasts 1,811 feeder quantities simultaneously.
- Produces all 15 future horizons in one forward pass.
- Maintains voltage RMSE near 0.4% of nominal voltage.
- Improves RMSE and maximum error over persistence.
- Compresses spatial output while retaining almost all training variation.
- Uses 721 thousand trainable parameters.
- Automatically trains and evaluates on CUDA.
- Saves a reproducible checkpoint, configuration, metrics, and predictions.

### Speaker notes

The model demonstrates that a compact recurrent architecture can forecast the
complete feeder state at one-minute resolution. It predicts every bus and line
simultaneously, rather than training a separate model for each location.

The PCA representation is especially effective: it reduces output complexity
while preserving almost all observed training variation. The complete workflow
also includes chronological splitting, leakage-aware preprocessing, validation
checkpointing, held-out evaluation, and saved prediction artifacts.

---

## Slide 17 — Limitations

### Current limitations

1. Only one simulated day is available.
2. Adjacent sliding windows are highly correlated.
3. Results are not yet validated on another day or feeder.
4. Inputs require recent voltage and current observations.
5. Future load demand is not provided.
6. Phase-specific voltage/current behavior is collapsed.
7. No line ampacity data is available for overload classification.
8. The development holdout was inspected during model refinement.

### Speaker notes

These results should be interpreted as a proof of concept, not as evidence of
production readiness.

The most important limitation is data diversity. Thousands of overlapping
windows do not replace independent operating days. The model has seen only one
daily trajectory, so it has not demonstrated generalization to different
weather, seasons, customer behavior, switching states, faults, or topology.

Because the held-out interval was reviewed during model development, a new
untouched day or simulation scenario is required for a genuinely unbiased
final evaluation.

---

## Slide 18 — Recommended future work

### Next experimental steps

1. Generate multiple OpenDSS days and operating scenarios.
2. Split training, validation, and test data by complete day.
3. Reserve an untouched final scenario.
4. Add forecast load demand as an exogenous input.
5. Investigate line 161 and high-error feeder branches.
6. Report metrics per bus, line, and current scale.
7. Evaluate phase-specific targets.
8. Compare against graph-aware GNN and GNN-LSTM models.
9. Test Huber, MAE, and peak-weighted objectives.
10. Validate inference speed and reliability under real-time conditions.

### Speaker notes

The immediate priority is additional data. Multiple days would allow the LSTM
to learn recurring temporal patterns and would support a clean, day-level test
split.

Future load forecasts are likely to be valuable because load changes are the
physical driver of voltage and current changes. A graph-aware model may also
improve spatial generalization by using the known feeder topology rather than
learning spatial relationships only from historical correlation.

Finally, the loss function should be aligned with the operational objective.
MAE or Huber loss may improve typical accuracy, while peak-weighted training
may be more appropriate if missing rare high-current events is the main risk.

---

## Slide 19 — Conclusion

### Final conclusion

> A compact LSTM can forecast the complete short-term state of the simulated
> low-voltage feeder with strong voltage accuracy and useful current accuracy,
> particularly at short horizons.

### Key findings

- Overall voltage RMSE: **0.00405 Vpu**
- Overall current RMSE: **4.91 A**
- Best performance occurs at one-minute lead time.
- Error increases predictably toward minute 15.
- The LSTM improves RMSE and extreme error over persistence.
- A fitted trend remains a highly competitive baseline.
- More independent operating days are required for final validation.

### Speaker notes

The project demonstrates a complete temporal forecasting pipeline, from
OpenDSS-generated data through preprocessing, dimensionality reduction, LSTM
training, and held-out evaluation.

The voltage result is promising, with average squared error corresponding to
roughly 0.4% of nominal voltage. Current forecasts are useful on average but
remain vulnerable to abrupt events.

The scientifically responsible conclusion is that the approach is promising
for very-short-term forecasting, but one simulated day is not sufficient to
establish general reliability. The next stage should focus on multi-day data,
untouched scenario-level testing, and comparison with graph-aware models.

---

## Slide 20 — Questions

### Thank you

**Questions and discussion**

Suggested closing statement:

> The strongest outcome of this work is not only the numerical accuracy, but a
> reproducible framework for testing temporal and graph-based forecasting
> methods on the same feeder data.

---

# Appendix — Likely questions and concise answers

## Why use minimum phase voltage?

It is a conservative voltage-quality target. A low value on one phase should
not be hidden by averaging it with healthier phases.

## Why use PCA?

The 906 voltage and 905 current series are strongly correlated. PCA captures
shared feeder-wide behavior, reduces model size, and discourages independent,
physically inconsistent output changes.

## Why does the model use previous voltage and current?

They provide the most recent observed electrical state. Without them, the
model must reconstruct the entire feeder state from loads alone. Their use
assumes real-time measurements or state estimates are available in deployment.

## Why not use random train/test splitting?

Adjacent minutes are highly correlated. Random splitting would place near-
duplicate temporal conditions in training and test data, producing optimistic
results through leakage.

## Why did training stop before 100 epochs?

One hundred was the maximum. Validation stopped improving after epoch 6, and
the 20-epoch patience rule stopped training at epoch 26. The epoch-6 checkpoint
was restored.

## Why is the current maximum error large?

The model missed an abrupt event on line 161. Current can change sharply after
a load step, and the model has no future load information. Rare high-current
events are also underrepresented in one day of training data.

## Is the LSTM better than simple baselines?

It improves RMSE and maximum error over persistence. The fitted trend baseline
is slightly better on average, so the LSTM has not yet demonstrated consistent
superiority over every statistical baseline.

## Is this ready for deployment?

No. It is a proof of concept based on one simulated day. Deployment would
require multi-day and multi-scenario validation, an untouched final test set,
measurement-quality analysis, uncertainty estimates, and operational safety
criteria.

