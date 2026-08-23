# CSV-only LSTM forecasting results

## Executive summary

The revised CSV-only LSTM is a clear improvement over the original load-only
LSTM. On the held-out final portion of the simulated day, it reduced voltage
RMSE by **21.3%** and current RMSE by **12.1%** relative to the earlier model.

The final model forecasts the next 15 minutes from the previous 30 minutes of:

- Load active power in kW.
- Minimum phase voltage at each bus in per-unit.
- Maximum conductor current at each line in amperes.

Its overall test errors are:

| Target | MAE | RMSE | Maximum absolute error |
|---|---:|---:|---:|
| Bus voltage | 0.002989 Vpu | 0.004050 Vpu | 0.018693 Vpu |
| Line current | 2.143 A | 4.914 A | 61.161 A |

A voltage RMSE of 0.00405 Vpu is approximately **0.405% of nominal voltage**.
The model is strongest at one-minute lead time and becomes progressively less
accurate toward minute 15, as expected for a forecast without future load
inputs.

## Evaluation setup

| Item | Value |
|---|---:|
| Available data | One simulated day, 1,440 one-minute timesteps |
| Historical window | 30 minutes |
| Forecast horizon | 15 minutes |
| Load inputs | 55 |
| Voltage targets | 906 buses |
| Current targets | 905 lines |
| Training windows | 963 |
| Validation windows | 202 |
| Test windows | 203 |
| Best validation epoch | 6 |
| Early stopping epoch | 26 |
| Trainable parameters | 721,040 |

The split is chronological: the first 70% of the day is training data, the
next 15% is validation data, and the final 15% is the reported test period.
All normalization, trend coefficients, and PCA bases are fitted from training
data only.

Because windows overlap, the 203 test windows are not 203 independent physical
experiments. They produce 3,045 origin/horizon combinations, covering
2,758,770 voltage predictions and 2,755,725 current predictions.

## Improvement over the previous LSTM

| Test metric | Previous load-only LSTM | Revised LSTM | Relative improvement |
|---|---:|---:|---:|
| Voltage MAE | 0.004012 Vpu | 0.002989 Vpu | **25.5%** |
| Voltage RMSE | 0.005149 Vpu | 0.004050 Vpu | **21.3%** |
| Current MAE | 2.930 A | 2.143 A | **26.9%** |
| Current RMSE | 5.588 A | 4.914 A | **12.1%** |

The main reasons for the improvement are:

1. The model is anchored by recent observed voltage and current instead of
   attempting to infer the entire electrical state from loads alone.
2. It predicts changes relative to a training-fitted short-term trend rather
   than absolute values from scratch.
3. Training-only PCA reduces the 906 voltage outputs to 16 spatial components
   and the 905 current outputs to 64 components. The components retain about
   99.99% of standardized voltage variation and 99.95% of standardized current
   variation in the training interval.
4. The number of trainable parameters fell from 3.81 million to 721 thousand,
   reducing the opportunity to memorize one day of data.
5. The learning rate was reduced to `1e-4`, with scheduling and early stopping.

## Comparison with simple baselines

| Method | Voltage MAE | Voltage RMSE | Current MAE | Current RMSE |
|---|---:|---:|---:|---:|
| Persistence: repeat the last value | **0.002967** | 0.004396 | **1.827 A** | 5.113 A |
| Training-fitted trend | **0.002888** | **0.004034** | 1.969 A | **4.700 A** |
| Revised LSTM | 0.002989 | 0.004050 | 2.143 A | 4.914 A |

Compared with persistence, the LSTM improves:

- Voltage RMSE by **7.9%**.
- Current RMSE by **3.9%**.
- Maximum voltage error by **21.4%**.
- Maximum current error by **19.7%**.

However, persistence has slightly lower voltage MAE and substantially lower
current MAE. This combination means the LSTM makes more small errors but
reduces some of the larger errors, which lowers RMSE and the observed maximum.
That behavior is consistent with training primarily against mean squared error.

The fitted trend baseline remains slightly better than the LSTM on average:
its voltage RMSE is 0.4% lower and current RMSE is 4.6% lower. The LSTM has
smaller maximum errors in this test set: 6.4% lower for voltage and 2.5% lower
for current. Therefore:

- Use the fitted trend baseline if average MAE/RMSE is the only criterion.
- Prefer the LSTM only if its reduction of larger errors remains consistent on
  additional days and scenarios.

This is a valuable result: a learned model should not be judged without a
strong persistence/trend comparison.

## Accuracy by forecast horizon

| Lead time | Voltage MAE | Voltage RMSE | Current MAE | Current RMSE |
|---:|---:|---:|---:|---:|
| 1 minute | 0.001566 Vpu | 0.002320 Vpu | 1.105 A | 3.030 A |
| 3 minutes | 0.002623 Vpu | 0.003584 Vpu | 1.853 A | 4.484 A |
| 5 minutes | 0.002860 Vpu | 0.003829 Vpu | 2.018 A | 4.722 A |
| 10 minutes | 0.003286 Vpu | 0.004360 Vpu | 2.356 A | 5.180 A |
| 15 minutes | 0.003594 Vpu | 0.004691 Vpu | 2.621 A | 5.618 A |

From minute 1 to minute 15:

- Voltage RMSE increases by **102%**.
- Current RMSE increases by **85%**.

The largest deterioration happens in the first few forecast steps; after
approximately five minutes, error increases more gradually. The model is most
credible as a very-short-term forecaster. Fifteen-minute results are still
usable as an aggregate estimate, but individual line events are less reliable.

## Where the model struggles

### Voltage

The highest per-bus RMSEs occur mainly around buses `639`, `633`, `626`,
`616`, and `813`. Their RMSEs are approximately 0.0051–0.0053 Vpu, compared
with the overall 0.00405 Vpu.

The concentration of difficult buses in similar ID ranges may indicate a
shared feeder branch or operating condition. This is an inference from the
identifiers; the topology should be checked in Neo4j before attributing a
physical cause.

The worst individual voltage prediction occurred at timestep 1367 for bus
`178`, at forecast minute 14:

```text
Predicted: 1.03964 Vpu
Actual:    1.02095 Vpu
Error:     0.01869 Vpu
```

### Current

The highest per-line RMSE occurs on `line27` and `line31`, at approximately
15.53 A. A second group—including `line64`, `line71`, `line78`, `line83`,
`line91`, `line94`, `line97`, and `line100`—has approximately 15.21 A RMSE.
These lines also carry much higher current than a typical lateral, so aggregate
current metrics are strongly influenced by them.

The worst individual current prediction occurred at timestep 1366 on
`line161`, at forecast minute 11:

```text
Predicted:  3.76 A
Actual:    64.92 A
Error:     61.16 A
```

This looks like an abrupt current event that was not indicated strongly enough
by the preceding observations. Without future load forecasts or event inputs,
such jumps are inherently difficult for the LSTM to anticipate.

## Training behavior

The untrained residual heads begin at the fitted trend baseline, whose
normalized validation loss is 1.1548. Training improved the best validation
loss to 1.1312 at epoch 6. After that, training loss continued to decrease but
validation stopped improving. Early stopping ended the run at epoch 26 and
restored epoch 6.

This is the expected sign of limited-data overfitting: there are millions of
individual target values, but only one daily trajectory and 963 highly
overlapping training windows. PCA and residual forecasting reduced the problem
substantially, but they cannot create new operating scenarios.

## Interpretation and limitations

1. **The revised model is meaningfully better than the original LSTM.** The
   improvement is present in all four main MAE/RMSE metrics.
2. **A simple trend is extremely competitive.** On this dataset it remains the
   best average predictor, so future model comparisons must include it.
3. **Short-horizon forecasts are substantially more trustworthy.** One-minute
   predictions are much stronger than 15-minute predictions.
4. **Current prediction is uneven across lines.** A small group of high-current
   lines and rare spikes dominate squared error.
5. **The model now requires recent voltage and current observations.** It is no
   longer a load-only surrogate. Deployment therefore assumes those values are
   measured or otherwise available in real time.
6. **The data represents only one OpenDSS-simulated day.** Performance cannot
   yet be generalized to another day, season, fault, topology, or feeder.
7. **The test interval was inspected during iterative model development.** It
   should now be treated as a development holdout, not a pristine final test.
   A new simulated day or untouched scenario is required for an unbiased final
   claim.
8. **Voltage targets are minimum phase values per bus and current targets are
   maximum magnitudes per line.** Phase-specific behavior and voltage
   unbalance are not evaluated.

## Recommended next steps

1. Generate multiple simulated days with different load profiles and split by
   entire day rather than by adjacent minutes.
2. Reserve at least one day or scenario that is never consulted during model
   development.
3. Keep persistence and fitted-trend results beside every learned-model result.
4. Add future load forecasts as exogenous inputs if they would be available in
   deployment; abrupt current changes are difficult to predict without them.
5. Report per-bus and per-line metrics, especially for high-current trunk lines,
   instead of relying only on feeder-wide averages.
6. Investigate the `line161` event and the difficult 600/800-series buses in
   OpenDSS and Neo4j to determine whether they correspond to load steps or
   particular feeder branches.
7. Evaluate separate objectives for average accuracy and extreme-event error.
   A Huber/MAE objective may improve typical errors, while weighted peak-event
   training may improve rare current spikes.

## Result artifacts

- [Configuration](./config.json)
- [Training history](./training_history.csv)
- [Overall test metrics](./test_metrics.json)
- [Metrics by forecast horizon](./test_metrics_by_horizon.csv)
- [Full test predictions](./test_predictions.npz)
- [One-minute voltage predictions](./test_voltage_predictions_h1.csv)
- [One-minute current predictions](./test_current_predictions_h1.csv)
- [Run summary](./run_summary.json)
- [Best model checkpoint](./best_model.pt)

