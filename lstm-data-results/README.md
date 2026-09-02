# LSTM data results

`scripts/train_lstm.py` writes all CSV-only LSTM training artifacts here. The
model uses the previous 30 minutes of load, bus-voltage, and line-current CSV
observations to forecast the next 15 minutes. It does not use Neo4j.

The current architecture uses training-only PCA bases (16 voltage components
and 64 current components), a one-layer LSTM with a 128-dimensional hidden
state, and residual forecasts around a training-fitted trend baseline. This
keeps feeder-wide predictions spatially coherent and reduces overfitting
compared with independent output heads.

See [`docs/CSV_ONLY_LSTM_MODEL.md`](../docs/CSV_ONLY_LSTM_MODEL.md) for the
complete data, preprocessing, architecture, training, and evaluation guide.

Expected outputs after a completed run:

- `config.json`: model, split, and training configuration.
- `training_history.csv`: train and validation losses in the configured model
  data space by epoch.
- `best_model.pt`: best validation checkpoint, scalers, and column names.
- `test_metrics.json`: held-out model, persistence-baseline, and trend-baseline
  metrics in Vpu/A.
- `test_metrics_by_horizon.csv`: metrics for forecast minutes 1 through 15.
- `test_predictions.npz`: full predictions and targets for every test window/horizon.
- `test_voltage_predictions_h1.csv`: human-readable one-minute-ahead bus predictions.
- `test_current_predictions_h1.csv`: human-readable one-minute-ahead line predictions.
- `run_summary.json`: best epoch and final test summary.

Run from the project root with:

```powershell
python scripts/train_lstm.py
```

Set `NORMALIZE_DATA` in `scripts/train_lstm.py` to choose per-feature training
standardization (`True`) or original CSV units (`False`). When enabled, all
normalization, PCA, and trend statistics are fitted on training data only.
