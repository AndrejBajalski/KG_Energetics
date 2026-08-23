# LSTM data results

`scripts/train_lstm.py` writes all CSV-only LSTM training artifacts here. The
model uses the previous 30 minutes of load, bus-voltage, and line-current CSV
observations to forecast the next 15 minutes. It does not use Neo4j.

The current architecture uses training-only PCA bases (16 voltage components
and 64 current components), a two-layer LSTM, and residual forecasts around a
training-fitted trend baseline. This keeps feeder-wide predictions spatially
coherent and reduces overfitting compared with independent output heads.

Expected outputs after a completed run:

- `config.json`: model, split, and training configuration.
- `training_history.csv`: normalized train and validation losses by epoch.
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

All normalization, PCA, and trend statistics are fitted on training data only.
