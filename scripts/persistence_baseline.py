"""
persistence_baseline.py

Trivial "current stays the same as the last observed minute" baseline for
the N-step-ahead current forecasting task, computed in the SAME per-line
normalized space and same masked_mse convention TemporalVoltageHeteroGNN
uses -- so its I-MSE is directly comparable to the model's.

Run this after the multi-task-interference diagnostic ruled out shared-
backbone competition as the cause of the ~0.76 train / ~2.15 val I-MSE
plateau. This tells us which of two very different situations we're in:

  - If persistence beats the model (lower MSE) -> the model isn't adding
    value yet, worth digging further into the architecture/training setup.
  - If persistence is similar or worse -> ~0.76-2.15 is close to the real
    ceiling for this signal at this horizon with one day of data, and the
    more useful next step is more/varied data (multiple simulated days),
    not more architecture tweaking.
"""

import os

from hetero_gnn_dataset import BASE_DIR
from temporal_dataset import FeederHeteroSnapshotDatasetWithCurrent, TemporalFeederWindowDataset
from temporal_model import masked_mse

WINDOW = 30
HORIZON = 15
TRAIN_FRAC, VAL_FRAC = 0.7, 0.15


def persistence_mse(window_ds: TemporalFeederWindowDataset) -> float:
    """For every valid window, predict all N future steps as equal to the
    last observed (already per-line-normalized) value -- the simplest
    possible forecast."""
    total, n = 0.0, 0
    for t in window_ds.valid_t:
        last_val = window_ds._line_all[t]                                    # (num_line,)
        target = window_ds._line_all[t + 1 : t + 1 + window_ds.horizon]         # (N, num_line)
        pred = last_val.unsqueeze(0).expand(window_ds.horizon, -1)               # repeat across horizon
        mask = window_ds._line_mask_all.unsqueeze(0).expand(window_ds.horizon, -1)

        loss = masked_mse(pred, target, mask)
        total += loss.item()
        n += 1
    return total / n


def main():
    ds = FeederHeteroSnapshotDatasetWithCurrent(
        neo4j_uri=os.environ.get("NEO4J_URI"),
        neo4j_user=os.environ.get("NEO4J_USERNAME"),
        neo4j_password=os.environ.get("NEO4J_PASSWORD"),
        load_profiles_path=BASE_DIR / "data" / "Computed" / "load_profiles.csv",
        voltage_labels_path=BASE_DIR / "data" / "Computed" / "bus_voltages.csv",
        line_currents_path=BASE_DIR / "data" / "Computed" / "line_currents.csv",
    )

    n = ds.n_timesteps
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    train_range = (0, n_train)
    val_range = (n_train, n_train + n_val)
    test_range = (n_train + n_val, n)   # same convention as train_temporal.py

    ds.fit_normalization(range(*train_range))

    train_ds = TemporalFeederWindowDataset(ds, window=WINDOW, horizon=HORIZON, split_range=train_range)
    val_ds = TemporalFeederWindowDataset(ds, window=WINDOW, horizon=HORIZON, split_range=val_range)
    test_ds = TemporalFeederWindowDataset(ds, window=WINDOW, horizon=HORIZON, split_range=test_range)

    print(f"Persistence baseline train I-MSE: {persistence_mse(train_ds):.6f}")
    print(f"Persistence baseline val I-MSE:   {persistence_mse(val_ds):.6f}")
    print(f"Persistence baseline test I-MSE:  {persistence_mse(test_ds):.6f}")


if __name__ == "__main__":
    main()