"""
train_temporal.py

Drop this alongside pyg_dataset.py in scripts/. Wires
FeederHeteroSnapshotDatasetWithCurrent + TemporalFeederWindowDataset +
TemporalVoltageHeteroGNN together, using the same 70/15/15 chronological
split convention as your existing time_split().
"""

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pyg_dataset import BASE_DIR
from temporal_dataset import (
    FeederHeteroSnapshotDatasetWithCurrent,
    TemporalFeederWindowDataset,
    collate_windows,
)
from temporal_model import TemporalVoltageHeteroGNN, masked_mse

WINDOW = 30            # minutes of history fed to the LSTM
HORIZON = 15            # minutes ahead to forecast
HIDDEN = 64
BATCH_SIZE = 8
LAMBDA_CURRENT = 0.3     # weight on current loss relative to voltage loss
EPOCHS = 20
TRAIN_FRAC, VAL_FRAC = 0.7, 0.15


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
    # test_range = (n_train + n_val, n)  -- held out until final evaluation

    # fit_normalization (bus/load/linecode/transformer/source features +
    # current label mean/std) on TRAIN indices only, same leakage guard as
    # your existing pipeline.
    ds.fit_normalization(range(*train_range))


    train_ds = TemporalFeederWindowDataset(ds, window=WINDOW, horizon=HORIZON, split_range=train_range)
    val_ds = TemporalFeederWindowDataset(ds, window=WINDOW, horizon=HORIZON, split_range=val_range)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_windows)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_windows)

    sample = ds.get(0)
    in_dims = {ntype: sample[ntype].x.size(1) for ntype in sample.node_types}
    relations = list(sample.edge_index_dict.keys())
    edge_attr_dim = sample[("bus", "line_segment", "bus")].edge_attr.size(1)

    model = TemporalVoltageHeteroGNN(
        in_dims=in_dims,
        relations=relations,
        edge_attr_dim=edge_attr_dim,
        hidden=HIDDEN,
        horizon=HORIZON,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        train_v_loss, train_c_loss, n_batches = 0.0, 0.0, 0
        for batch in train_loader:
            opt.zero_grad()
            voltage_pred, current_pred = model(batch["graph_batch"], batch["B"], batch["W"])

            v_loss = masked_mse(voltage_pred, batch["bus_targets"], batch["bus_mask"])
            c_loss = masked_mse(current_pred, batch["line_targets"], batch["line_mask"])
            loss = v_loss + LAMBDA_CURRENT * c_loss
            loss.backward()
            opt.step()

            train_v_loss += v_loss.item()
            train_c_loss += c_loss.item()
            n_batches += 1

        model.eval()
        val_v_loss, val_c_loss, n_val_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                voltage_pred, current_pred = model(batch["graph_batch"], batch["B"], batch["W"])
                val_v_loss += masked_mse(voltage_pred, batch["bus_targets"], batch["bus_mask"]).item()
                val_c_loss += masked_mse(current_pred, batch["line_targets"], batch["line_mask"]).item()
                n_val_batches += 1

        print(
            f"epoch {epoch:02d} | "
            f"train V-MSE {train_v_loss / n_batches:.6f} I-MSE {train_c_loss / n_batches:.6f} | "
            f"val V-MSE {val_v_loss / n_val_batches:.6f} I-MSE {val_c_loss / n_val_batches:.6f}"
        )


if __name__ == "__main__":
    main()