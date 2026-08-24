"""
train_temporal.py

Drop this alongside hetero_gnn_dataset.py in scripts/. Wires
FeederHeteroSnapshotDatasetWithCurrent + TemporalFeederWindowDataset +
TemporalVoltageHeteroGNN together, using the same 70/15/15 chronological
split convention as your existing time_split().

Adds two things over the previous version:
  - A held-out TEST split, evaluated exactly once after training finishes
    (never touched during the epoch loop, so it stays an honest estimate).
  - Results written to a timestamped log file under results/, not just
    printed -- each line is written and flushed immediately so a crash or
    interrupt mid-run still leaves you a usable partial log.
"""

import os
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hetero_gnn_dataset import BASE_DIR
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = BASE_DIR / "results"


def move_batch_to_device(batch, device):
    """collate_windows returns a dict mixing a PyG Batch (supports .to())
    with plain tensors -- move each appropriately."""
    return {
        "graph_batch": batch["graph_batch"].to(device),
        "B": batch["B"],
        "W": batch["W"],
        "bus_targets": batch["bus_targets"].to(device),
        "bus_mask": batch["bus_mask"].to(device),
        "line_targets": batch["line_targets"].to(device),
        "line_mask": batch["line_mask"].to(device),
    }


def train_one_epoch(model, loader, opt):
    model.train()
    v_total, c_total, n_batches = 0.0, 0.0, 0
    for batch in loader:
        batch = move_batch_to_device(batch, DEVICE)
        opt.zero_grad()
        voltage_pred, current_pred = model(batch["graph_batch"], batch["B"], batch["W"])

        v_loss = masked_mse(voltage_pred, batch["bus_targets"], batch["bus_mask"])
        c_loss = masked_mse(current_pred, batch["line_targets"], batch["line_mask"])
        loss = v_loss + LAMBDA_CURRENT * c_loss
        loss.backward()
        opt.step()

        v_total += v_loss.item()
        c_total += c_loss.item()
        n_batches += 1

    return v_total / n_batches, c_total / n_batches


def evaluate(model, loader):
    """Shared by val (called every epoch) and test (called once at the
    end) -- no gradient, frozen weights, one pass through."""
    model.eval()
    v_total, c_total, n_batches = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, DEVICE)
            voltage_pred, current_pred = model(batch["graph_batch"], batch["B"], batch["W"])
            v_total += masked_mse(voltage_pred, batch["bus_targets"], batch["bus_mask"]).item()
            c_total += masked_mse(current_pred, batch["line_targets"], batch["line_mask"]).item()
            n_batches += 1

    return v_total / n_batches, c_total / n_batches


def log(msg: str, log_file) -> None:
    """Write to both the console and the log file, flushing immediately
    so a crash mid-run doesn't lose what's already happened."""
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = RESULTS_DIR / f"train_run_{run_id}.txt"

    with open(log_path, "w") as log_file:
        log(f"Run started {run_id} | device={DEVICE} | "
            f"window={WINDOW} horizon={HORIZON} hidden={HIDDEN} "
            f"batch_size={BATCH_SIZE} lambda_current={LAMBDA_CURRENT} epochs={EPOCHS}",
            log_file)

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
        test_range = (n_train + n_val, n)   # held out until the very end

        # fit_normalization (bus/load/linecode/transformer/source features +
        # current label mean/std) on TRAIN indices only, same leakage guard
        # as your existing pipeline.
        ds.fit_normalization(range(*train_range))

        train_ds = TemporalFeederWindowDataset(ds, window=WINDOW, horizon=HORIZON, split_range=train_range)
        val_ds = TemporalFeederWindowDataset(ds, window=WINDOW, horizon=HORIZON, split_range=val_range)
        test_ds = TemporalFeederWindowDataset(ds, window=WINDOW, horizon=HORIZON, split_range=test_range)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_windows)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_windows)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_windows)

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
        ).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(EPOCHS):
            train_v, train_c = train_one_epoch(model, train_loader, opt)
            val_v, val_c = evaluate(model, val_loader)
            log(
                f"epoch {epoch:02d} | "
                f"train V-MSE {train_v:.6f} I-MSE {train_c:.6f} | "
                f"val V-MSE {val_v:.6f} I-MSE {val_c:.6f}",
                log_file,
            )

        # TEST is evaluated exactly once, here, after all training and all
        # epoch-by-epoch decisions are already finished. Don't loop this,
        # don't tune anything based on it, don't run it again with
        # different hyperparameters -- that's what keeps it honest.
        test_v, test_c = evaluate(model, test_loader)
        log(
            f"\nFINAL TEST (evaluated once, held out through all of training) | "
            f"V-MSE {test_v:.6f} I-MSE {test_c:.6f}",
            log_file,
        )

    print(f"\nFull log written to {log_path}")


if __name__ == "__main__":
    main()