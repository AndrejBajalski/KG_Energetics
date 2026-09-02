"""
train_temporal.py

Drop this alongside pyg_dataset.py in scripts/. Wires
FeederHeteroSnapshotDatasetWithCurrent + TemporalFeederWindowDataset +
TemporalVoltageHeteroGNN together, using the same 70/15/15 chronological
split convention as your existing time_split().

Adds over the previous version:
  - A held-out TEST split, evaluated exactly once after training finishes
    (never touched during the epoch loop, so it stays an honest estimate).
  - Results written to a timestamped log file under results/, not just
    printed -- each line is written and flushed immediately so a crash or
    interrupt mid-run still leaves you a usable partial log.
  - R^2 alongside MSE. R^2 is NOT averaged per-batch (that's a biased way
    to compute it) -- every batch's masked predictions/targets are
    collected (detached, on CPU) and R^2 is computed once at the end of
    the epoch over the full pooled set, same convention MSE now uses too.

Reading R^2: 1.0 is a perfect fit, 0.0 means "no better than always
predicting the pooled mean of the targets", negative means worse than that.
Note this is a DIFFERENT reference point than the persistence baseline from
persistence_baseline.py (which compares against "predict the last observed
value", not the mean) -- don't conflate "R^2 > 0" with "beats persistence".
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
EARLY_STOP_PATIENCE = 8   # stop if val (V-MSE + lambda*I-MSE) hasn't improved in this many epochs
TRAIN_FRAC, VAL_FRAC = 0.7, 0.15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = BASE_DIR / "temporal-data-results"


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


def masked_flatten(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """Same masking convention as masked_mse (boolean-index, not multiply --
    targets can contain NaN in unmasked positions, and NaN*0 is still NaN).
    Detaches and moves to CPU immediately so accumulating these across a
    whole epoch doesn't retain the autograd graph or hold GPU memory."""
    mask = mask.bool()
    return pred[mask].detach().cpu(), target[mask].detach().cpu()


def mse_and_r2(preds_list, targets_list):
    """Pools every batch's masked (pred, target) pairs from the epoch and
    computes MSE and R^2 once over the full set -- R^2 in particular isn't
    valid to average per-batch."""
    if not preds_list:
        return float("nan"), float("nan")

    preds = torch.cat(preds_list)
    targets = torch.cat(targets_list)
    if preds.numel() == 0:
        return float("nan"), float("nan")

    mse = torch.mean((preds - targets) ** 2).item()

    ss_res = torch.sum((preds - targets) ** 2)
    ss_tot = torch.sum((targets - targets.mean()) ** 2)
    r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else float("nan")

    return mse, r2


def train_one_epoch(model, loader, opt):
    model.train()
    v_preds, v_targets, c_preds, c_targets = [], [], [], []

    for batch in loader:
        batch = move_batch_to_device(batch, DEVICE)
        opt.zero_grad()
        voltage_pred, current_pred = model(batch["graph_batch"], batch["B"], batch["W"])

        v_loss = masked_mse(voltage_pred, batch["bus_targets"], batch["bus_mask"])
        c_loss = masked_mse(current_pred, batch["line_targets"], batch["line_mask"])
        loss = v_loss + LAMBDA_CURRENT * c_loss
        loss.backward()
        opt.step()

        vp, vt = masked_flatten(voltage_pred, batch["bus_targets"], batch["bus_mask"])
        cp, ct = masked_flatten(current_pred, batch["line_targets"], batch["line_mask"])
        v_preds.append(vp); v_targets.append(vt)
        c_preds.append(cp); c_targets.append(ct)

    v_mse, v_r2 = mse_and_r2(v_preds, v_targets)
    c_mse, c_r2 = mse_and_r2(c_preds, c_targets)
    return v_mse, v_r2, c_mse, c_r2


def evaluate(model, loader):
    """Shared by val (called every epoch) and test (called once at the
    end) -- no gradient, frozen weights, one pass through."""
    model.eval()
    v_preds, v_targets, c_preds, c_targets = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, DEVICE)
            voltage_pred, current_pred = model(batch["graph_batch"], batch["B"], batch["W"])

            vp, vt = masked_flatten(voltage_pred, batch["bus_targets"], batch["bus_mask"])
            cp, ct = masked_flatten(current_pred, batch["line_targets"], batch["line_mask"])
            v_preds.append(vp); v_targets.append(vt)
            c_preds.append(cp); c_targets.append(ct)

    v_mse, v_r2 = mse_and_r2(v_preds, v_targets)
    c_mse, c_r2 = mse_and_r2(c_preds, c_targets)
    return v_mse, v_r2, c_mse, c_r2


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

        checkpoint_path = RESULTS_DIR / f"best_model_{run_id}.pt"
        best_val_metric = float("inf")
        best_epoch = -1
        epochs_since_improvement = 0

        for epoch in range(EPOCHS):
            train_v_mse, train_v_r2, train_c_mse, train_c_r2 = train_one_epoch(model, train_loader, opt)
            val_v_mse, val_v_r2, val_c_mse, val_c_r2 = evaluate(model, val_loader)

            # Same combined objective the model is actually trained on --
            # this is what decides "best", not either metric in isolation.
            val_metric = val_v_mse + LAMBDA_CURRENT * val_c_mse
            improved = val_metric < best_val_metric
            if improved:
                best_val_metric = val_metric
                best_epoch = epoch
                epochs_since_improvement = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                epochs_since_improvement += 1

            log(
                f"epoch {epoch:02d} | "
                f"train V-MSE {train_v_mse:.6f} V-R2 {train_v_r2:.4f} "
                f"I-MSE {train_c_mse:.6f} I-R2 {train_c_r2:.4f} | "
                f"val V-MSE {val_v_mse:.6f} V-R2 {val_v_r2:.4f} "
                f"I-MSE {val_c_mse:.6f} I-R2 {val_c_r2:.4f}"
                f"{'  <- best so far, checkpointed' if improved else ''}",
                log_file,
            )

            if epochs_since_improvement >= EARLY_STOP_PATIENCE:
                log(f"No val improvement for {EARLY_STOP_PATIENCE} epochs, stopping early "
                    f"at epoch {epoch} (best was epoch {best_epoch}).", log_file)
                break

        log(f"\nBest val checkpoint: epoch {best_epoch} "
            f"(val V-MSE + {LAMBDA_CURRENT}*I-MSE = {best_val_metric:.6f})", log_file)

        # Reload the BEST checkpoint before the final test pass -- not the
        # final epoch's weights, which epoch 19 above showed can be
        # meaningfully worse than an earlier epoch once the current head
        # starts overfitting train's single-day quirks.
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

        # TEST is evaluated exactly once, here, using the best checkpoint,
        # after all training and all epoch-by-epoch decisions are already
        # finished. Don't loop this, don't tune anything based on it, don't
        # run it again with different hyperparameters -- that's what keeps
        # it honest.
        test_v_mse, test_v_r2, test_c_mse, test_c_r2 = evaluate(model, test_loader)
        log(
            f"\nFINAL TEST (best-val checkpoint from epoch {best_epoch}, "
            f"held out through all of training) | "
            f"V-MSE {test_v_mse:.6f} V-R2 {test_v_r2:.4f} "
            f"I-MSE {test_c_mse:.6f} I-R2 {test_c_r2:.4f}",
            log_file,
        )

    print(f"\nFull log written to {log_path}")


if __name__ == "__main__":
    main()