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
  - Early stopping on the combined val objective, with the best-val
    checkpoint saved and reloaded before the final test pass.
  - R^2 alongside MSE. R^2 is NOT averaged per-batch (that's a biased way
    to compute it) -- sufficient statistics are accumulated across every
    batch and R^2 is computed once at the end over the full pooled set,
    same convention MSE uses too.

Reporting is in REAL PHYSICAL UNITS, matching hetero_gnn_model.py:
  - Voltage is already reported in Vpu (V-MSE in Vpu^2) -- the window
    dataset builds bus targets straight from the raw voltage CSV, so the
    voltage head trains and reports in real per-unit volts already.
  - Current is trained in per-line z-scored space (that's the right loss
    scale -- see temporal_dataset.fit_normalization) but the *reported*
    current MSE is un-normalized back to Amps (I-MSE in A^2), exactly like
    hetero_gnn_model.py un-normalizes with per-line (i_mean, i_std) before
    logging. The normalized current MSE (the training-loss scale) is kept
    as a secondary reference, and is what the early-stopping / checkpoint
    objective uses so model selection still matches the training loss.
  - R^2 is per-entity: each bus / each line is scored against its OWN mean
    over the split, i.e. sklearn's variance_weighted multioutput R^2,
    identical to hetero_gnn_model.py's _r2 (one pooled SS_res/SS_tot ratio
    over all entities). NOT a single global pooled mean -- pooling across
    entities with different means inflates R^2 badly once the targets are
    in real Amps, because between-line scale differences get counted as
    "explained" variance.

Reading R^2: 1.0 is a perfect fit, 0.0 means "no better than always
predicting that entity's own mean", negative means worse than that. Note
this is a DIFFERENT reference point than the persistence baseline from
persistence_baseline.py (which compares against "predict the last observed
value", not the mean) -- don't conflate "R^2 > 0" with "beats persistence".

Reporting current R^2 in real Amps rather than the normalized space is what
makes it comparable to hetero_gnn_model.py: because the aggregation is
variance-weighted, each line is weighted by its real-Amps variance, so
high-current trunk lines dominate the score in BOTH models. (A
normalized-space R^2 weights every line roughly equally instead, which is
why it reads differently -- it is not the same number.)
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


# ----------------------------------------------------------------------------
# Per-entity MSE / R^2 accumulation, matching hetero_gnn_model.py's
# _accumulate/_r2. Replaces the earlier masked_flatten + mse_and_r2 pair:
# same "pool the whole split, never average R^2 per batch" property, but
# centered per entity instead of on one global mean, and it accumulates
# scalars instead of retaining every masked prediction of the epoch.
#
# Every (window, horizon-step) prediction is flattened into rows and, per
# entity (bus or line), we accumulate sum(y), sum(y^2), sum(residual^2) and
# a row count, then combine:
#     SS_tot = sum_e[ sum(y^2) - sum(y)^2 / rows ]   (per-entity centered)
#     R^2    = 1 - SS_res / SS_tot
# The label mask is static per entity (an entity is labeled or not, the same
# for every timestep), so mask[0] gives the per-entity valid set, exactly the
# assumption hetero_gnn_model.py's _r2 makes.
# ----------------------------------------------------------------------------
def _new_acc():
    return {"sy": None, "sy2": 0.0, "sres": 0.0, "rows": 0, "valid": None, "nvalid": 0}


def _accumulate(a, y, p, m):
    """y, p: (rows, n_entity) targets/preds. m: (rows, n_entity) bool mask.
    Masked-out entries are zeroed on both sides so they contribute nothing to
    any of the sums (NaN-safe: targets carry NaN where unlabeled)."""
    m = m.bool()
    zeros = torch.zeros_like(y)
    y0 = torch.nan_to_num(torch.where(m, y, zeros), nan=0.0)
    p0 = torch.nan_to_num(torch.where(m, p, zeros), nan=0.0)
    res = torch.where(m, (p0 - y0) ** 2, zeros)
    col_sum = y0.sum(dim=0)                              # (n_entity,)
    a["sy"] = col_sum if a["sy"] is None else a["sy"] + col_sum
    a["sy2"] += (y0 ** 2).sum().item()
    a["sres"] += res.sum().item()
    a["rows"] += y.size(0)
    a["valid"] = m[0]
    a["nvalid"] = int(m[0].sum().item())


def _acc_mse(a):
    """Mean masked squared error over all valid (entity, row) observations."""
    denom = a["nvalid"] * a["rows"]
    return a["sres"] / denom if denom > 0 else float("nan")


def _acc_r2(a):
    if a["sy"] is None or a["rows"] == 0:
        return float("nan")
    sy_e = a["sy"][a["valid"]]
    ss_tot = a["sy2"] - (sy_e * sy_e / a["rows"]).sum().item()
    return 1.0 - a["sres"] / ss_tot if ss_tot > 0 else float("nan")


def _flatten(t):
    """(B, horizon, n_entity) -> (B*horizon, n_entity)."""
    return t.reshape(-1, t.size(-1))


def _accumulate_batch(acc_v, acc_i, acc_i_norm, batch, voltage_pred, current_pred, i_mean, i_std):
    """Shared metric accumulation for both the train and eval loops, so the
    two report the exact same quantities in the exact same units."""
    v_pred = _flatten(voltage_pred)                             # already real Vpu
    v_tgt = _flatten(batch["bus_targets"])
    v_mask = _flatten(batch["bus_mask"])

    i_pred_norm = _flatten(current_pred)                        # normalized (loss space)
    i_tgt_norm = _flatten(batch["line_targets"])
    i_mask = _flatten(batch["line_mask"])
    i_pred_real = i_pred_norm * i_std + i_mean                  # -> Amps
    i_tgt_real = i_tgt_norm * i_std + i_mean

    _accumulate(acc_v, v_tgt, v_pred, v_mask)
    _accumulate(acc_i, i_tgt_real, i_pred_real, i_mask)
    _accumulate(acc_i_norm, i_tgt_norm, i_pred_norm, i_mask)


def _metrics(acc_v, acc_i, acc_i_norm):
    return {
        "v_mse": _acc_mse(acc_v),          # Vpu^2
        "v_r2": _acc_r2(acc_v),            # per-bus reference
        "i_mse": _acc_mse(acc_i),          # Amps^2
        "i_r2": _acc_r2(acc_i),            # per-line, real Amps, variance-weighted
        "i_mse_norm": _acc_mse(acc_i_norm),  # training-loss scale
    }


def train_one_epoch(model, loader, opt, i_mean, i_std):
    """Objective is unchanged: real-Vpu voltage MSE + LAMBDA * NORMALIZED
    current MSE (the space the current target is z-scored into). Only the
    reported numbers are un-normalized to real units."""
    model.train()
    acc_v, acc_i, acc_i_norm = _new_acc(), _new_acc(), _new_acc()
    for batch in loader:
        batch = move_batch_to_device(batch, DEVICE)
        opt.zero_grad()
        voltage_pred, current_pred = model(batch["graph_batch"], batch["B"], batch["W"])

        # Voltage target is already real Vpu; current target is normalized.
        v_loss = masked_mse(voltage_pred, batch["bus_targets"], batch["bus_mask"])
        c_loss_norm = masked_mse(current_pred, batch["line_targets"], batch["line_mask"])
        loss = v_loss + LAMBDA_CURRENT * c_loss_norm
        loss.backward()
        opt.step()

        # Metrics only -- detached, no autograd graph retained across the epoch.
        with torch.no_grad():
            _accumulate_batch(acc_v, acc_i, acc_i_norm, batch,
                              voltage_pred.detach(), current_pred.detach(), i_mean, i_std)

    return _metrics(acc_v, acc_i, acc_i_norm)


def evaluate(model, loader, i_mean, i_std):
    """Shared by val (called every epoch) and test (called once at the
    end) -- no gradient, frozen weights, one pass through. Returns real-unit
    MSE (Vpu^2 / Amps^2) and per-entity R^2 for both heads, plus the
    normalized current MSE (training-loss scale) as a reference."""
    model.eval()
    acc_v, acc_i, acc_i_norm = _new_acc(), _new_acc(), _new_acc()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, DEVICE)
            voltage_pred, current_pred = model(batch["graph_batch"], batch["B"], batch["W"])
            _accumulate_batch(acc_v, acc_i, acc_i_norm, batch,
                              voltage_pred, current_pred, i_mean, i_std)

    return _metrics(acc_v, acc_i, acc_i_norm)


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
        log("Reporting in real units: V-MSE in Vpu^2, I-MSE in Amps^2; R^2 is "
            "per-entity (variance-weighted), same convention as hetero_gnn_model.py.",
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

        # Per-line current target stats (fit on TRAIN inside fit_normalization),
        # used to un-normalize current predictions/targets back to Amps for
        # the reported MSE/R^2. Aligned with the line_segment edge order, i.e.
        # the last dim of current_pred / line_targets. NaN entries are
        # unlabeled lines and are always excluded by the label mask.
        i_mean, i_std = ds._current_norm_stats
        i_mean = i_mean.to(DEVICE)
        i_std = i_std.to(DEVICE)

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
            train = train_one_epoch(model, train_loader, opt, i_mean, i_std)
            val = evaluate(model, val_loader, i_mean, i_std)

            # Same combined objective the model is actually trained on --
            # this is what decides "best", not either metric in isolation.
            # NOTE: uses the NORMALIZED current MSE on purpose. The reported
            # I-MSE is in Amps^2, which is orders of magnitude larger than
            # Vpu^2; feeding that in here would let current alone decide
            # every checkpoint and silently change model selection.
            val_metric = val["v_mse"] + LAMBDA_CURRENT * val["i_mse_norm"]
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
                f"train V-MSE {train['v_mse']:.6f} V-R2 {train['v_r2']:.4f} "
                f"I-MSE {train['i_mse']:.6f} I-R2 {train['i_r2']:.4f} | "
                f"val V-MSE {val['v_mse']:.6f} V-R2 {val['v_r2']:.4f} "
                f"I-MSE {val['i_mse']:.6f} I-R2 {val['i_r2']:.4f}"
                f"{'  <- best so far, checkpointed' if improved else ''}",
                log_file,
            )

            if epochs_since_improvement >= EARLY_STOP_PATIENCE:
                log(f"No val improvement for {EARLY_STOP_PATIENCE} epochs, stopping early "
                    f"at epoch {epoch} (best was epoch {best_epoch}).", log_file)
                break

        log(f"\nBest val checkpoint: epoch {best_epoch} "
            f"(val V-MSE + {LAMBDA_CURRENT}*I-MSE[normalized] = {best_val_metric:.6f})",
            log_file)

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
        test = evaluate(model, test_loader, i_mean, i_std)
        log(
            f"\nFINAL TEST (best-val checkpoint from epoch {best_epoch}, "
            f"held out through all of training) | "
            f"V-MSE {test['v_mse']:.6f} Vpu^2 V-R2 {test['v_r2']:.4f} | "
            f"I-MSE {test['i_mse']:.6f} A^2 I-R2 {test['i_r2']:.4f} "
            f"(I-MSE normalized {test['i_mse_norm']:.6f})",
            log_file,
        )

    print(f"\nFull log written to {log_path}")


if __name__ == "__main__":
    main()
