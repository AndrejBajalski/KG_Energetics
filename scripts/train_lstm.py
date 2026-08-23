"""
Train and evaluate a CSV-only LSTM forecasting baseline.

Inputs:
    data/Computed/load_profiles.csv   (past load kW)
    data/Computed/bus_voltages.csv    (past observations + future targets)
    data/Computed/line_currents.csv   (past observations + future targets)

No Neo4j data and no graph operations are used. Results are written under
the project-root lstm-data-results/ directory.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lstm_dataset import (
    FeatureStandardizer,
    LSTMForecastWindowDataset,
    fit_standardizers,
    load_computed_time_series,
)
from lstm_model import CSVForecastLSTM, count_trainable_parameters


BASE_DIR = Path(__file__).resolve().parents[1]
COMPUTED_DIR = BASE_DIR / "data" / "Computed"
RESULTS_DIR = BASE_DIR / "lstm-data-results"

WINDOW = 30
HORIZON = 15
HIDDEN = 192
LSTM_LAYERS = 2
DROPOUT = 0.1
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
LAMBDA_CURRENT = 0.3
EARLY_STOPPING_PATIENCE = 20
LR_PATIENCE = 8
LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-5
GRAD_CLIP_NORM = 1.0
TREND_COEFFICIENT_CLIP = 20.0
VOLTAGE_PCA_COMPONENTS = 16
CURRENT_PCA_COMPONENTS = 64
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    dataset: LSTMForecastWindowDataset,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def evaluate_normalized(
    model: CSVForecastLSTM,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    sums = {"total": 0.0, "voltage": 0.0, "current": 0.0}
    examples = 0

    with torch.no_grad():
        for batch in loader:
            load_history = batch["load_history"].to(device, non_blocking=True)
            voltage_history = batch["voltage_history"].to(device, non_blocking=True)
            current_history = batch["current_history"].to(device, non_blocking=True)
            voltage_target = batch["voltage_targets"].to(device, non_blocking=True)
            current_target = batch["current_targets"].to(device, non_blocking=True)
            voltage_pred, current_pred = model(
                load_history, voltage_history, current_history
            )

            voltage_loss = F.mse_loss(voltage_pred, voltage_target)
            current_loss = F.mse_loss(current_pred, current_target)
            total_loss = voltage_loss + LAMBDA_CURRENT * current_loss

            batch_size = load_history.size(0)
            sums["total"] += total_loss.item() * batch_size
            sums["voltage"] += voltage_loss.item() * batch_size
            sums["current"] += current_loss.item() * batch_size
            examples += batch_size

    return {name: value / examples for name, value in sums.items()}


def scaler_state(scaler: FeatureStandardizer) -> dict[str, torch.Tensor]:
    return {"mean": scaler.mean.cpu(), "std": scaler.std.cpu()}


def fit_trend_coefficients(
    values: torch.Tensor,
    origins: list[int],
    window: int,
    horizon: int,
) -> torch.Tensor:
    """Fit a per-target mean-reversion coefficient using training windows only."""

    origin_index = torch.tensor(origins, dtype=torch.long)
    slope = (values[origin_index] - values[origin_index - (window - 1)]) / (window - 1)
    denominator = slope.square().sum(dim=0).clamp(min=1e-8)
    coefficients = []
    for step in range(1, horizon + 1):
        future_change = values[origin_index + step] - values[origin_index]
        coefficients.append((slope * future_change).sum(dim=0) / denominator)
    return torch.stack(coefficients).nan_to_num().clamp(
        min=-TREND_COEFFICIENT_CLIP, max=TREND_COEFFICIENT_CLIP
    )


def fit_spatial_basis(
    values: torch.Tensor,
    train_range: tuple[int, int],
    components: int,
) -> torch.Tensor:
    """Fit a training-only PCA basis for coherent low-rank output changes."""

    lo, hi = train_range
    train_values = values[lo:hi]
    rank = min(components, train_values.size(0), train_values.size(1))
    _, _, basis = torch.pca_lowrank(
        train_values,
        q=rank,
        center=False,
        niter=4,
    )
    return basis


def save_checkpoint(
    path: Path,
    model: CSVForecastLSTM,
    epoch: int,
    best_val_loss: float,
    load_scaler: FeatureStandardizer,
    voltage_scaler: FeatureStandardizer,
    current_scaler: FeatureStandardizer,
    names: dict[str, list[str]],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model_config": model.configuration(),
            "model_state_dict": model.state_dict(),
            "scalers": {
                "loads": scaler_state(load_scaler),
                "voltages": scaler_state(voltage_scaler),
                "currents": scaler_state(current_scaler),
            },
            "names": names,
        },
        path,
    )


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    difference = prediction.astype(np.float64) - target.astype(np.float64)
    return {
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "max_absolute_error": float(np.max(np.abs(difference))),
    }


def predict_test_set(
    model: CSVForecastLSTM,
    loader: DataLoader,
    device: torch.device,
    voltage_scaler: FeatureStandardizer,
    current_scaler: FeatureStandardizer,
) -> dict[str, np.ndarray]:
    model.eval()
    voltage_mean = voltage_scaler.mean.to(device)
    voltage_std = voltage_scaler.std.to(device)
    current_mean = current_scaler.mean.to(device)
    current_std = current_scaler.std.to(device)

    collected: dict[str, list[np.ndarray]] = {
        "voltage_predictions": [],
        "voltage_targets": [],
        "current_predictions": [],
        "current_targets": [],
        "persistence_voltage_predictions": [],
        "persistence_current_predictions": [],
        "trend_voltage_predictions": [],
        "trend_current_predictions": [],
        "origin_indices": [],
        "target_indices": [],
    }

    with torch.no_grad():
        for batch in loader:
            load_history = batch["load_history"].to(device, non_blocking=True)
            voltage_history = batch["voltage_history"].to(device, non_blocking=True)
            current_history = batch["current_history"].to(device, non_blocking=True)
            voltage_target = batch["voltage_targets"].to(device, non_blocking=True)
            current_target = batch["current_targets"].to(device, non_blocking=True)
            voltage_pred, current_pred = model(
                load_history, voltage_history, current_history
            )

            persistence_voltage = voltage_history[:, -1:, :].expand_as(voltage_pred)
            persistence_current = current_history[:, -1:, :].expand_as(current_pred)
            trend_voltage, trend_current = model.trend_baseline(
                voltage_history, current_history
            )

            voltage_pred = voltage_pred * voltage_std + voltage_mean
            voltage_target = voltage_target * voltage_std + voltage_mean
            current_pred = current_pred * current_std + current_mean
            current_target = current_target * current_std + current_mean
            persistence_voltage = persistence_voltage * voltage_std + voltage_mean
            persistence_current = persistence_current * current_std + current_mean
            trend_voltage = trend_voltage * voltage_std + voltage_mean
            trend_current = trend_current * current_std + current_mean

            collected["voltage_predictions"].append(voltage_pred.cpu().numpy())
            collected["voltage_targets"].append(voltage_target.cpu().numpy())
            collected["current_predictions"].append(current_pred.cpu().numpy())
            collected["current_targets"].append(current_target.cpu().numpy())
            collected["persistence_voltage_predictions"].append(
                persistence_voltage.cpu().numpy()
            )
            collected["persistence_current_predictions"].append(
                persistence_current.cpu().numpy()
            )
            collected["trend_voltage_predictions"].append(trend_voltage.cpu().numpy())
            collected["trend_current_predictions"].append(trend_current.cpu().numpy())
            collected["origin_indices"].append(batch["origin_index"].numpy())
            collected["target_indices"].append(batch["target_indices"].numpy())

    return {name: np.concatenate(parts, axis=0) for name, parts in collected.items()}


def save_test_results(
    predictions: dict[str, np.ndarray],
    timesteps: np.ndarray,
    voltage_names: list[str],
    current_names: list[str],
    results_dir: Path,
) -> dict[str, object]:
    voltage_pred = predictions["voltage_predictions"]
    voltage_true = predictions["voltage_targets"]
    current_pred = predictions["current_predictions"]
    current_true = predictions["current_targets"]
    persistence_voltage = predictions["persistence_voltage_predictions"]
    persistence_current = predictions["persistence_current_predictions"]
    trend_voltage = predictions["trend_voltage_predictions"]
    trend_current = predictions["trend_current_predictions"]
    target_indices = predictions["target_indices"]

    per_horizon_rows = []
    for horizon_index in range(voltage_pred.shape[1]):
        voltage_metrics = regression_metrics(
            voltage_pred[:, horizon_index], voltage_true[:, horizon_index]
        )
        current_metrics = regression_metrics(
            current_pred[:, horizon_index], current_true[:, horizon_index]
        )
        per_horizon_rows.append(
            {
                "horizon_minute": horizon_index + 1,
                "voltage_mae_vpu": voltage_metrics["mae"],
                "voltage_rmse_vpu": voltage_metrics["rmse"],
                "voltage_max_abs_error_vpu": voltage_metrics["max_absolute_error"],
                "current_mae_a": current_metrics["mae"],
                "current_rmse_a": current_metrics["rmse"],
                "current_max_abs_error_a": current_metrics["max_absolute_error"],
            }
        )
    pd.DataFrame(per_horizon_rows).to_csv(
        results_dir / "test_metrics_by_horizon.csv", index=False
    )

    model_voltage_metrics = regression_metrics(voltage_pred, voltage_true)
    model_current_metrics = regression_metrics(current_pred, current_true)
    baseline_voltage_metrics = regression_metrics(persistence_voltage, voltage_true)
    baseline_current_metrics = regression_metrics(persistence_current, current_true)
    trend_voltage_metrics = regression_metrics(trend_voltage, voltage_true)
    trend_current_metrics = regression_metrics(trend_current, current_true)
    overall = {
        "voltage_vpu": regression_metrics(voltage_pred, voltage_true),
        "current_amperes": regression_metrics(current_pred, current_true),
        "persistence_baseline": {
            "voltage_vpu": baseline_voltage_metrics,
            "current_amperes": baseline_current_metrics,
        },
        "training_fitted_trend_baseline": {
            "voltage_vpu": trend_voltage_metrics,
            "current_amperes": trend_current_metrics,
        },
        "rmse_improvement_over_persistence_percent": {
            "voltage": 100.0
            * (baseline_voltage_metrics["rmse"] - model_voltage_metrics["rmse"])
            / baseline_voltage_metrics["rmse"],
            "current": 100.0
            * (baseline_current_metrics["rmse"] - model_current_metrics["rmse"])
            / baseline_current_metrics["rmse"],
        },
        "test_windows": int(voltage_pred.shape[0]),
        "forecast_horizon": int(voltage_pred.shape[1]),
    }
    with (results_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(overall, handle, indent=2)

    # Full multi-horizon arrays are kept in a compressed binary file to avoid
    # creating enormous, slow CSVs. Names and indices make every value traceable.
    np.savez_compressed(
        results_dir / "test_predictions.npz",
        **predictions,
        target_timesteps=timesteps[target_indices],
        voltage_bus_names=np.asarray(voltage_names),
        current_line_names=np.asarray(current_names),
    )

    # Also save the one-minute-ahead predictions as human-readable CSV files.
    voltage_h1 = pd.DataFrame(voltage_pred[:, 0, :], columns=voltage_names)
    voltage_h1.insert(0, "target_timestep", timesteps[target_indices[:, 0]])
    voltage_h1.to_csv(results_dir / "test_voltage_predictions_h1.csv", index=False)

    current_h1 = pd.DataFrame(current_pred[:, 0, :], columns=current_names)
    current_h1.insert(0, "target_timestep", timesteps[target_indices[:, 0]])
    current_h1.to_csv(results_dir / "test_current_predictions_h1.csv", index=False)
    return overall


def main() -> None:
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_computed_time_series(
        load_profiles_path=COMPUTED_DIR / "load_profiles.csv",
        voltage_labels_path=COMPUTED_DIR / "bus_voltages.csv",
        line_currents_path=COMPUTED_DIR / "line_currents.csv",
    )
    n = data.n_timesteps
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    train_range = (0, n_train)
    val_range = (n_train, n_train + n_val)
    test_range = (n_train + n_val, n)

    load_scaler, voltage_scaler, current_scaler = fit_standardizers(data, train_range)
    normalized_loads = load_scaler.transform(data.loads)
    normalized_voltages = voltage_scaler.transform(data.voltages)
    normalized_currents = current_scaler.transform(data.currents)

    dataset_args = {
        "normalized_loads": normalized_loads,
        "normalized_voltages": normalized_voltages,
        "normalized_currents": normalized_currents,
        "window": WINDOW,
        "horizon": HORIZON,
    }
    train_dataset = LSTMForecastWindowDataset(**dataset_args, split_range=train_range)
    val_dataset = LSTMForecastWindowDataset(**dataset_args, split_range=val_range)
    test_dataset = LSTMForecastWindowDataset(**dataset_args, split_range=test_range)

    train_loader = make_loader(train_dataset, shuffle=True, device=device)
    val_loader = make_loader(val_dataset, shuffle=False, device=device)
    test_loader = make_loader(test_dataset, shuffle=False, device=device)

    model = CSVForecastLSTM(
        load_input_dim=data.loads.size(1),
        num_voltage_targets=data.voltages.size(1),
        num_current_targets=data.currents.size(1),
        hidden=HIDDEN,
        horizon=HORIZON,
        lstm_layers=LSTM_LAYERS,
        dropout=DROPOUT,
        voltage_basis=fit_spatial_basis(
            normalized_voltages, train_range, VOLTAGE_PCA_COMPONENTS
        ),
        current_basis=fit_spatial_basis(
            normalized_currents, train_range, CURRENT_PCA_COMPONENTS
        ),
        voltage_trend_coefficients=fit_trend_coefficients(
            normalized_voltages,
            train_dataset.valid_t,
            WINDOW,
            HORIZON,
        ),
        current_trend_coefficients=fit_trend_coefficients(
            normalized_currents,
            train_dataset.valid_t,
            WINDOW,
            HORIZON,
        ),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=MIN_LEARNING_RATE,
    )

    config = {
        "window": WINDOW,
        "horizon": HORIZON,
        "hidden": HIDDEN,
        "lstm_layers": LSTM_LAYERS,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "lambda_current": LAMBDA_CURRENT,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "lr_scheduler_patience": LR_PATIENCE,
        "lr_scheduler_factor": LR_FACTOR,
        "minimum_learning_rate": MIN_LEARNING_RATE,
        "gradient_clip_norm": GRAD_CLIP_NORM,
        "trend_coefficient_clip": TREND_COEFFICIENT_CLIP,
        "voltage_pca_components": VOLTAGE_PCA_COMPONENTS,
        "current_pca_components": CURRENT_PCA_COMPONENTS,
        "train_fraction": TRAIN_FRAC,
        "validation_fraction": VAL_FRAC,
        "seed": SEED,
        "device": str(device),
        "trainable_parameters": count_trainable_parameters(model),
        "timesteps": n,
        "load_features": data.loads.size(1),
        "voltage_targets": data.voltages.size(1),
        "current_targets": data.currents.size(1),
        "input_definition": "past load, bus-voltage, and line-current observations",
        "train_windows": len(train_dataset),
        "validation_windows": len(val_dataset),
        "test_windows": len(test_dataset),
        "voltage_target_definition": "minimum phase Vpu per bus; SourceBus excluded",
        "current_target_definition": "maximum OpenDSS current magnitude per line in amperes",
    }
    with (RESULTS_DIR / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    names = {
        "loads": data.load_names,
        "voltage_buses": data.voltage_bus_names,
        "current_lines": data.current_line_names,
    }
    checkpoint_path = RESULTS_DIR / "best_model.pt"
    history_rows: list[dict[str, float | int]] = []
    # The zero-initialized residual heads make epoch 0 the training-fitted
    # trend baseline. Save it before optimization so training can never
    # select a checkpoint that validates worse than this baseline.
    initial_val_metrics = evaluate_normalized(model, val_loader, device)
    best_val_loss = initial_val_metrics["total"]
    best_epoch = 0
    stale_epochs = 0
    save_checkpoint(
        checkpoint_path,
        model,
        best_epoch,
        best_val_loss,
        load_scaler,
        voltage_scaler,
        current_scaler,
        names,
    )

    print(
        f"device={device} | train/val/test windows="
        f"{len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)} | "
        f"parameters={config['trainable_parameters']:,}"
    )
    print(
        f"epoch 000 trend baseline | val V-MSE {initial_val_metrics['voltage']:.6f} "
        f"I-MSE {initial_val_metrics['current']:.6f}"
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        sums = {"total": 0.0, "voltage": 0.0, "current": 0.0}
        examples = 0

        for batch in train_loader:
            load_history = batch["load_history"].to(device, non_blocking=True)
            voltage_history = batch["voltage_history"].to(device, non_blocking=True)
            current_history = batch["current_history"].to(device, non_blocking=True)
            voltage_target = batch["voltage_targets"].to(device, non_blocking=True)
            current_target = batch["current_targets"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            voltage_pred, current_pred = model(
                load_history, voltage_history, current_history
            )
            voltage_loss = F.mse_loss(voltage_pred, voltage_target)
            current_loss = F.mse_loss(current_pred, current_target)
            total_loss = voltage_loss + LAMBDA_CURRENT * current_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            batch_size = load_history.size(0)
            sums["total"] += total_loss.item() * batch_size
            sums["voltage"] += voltage_loss.item() * batch_size
            sums["current"] += current_loss.item() * batch_size
            examples += batch_size

        train_metrics = {name: value / examples for name, value in sums.items()}
        val_metrics = evaluate_normalized(model, val_loader, device)
        scheduler.step(val_metrics["total"])
        learning_rate = optimizer.param_groups[0]["lr"]
        history_rows.append(
            {
                "epoch": epoch,
                "train_total_loss": train_metrics["total"],
                "train_voltage_mse_normalized": train_metrics["voltage"],
                "train_current_mse_normalized": train_metrics["current"],
                "val_total_loss": val_metrics["total"],
                "val_voltage_mse_normalized": val_metrics["voltage"],
                "val_current_mse_normalized": val_metrics["current"],
                "learning_rate": learning_rate,
            }
        )
        pd.DataFrame(history_rows).to_csv(RESULTS_DIR / "training_history.csv", index=False)

        print(
            f"epoch {epoch:03d} | "
            f"train V-MSE {train_metrics['voltage']:.6f} "
            f"I-MSE {train_metrics['current']:.6f} | "
            f"val V-MSE {val_metrics['voltage']:.6f} "
            f"I-MSE {val_metrics['current']:.6f} | lr {learning_rate:.2e}"
        )

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(
                checkpoint_path,
                model,
                epoch,
                best_val_loss,
                load_scaler,
                voltage_scaler,
                current_scaler,
                names,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= EARLY_STOPPING_PATIENCE:
                print(f"early stopping after {epoch} epochs; best epoch was {best_epoch}")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    predictions = predict_test_set(
        model, test_loader, device, voltage_scaler, current_scaler
    )
    test_metrics = save_test_results(
        predictions,
        data.timesteps,
        data.voltage_bus_names,
        data.current_line_names,
        RESULTS_DIR,
    )

    run_summary = {
        "best_epoch": best_epoch,
        "best_validation_total_loss": best_val_loss,
        "test_metrics": test_metrics,
        "results_directory": str(RESULTS_DIR),
    }
    with (RESULTS_DIR / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2)

    print(
        f"best epoch={best_epoch} | "
        f"test V-RMSE={test_metrics['voltage_vpu']['rmse']:.6f} Vpu | "
        f"test I-RMSE={test_metrics['current_amperes']['rmse']:.6f} A"
    )
    print(f"saved LSTM results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
