"""Tune the CSV-only LSTM for five-minute-ahead line-current R2.

Trials are ranked only on the chronological validation interval. The test
interval is deliberately not loaded or evaluated here; train_lstm.py performs
the final held-out evaluation after the winning configuration is selected.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lstm_dataset import LSTMForecastWindowDataset, fit_standardizers, load_computed_time_series
from lstm_model import CSVForecastLSTM, count_trainable_parameters
from train_lstm import (
    COMPUTED_DIR,
    HORIZON,
    TRAIN_FRAC,
    TREND_COEFFICIENT_CLIP,
    VAL_FRAC,
    fit_spatial_basis,
    fit_trend_coefficients,
    set_seed,
)


RESULTS_DIR = Path(__file__).resolve().parents[1] / "lstm-tuning-results"
FOCUS_HORIZON = 5
MAX_EPOCHS = 35
EARLY_STOPPING_PATIENCE = 8
SCHEDULER_PATIENCE = 4
SCHEDULER_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-5
GRAD_CLIP_NORM = 1.0
SEED = 42


@dataclass(frozen=True)
class TrialConfig:
    name: str
    window: int = 30
    hidden: int = 128
    lstm_layers: int = 1
    dropout: float = 0.2
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    lambda_voltage: float = 0.3
    loss_function: str = "huber"
    huber_delta: float = 1.0
    current_focus_weight: float = 1.0
    voltage_pca_components: int = 16
    current_pca_components: int = 64


TRIALS = [
    TrialConfig("baseline", lambda_voltage=1.0),
    TrialConfig("vweight_070", lambda_voltage=0.7),
    TrialConfig("vweight_030", lambda_voltage=0.3),
    TrialConfig("vweight_010", lambda_voltage=0.1),
    TrialConfig("current_only", lambda_voltage=0.0),
    TrialConfig("mse_v030", lambda_voltage=0.3, loss_function="mse"),
    TrialConfig("mse_current_only", lambda_voltage=0.0, loss_function="mse"),
    TrialConfig("huber_delta_050", lambda_voltage=0.3, huber_delta=0.5),
    TrialConfig("huber_delta_200", lambda_voltage=0.3, huber_delta=2.0),
    TrialConfig("focus_weight_3", lambda_voltage=0.3, current_focus_weight=3.0),
    TrialConfig("focus_weight_8", lambda_voltage=0.3, current_focus_weight=8.0),
    TrialConfig(
        "hidden_64_focus_3", hidden=64, lambda_voltage=0.3, current_focus_weight=3.0
    ),
    TrialConfig(
        "hidden_192_focus_3", hidden=192, lambda_voltage=0.3, current_focus_weight=3.0
    ),
    TrialConfig(
        "layers_2_focus_3",
        lstm_layers=2,
        lambda_voltage=0.3,
        current_focus_weight=3.0,
    ),
    TrialConfig(
        "current_pca_32", lambda_voltage=0.3, current_pca_components=32
    ),
    TrialConfig(
        "current_pca_128", lambda_voltage=0.3, current_pca_components=128
    ),
    TrialConfig("window_15", window=15, lambda_voltage=0.3),
    TrialConfig("window_60", window=60, lambda_voltage=0.3),
    TrialConfig("learning_rate_2e4", lambda_voltage=0.3, learning_rate=2e-4),
    TrialConfig("learning_rate_5e5", lambda_voltage=0.3, learning_rate=5e-5),
    TrialConfig(
        "batch_64_focus_3",
        batch_size=64,
        lambda_voltage=0.3,
        current_focus_weight=3.0,
    ),
    TrialConfig(
        "dropout_030_focus_3",
        dropout=0.3,
        lambda_voltage=0.3,
        current_focus_weight=3.0,
    ),
    TrialConfig("refine_window_45", window=45, lambda_voltage=0.3),
    TrialConfig("refine_window_75", window=75, lambda_voltage=0.3),
    TrialConfig("refine_window_90", window=90, lambda_voltage=0.3),
    TrialConfig("refine_window_120", window=120, lambda_voltage=0.3),
    TrialConfig("w60_vweight_100", window=60, lambda_voltage=1.0),
    TrialConfig("w60_vweight_070", window=60, lambda_voltage=0.7),
    TrialConfig("w60_vweight_010", window=60, lambda_voltage=0.1),
    TrialConfig("w60_current_only", window=60, lambda_voltage=0.0),
    TrialConfig("w60_mse", window=60, lambda_voltage=0.3, loss_function="mse"),
    TrialConfig(
        "w60_focus_weight_3",
        window=60,
        lambda_voltage=0.3,
        current_focus_weight=3.0,
    ),
    TrialConfig(
        "w60_focus_weight_8",
        window=60,
        lambda_voltage=0.3,
        current_focus_weight=8.0,
    ),
    TrialConfig("w60_hidden_192", window=60, hidden=192, lambda_voltage=0.3),
    TrialConfig(
        "w60_current_pca_128",
        window=60,
        lambda_voltage=0.3,
        current_pca_components=128,
    ),
    TrialConfig(
        "w60_hidden_192_pca_128",
        window=60,
        hidden=192,
        lambda_voltage=0.3,
        current_pca_components=128,
    ),
    TrialConfig(
        "w60_learning_rate_2e4", window=60, lambda_voltage=0.3, learning_rate=2e-4
    ),
    TrialConfig(
        "w60_learning_rate_5e5", window=60, lambda_voltage=0.3, learning_rate=5e-5
    ),
    TrialConfig("w60_dropout_010", window=60, dropout=0.1, lambda_voltage=0.3),
    TrialConfig("w60_dropout_030", window=60, dropout=0.3, lambda_voltage=0.3),
    TrialConfig("w60_batch_64", window=60, batch_size=64, lambda_voltage=0.3),
    TrialConfig(
        "w60_layers_2", window=60, lstm_layers=2, lambda_voltage=0.3
    ),
    TrialConfig("refine_window_66", window=66, lambda_voltage=0.3),
    TrialConfig("refine_window_76", window=76, lambda_voltage=0.3),
    TrialConfig("refine_window_77", window=77, lambda_voltage=0.3),
    TrialConfig("refine_window_78", window=78, lambda_voltage=0.3),
    TrialConfig("refine_window_79", window=79, lambda_voltage=0.3),
    TrialConfig("refine_window_102", window=102, lambda_voltage=0.3),
    TrialConfig("refine_window_119", window=119, lambda_voltage=0.3),
    TrialConfig("w78_vweight_100", window=78, lambda_voltage=1.0),
    TrialConfig("w78_vweight_010", window=78, lambda_voltage=0.1),
    TrialConfig("w78_current_only", window=78, lambda_voltage=0.0),
    TrialConfig("w78_mse", window=78, lambda_voltage=0.3, loss_function="mse"),
    TrialConfig("w78_hidden_192", window=78, hidden=192, lambda_voltage=0.3),
    TrialConfig("w78_hidden_256", window=78, hidden=256, lambda_voltage=0.3),
    TrialConfig(
        "w78_current_pca_128",
        window=78,
        lambda_voltage=0.3,
        current_pca_components=128,
    ),
    TrialConfig(
        "w78_learning_rate_2e4", window=78, lambda_voltage=0.3, learning_rate=2e-4
    ),
    TrialConfig(
        "w78_learning_rate_5e5", window=78, lambda_voltage=0.3, learning_rate=5e-5
    ),
    TrialConfig("w78_dropout_000", window=78, dropout=0.0, lambda_voltage=0.3),
    TrialConfig("w78_dropout_030", window=78, dropout=0.3, lambda_voltage=0.3),
    TrialConfig("w78_batch_64", window=78, batch_size=64, lambda_voltage=0.3),
    TrialConfig("w78_layers_2", window=78, lstm_layers=2, lambda_voltage=0.3),
    TrialConfig("w78_weight_decay_0", window=78, weight_decay=0.0, lambda_voltage=0.3),
    TrialConfig(
        "w78_weight_decay_1e3", window=78, weight_decay=1e-3, lambda_voltage=0.3
    ),
    TrialConfig("refine_window_100", window=100, lambda_voltage=0.3),
    TrialConfig("refine_window_101", window=101, lambda_voltage=0.3),
    TrialConfig("refine_window_103", window=103, lambda_voltage=0.3),
    TrialConfig("refine_window_104", window=104, lambda_voltage=0.3),
    TrialConfig("refine_window_105", window=105, lambda_voltage=0.3),
    TrialConfig("w102_vweight_100", window=102, lambda_voltage=1.0),
    TrialConfig("w102_vweight_010", window=102, lambda_voltage=0.1),
    TrialConfig("w102_current_only", window=102, lambda_voltage=0.0),
    TrialConfig("w102_mse", window=102, lambda_voltage=0.3, loss_function="mse"),
    TrialConfig("w102_hidden_192", window=102, hidden=192, lambda_voltage=0.3),
    TrialConfig("w102_hidden_256", window=102, hidden=256, lambda_voltage=0.3),
    TrialConfig(
        "w102_current_pca_128",
        window=102,
        lambda_voltage=0.3,
        current_pca_components=128,
    ),
    TrialConfig(
        "w102_hidden_192_pca_128",
        window=102,
        hidden=192,
        lambda_voltage=0.3,
        current_pca_components=128,
    ),
    TrialConfig(
        "w102_learning_rate_5e5", window=102, lambda_voltage=0.3, learning_rate=5e-5
    ),
    TrialConfig("w102_batch_64", window=102, batch_size=64, lambda_voltage=0.3),
    TrialConfig("w102_layers_2", window=102, lstm_layers=2, lambda_voltage=0.3),
    TrialConfig(
        "w102_hidden_192_mse",
        window=102,
        hidden=192,
        lambda_voltage=0.3,
        loss_function="mse",
    ),
    TrialConfig(
        "w102_hidden_192_vweight_010",
        window=102,
        hidden=192,
        lambda_voltage=0.1,
    ),
    TrialConfig(
        "w102_hidden_192_current_only",
        window=102,
        hidden=192,
        lambda_voltage=0.0,
    ),
    TrialConfig(
        "w102_hidden_192_batch_64",
        window=102,
        hidden=192,
        batch_size=64,
        lambda_voltage=0.3,
    ),
    TrialConfig(
        "w102_hidden_192_layers_2",
        window=102,
        hidden=192,
        lstm_layers=2,
        lambda_voltage=0.3,
    ),
    TrialConfig(
        "w102_hidden_192_learning_rate_5e5",
        window=102,
        hidden=192,
        lambda_voltage=0.3,
        learning_rate=5e-5,
    ),
    TrialConfig("w102_hidden_160", window=102, hidden=160, lambda_voltage=0.3),
    TrialConfig("w102_hidden_224", window=102, hidden=224, lambda_voltage=0.3),
    TrialConfig("w102_hidden_208", window=102, hidden=208, lambda_voltage=0.3),
    TrialConfig("w102_hidden_240", window=102, hidden=240, lambda_voltage=0.3),
    TrialConfig(
        "w102_hidden_224_mse",
        window=102,
        hidden=224,
        lambda_voltage=0.3,
        loss_function="mse",
    ),
    TrialConfig(
        "w102_hidden_224_vweight_010",
        window=102,
        hidden=224,
        lambda_voltage=0.1,
    ),
    TrialConfig(
        "w102_hidden_224_learning_rate_5e5",
        window=102,
        hidden=224,
        lambda_voltage=0.3,
        learning_rate=5e-5,
    ),
    TrialConfig(
        "w102_hidden_224_batch_64",
        window=102,
        hidden=224,
        batch_size=64,
        lambda_voltage=0.3,
    ),
]


def elementwise_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_function: str,
    huber_delta: float,
) -> torch.Tensor:
    if loss_function == "huber":
        return F.huber_loss(prediction, target, delta=huber_delta, reduction="none")
    if loss_function == "mse":
        return F.mse_loss(prediction, target, reduction="none")
    raise ValueError(f"unsupported loss function: {loss_function}")


def reduced_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    config: TrialConfig,
    focus_current: bool,
) -> torch.Tensor:
    per_horizon = elementwise_loss(
        prediction, target, config.loss_function, config.huber_delta
    ).mean(dim=(0, 2))
    if not focus_current or config.current_focus_weight == 1.0:
        return per_horizon.mean()
    weights = torch.ones_like(per_horizon)
    weights[FOCUS_HORIZON - 1] = config.current_focus_weight
    return (per_horizon * weights).sum() / weights.sum()


@torch.no_grad()
def evaluate(
    model: CSVForecastLSTM,
    loader: DataLoader,
    device: torch.device,
    current_mean: torch.Tensor,
    current_std: torch.Tensor,
    voltage_mean: torch.Tensor,
    voltage_std: torch.Tensor,
    config: TrialConfig,
) -> dict[str, float]:
    model.eval()
    loss_sums = {"total": 0.0, "voltage": 0.0, "current": 0.0}
    examples = 0
    stats = {
        "current_sse": 0.0,
        "current_sum": 0.0,
        "current_sum_sq": 0.0,
        "current_count": 0,
        "voltage_sse": 0.0,
        "voltage_sum": 0.0,
        "voltage_sum_sq": 0.0,
        "voltage_count": 0,
    }

    for batch in loader:
        load_history = batch["load_history"].to(device, non_blocking=True)
        voltage_history = batch["voltage_history"].to(device, non_blocking=True)
        current_history = batch["current_history"].to(device, non_blocking=True)
        voltage_target = batch["voltage_targets"].to(device, non_blocking=True)
        current_target = batch["current_targets"].to(device, non_blocking=True)
        voltage_pred, current_pred = model(
            load_history, voltage_history, current_history
        )

        voltage_loss = reduced_loss(voltage_pred, voltage_target, config, False)
        current_loss = reduced_loss(current_pred, current_target, config, True)
        total_loss = config.lambda_voltage * voltage_loss + current_loss
        batch_size = load_history.size(0)
        loss_sums["total"] += total_loss.item() * batch_size
        loss_sums["voltage"] += voltage_loss.item() * batch_size
        loss_sums["current"] += current_loss.item() * batch_size
        examples += batch_size

        h = FOCUS_HORIZON - 1
        current_pred_h = current_pred[:, h] * current_std + current_mean
        current_target_h = current_target[:, h] * current_std + current_mean
        current_error = (current_pred_h - current_target_h).double()
        current_target_64 = current_target_h.double()
        stats["current_sse"] += current_error.square().sum().item()
        stats["current_sum"] += current_target_64.sum().item()
        stats["current_sum_sq"] += current_target_64.square().sum().item()
        stats["current_count"] += current_target_64.numel()

        voltage_pred_h = voltage_pred[:, h] * voltage_std + voltage_mean
        voltage_target_h = voltage_target[:, h] * voltage_std + voltage_mean
        voltage_error = (voltage_pred_h - voltage_target_h).double()
        voltage_target_64 = voltage_target_h.double()
        stats["voltage_sse"] += voltage_error.square().sum().item()
        stats["voltage_sum"] += voltage_target_64.sum().item()
        stats["voltage_sum_sq"] += voltage_target_64.square().sum().item()
        stats["voltage_count"] += voltage_target_64.numel()

    result = {name: value / examples for name, value in loss_sums.items()}
    for head in ("current", "voltage"):
        count = stats[f"{head}_count"]
        sse = stats[f"{head}_sse"]
        total_sum_squares = (
            stats[f"{head}_sum_sq"] - stats[f"{head}_sum"] ** 2 / count
        )
        result[f"{head}_mse_h5"] = sse / count
        result[f"{head}_r2_h5"] = 1.0 - sse / total_sum_squares
    return result


def run_trial(
    config: TrialConfig,
    data,
    model_loads: torch.Tensor,
    model_voltages: torch.Tensor,
    model_currents: torch.Tensor,
    train_range: tuple[int, int],
    val_range: tuple[int, int],
    voltage_scaler,
    current_scaler,
    device: torch.device,
    basis_cache: dict[tuple[str, int], torch.Tensor],
) -> dict[str, object]:
    set_seed(SEED)
    dataset_args = {
        "normalized_loads": model_loads,
        "normalized_voltages": model_voltages,
        "normalized_currents": model_currents,
        "window": config.window,
        "horizon": HORIZON,
    }
    train_dataset = LSTMForecastWindowDataset(**dataset_args, split_range=train_range)
    val_dataset = LSTMForecastWindowDataset(**dataset_args, split_range=val_range)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    voltage_key = ("voltage", config.voltage_pca_components)
    current_key = ("current", config.current_pca_components)
    if voltage_key not in basis_cache:
        set_seed(SEED + config.voltage_pca_components)
        basis_cache[voltage_key] = fit_spatial_basis(
            model_voltages, train_range, config.voltage_pca_components
        )
    if current_key not in basis_cache:
        set_seed(SEED + 1000 + config.current_pca_components)
        basis_cache[current_key] = fit_spatial_basis(
            model_currents, train_range, config.current_pca_components
        )

    set_seed(SEED)
    model = CSVForecastLSTM(
        load_input_dim=data.loads.size(1),
        num_voltage_targets=data.voltages.size(1),
        num_current_targets=data.currents.size(1),
        hidden=config.hidden,
        horizon=HORIZON,
        lstm_layers=config.lstm_layers,
        dropout=config.dropout,
        voltage_basis=basis_cache[voltage_key],
        current_basis=basis_cache[current_key],
        voltage_trend_coefficients=fit_trend_coefficients(
            model_voltages, train_dataset.valid_t, config.window, HORIZON
        ).clamp(-TREND_COEFFICIENT_CLIP, TREND_COEFFICIENT_CLIP),
        current_trend_coefficients=fit_trend_coefficients(
            model_currents, train_dataset.valid_t, config.window, HORIZON
        ).clamp(-TREND_COEFFICIENT_CLIP, TREND_COEFFICIENT_CLIP),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=MIN_LEARNING_RATE,
    )
    current_mean = current_scaler.mean.to(device)
    current_std = current_scaler.std.to(device)
    voltage_mean = voltage_scaler.mean.to(device)
    voltage_std = voltage_scaler.std.to(device)

    initial = evaluate(
        model,
        val_loader,
        device,
        current_mean,
        current_std,
        voltage_mean,
        voltage_std,
        config,
    )
    best = initial.copy()
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
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
            voltage_loss = reduced_loss(voltage_pred, voltage_target, config, False)
            current_loss = reduced_loss(current_pred, current_target, config, True)
            total_loss = config.lambda_voltage * voltage_loss + current_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

        metrics = evaluate(
            model,
            val_loader,
            device,
            current_mean,
            current_std,
            voltage_mean,
            voltage_std,
            config,
        )
        scheduler.step(metrics["current_r2_h5"])
        if metrics["current_r2_h5"] > best["current_r2_h5"] + 1e-7:
            best = metrics.copy()
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= EARLY_STOPPING_PATIENCE:
                break

    result: dict[str, object] = asdict(config)
    result.update(
        {
            "parameters": count_trainable_parameters(model),
            "best_epoch": best_epoch,
            "epochs_run": epoch,
            "initial_val_current_r2_h5": initial["current_r2_h5"],
            "best_val_current_r2_h5": best["current_r2_h5"],
            "best_val_current_mse_h5": best["current_mse_h5"],
            "best_val_voltage_r2_h5": best["voltage_r2_h5"],
            "best_val_voltage_mse_h5": best["voltage_mse_h5"],
            "best_val_total_loss": best["total"],
        }
    )
    del model, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_computed_time_series(
        COMPUTED_DIR / "load_profiles.csv",
        COMPUTED_DIR / "bus_voltages.csv",
        COMPUTED_DIR / "line_currents.csv",
    )
    n = data.n_timesteps
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    train_range = (0, n_train)
    val_range = (n_train, n_train + n_val)
    load_scaler, voltage_scaler, current_scaler = fit_standardizers(
        data, train_range, enabled=True
    )
    model_loads = load_scaler.transform(data.loads)
    model_voltages = voltage_scaler.transform(data.voltages)
    model_currents = current_scaler.transform(data.currents)
    basis_cache: dict[tuple[str, int], torch.Tensor] = {}
    results = []
    existing_results: dict[str, dict[str, object]] = {}
    summary_path = RESULTS_DIR / "tuning_summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            previous_summary = json.load(handle)
        existing_results = {
            row["name"]: row for row in previous_summary.get("ranked_trials", [])
        }

    print(
        f"device={device} | trials={len(TRIALS)} | objective=validation current "
        f"R2 at {FOCUS_HORIZON} minutes",
        flush=True,
    )
    for index, config in enumerate(TRIALS, start=1):
        previous = existing_results.get(config.name)
        if previous is not None and all(
            previous.get(key) == value for key, value in asdict(config).items()
        ):
            result = previous
            status = "reused"
        else:
            result = run_trial(
                config,
                data,
                model_loads,
                model_voltages,
                model_currents,
                train_range,
                val_range,
                voltage_scaler,
                current_scaler,
                device,
                basis_cache,
            )
            status = "trained"
        results.append(result)
        print(
            f"[{index:02d}/{len(TRIALS):02d}] {config.name} ({status}): "
            f"best epoch={result['best_epoch']} | "
            f"val I-R2@5={result['best_val_current_r2_h5']:.6f} | "
            f"val I-MSE@5={result['best_val_current_mse_h5']:.6f}",
            flush=True,
        )
        pd.DataFrame(results).sort_values(
            "best_val_current_r2_h5", ascending=False
        ).to_csv(RESULTS_DIR / "trials.csv", index=False)

    ranked = sorted(
        results, key=lambda row: row["best_val_current_r2_h5"], reverse=True
    )
    summary = {
        "objective": "maximize validation current R2 at 5-minute horizon",
        "test_set_used_for_tuning": False,
        "focus_horizon_minutes": FOCUS_HORIZON,
        "trial_count": len(results),
        "best_trial": ranked[0],
        "ranked_trials": ranked,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(
        f"best={ranked[0]['name']} | "
        f"validation current R2@5={ranked[0]['best_val_current_r2_h5']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
