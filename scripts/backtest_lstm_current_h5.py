"""Multi-block backtest for robust five-minute current hyperparameters.

All folds end before the final held-out test interval. Each fold refits its
standardizers, PCA bases, trend coefficients, and model using past data only.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lstm_dataset import fit_standardizers, load_computed_time_series
from train_lstm import COMPUTED_DIR
from tune_lstm_current_h5 import TrialConfig, run_trial


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "lstm-backtest-results"
FOLDS = (
    ("early", (0, 576), (576, 792)),
    ("middle", (0, 792), (792, 1008)),
    ("recent", (0, 1008), (1008, 1224)),
)

CONFIGS = (
    TrialConfig("window_20_h128", window=20, hidden=128, lambda_voltage=0.3),
    TrialConfig("window_21_h128", window=21, hidden=128, lambda_voltage=0.3),
    TrialConfig("window_22_h128", window=22, hidden=128, lambda_voltage=0.3),
    TrialConfig("window_23_h128", window=23, hidden=128, lambda_voltage=0.3),
    TrialConfig("window_30_h128", window=30, hidden=128, lambda_voltage=1.0),
    TrialConfig("window_44_h128", window=44, hidden=128, lambda_voltage=0.3),
    TrialConfig("window_60_h128", window=60, hidden=128, lambda_voltage=0.3),
    TrialConfig("window_80_h128", window=80, hidden=128, lambda_voltage=0.3),
    TrialConfig("window_102_h128", window=102, hidden=128, lambda_voltage=0.3),
    TrialConfig("window_22_h192", window=22, hidden=192, lambda_voltage=0.1),
    TrialConfig("window_22_h224", window=22, hidden=224, lambda_voltage=0.1),
    TrialConfig(
        "window_22_h192_mse",
        window=22,
        hidden=192,
        lambda_voltage=0.1,
        loss_function="mse",
    ),
    TrialConfig(
        "window_22_h224_mse",
        window=22,
        hidden=224,
        lambda_voltage=0.1,
        loss_function="mse",
    ),
)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_computed_time_series(
        COMPUTED_DIR / "load_profiles.csv",
        COMPUTED_DIR / "bus_voltages.csv",
        COMPUTED_DIR / "line_currents.csv",
    )

    fold_contexts = {}
    for fold_name, train_range, val_range in FOLDS:
        load_scaler, voltage_scaler, current_scaler = fit_standardizers(
            data, train_range, enabled=True
        )
        fold_contexts[fold_name] = {
            "train_range": train_range,
            "val_range": val_range,
            "load_scaler": load_scaler,
            "voltage_scaler": voltage_scaler,
            "current_scaler": current_scaler,
            "model_loads": load_scaler.transform(data.loads),
            "model_voltages": voltage_scaler.transform(data.voltages),
            "model_currents": current_scaler.transform(data.currents),
            "basis_cache": {},
        }

    fold_rows = []
    aggregate_rows = []
    print(
        f"device={device} | configs={len(CONFIGS)} | folds={len(FOLDS)} | "
        "objective=mean validation current R2 at 5 minutes",
        flush=True,
    )
    for config_index, config in enumerate(CONFIGS, start=1):
        config_results = []
        for fold_name, _, _ in FOLDS:
            context = fold_contexts[fold_name]
            result = run_trial(
                config,
                data,
                context["model_loads"],
                context["model_voltages"],
                context["model_currents"],
                context["train_range"],
                context["val_range"],
                context["voltage_scaler"],
                context["current_scaler"],
                device,
                context["basis_cache"],
            )
            row = {"config": config.name, "fold": fold_name, **result}
            fold_rows.append(row)
            config_results.append(result)
            print(
                f"[{config_index:02d}/{len(CONFIGS):02d}] {config.name} / "
                f"{fold_name}: R2={result['best_val_current_r2_h5']:.6f} "
                f"MSE={result['best_val_current_mse_h5']:.6f}",
                flush=True,
            )

        scores = np.asarray(
            [row["best_val_current_r2_h5"] for row in config_results],
            dtype=np.float64,
        )
        mses = np.asarray(
            [row["best_val_current_mse_h5"] for row in config_results],
            dtype=np.float64,
        )
        epochs = [int(row["best_epoch"]) for row in config_results]
        aggregate = {
            **asdict(config),
            "mean_val_current_r2_h5": float(scores.mean()),
            "worst_val_current_r2_h5": float(scores.min()),
            "std_val_current_r2_h5": float(scores.std()),
            "mean_val_current_mse_h5": float(mses.mean()),
            "median_best_epoch": int(np.median(epochs)),
            "fold_current_r2_h5": scores.tolist(),
            "fold_current_mse_h5": mses.tolist(),
            "fold_best_epochs": epochs,
        }
        aggregate_rows.append(aggregate)
        pd.DataFrame(fold_rows).to_csv(RESULTS_DIR / "fold_results.csv", index=False)
        pd.DataFrame(aggregate_rows).sort_values(
            "mean_val_current_r2_h5", ascending=False
        ).to_csv(RESULTS_DIR / "config_ranking.csv", index=False)

    ranked = sorted(
        aggregate_rows,
        key=lambda row: row["mean_val_current_r2_h5"],
        reverse=True,
    )
    summary = {
        "objective": "maximize mean validation current R2 at 5 minutes",
        "test_set_used_for_backtest": False,
        "folds": [
            {"name": name, "train_range": train, "validation_range": val}
            for name, train, val in FOLDS
        ],
        "configuration_count": len(CONFIGS),
        "training_run_count": len(CONFIGS) * len(FOLDS),
        "best_configuration": ranked[0],
        "ranked_configurations": ranked,
    }
    with (RESULTS_DIR / "backtest_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(
        f"best={ranked[0]['name']} | mean R2@5="
        f"{ranked[0]['mean_val_current_r2_h5']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
