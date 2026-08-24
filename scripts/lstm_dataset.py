"""
CSV-only temporal dataset for the LSTM baseline.

This module intentionally does not use Neo4j or PyTorch Geometric. It uses
the same forecasting framing as train_temporal.py:

    30 minutes of load, voltage, and current observations -> next 15 minutes
    of bus voltage and line current.

Voltage phase columns are collapsed to one conservative target per bus by
taking the minimum phase voltage, matching hetero_gnn_dataset.py. SourceBus is
excluded because the graph models represent it as a Source rather than a Bus.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class CSVTimeSeries:
    """Aligned raw tensors and names loaded from data/Computed."""

    loads: torch.Tensor
    voltages: torch.Tensor
    currents: torch.Tensor
    timesteps: np.ndarray
    load_names: list[str]
    voltage_bus_names: list[str]
    current_line_names: list[str]

    @property
    def n_timesteps(self) -> int:
        return self.loads.size(0)


@dataclass
class FeatureStandardizer:
    """Per-feature mean/std fitted on training timesteps only."""

    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, values: torch.Tensor, train_range: tuple[int, int]) -> "FeatureStandardizer":
        lo, hi = train_range
        train_values = values[lo:hi]
        mean = train_values.mean(dim=0)
        std = train_values.std(dim=0, unbiased=False).clamp(min=1e-6)
        return cls(mean=mean, std=std)

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.std

    def inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std + self.mean


def _collapse_phase_voltages(raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse OpenDSS columns such as 1.1/1.2/1.3 to minimum Vpu per bus."""

    columns_by_bus: dict[str, list[str]] = {}
    for column in raw.columns:
        bus_name = str(column).split(".")[0]
        columns_by_bus.setdefault(bus_name, []).append(column)

    # Match the deterministic ordering used by hetero_gnn_dataset.py.
    ordered_buses = sorted(columns_by_bus, key=lambda name: (len(name), name))
    collapsed = {
        bus_name: raw[columns_by_bus[bus_name]].min(axis=1)
        for bus_name in ordered_buses
        if bus_name.lower() != "sourcebus"
    }
    return pd.DataFrame(collapsed, index=raw.index)


def _to_finite_tensor(frame: pd.DataFrame, label: str) -> torch.Tensor:
    values = frame.to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(values).all():
        bad = int(values.size - np.isfinite(values).sum())
        raise ValueError(f"{label} contains {bad} missing or non-finite values")
    return torch.from_numpy(values)


def load_computed_time_series(
    load_profiles_path: str | Path,
    voltage_labels_path: str | Path,
    line_currents_path: str | Path,
) -> CSVTimeSeries:
    """Load and validate the three row-aligned generated CSV files."""

    load_frame = pd.read_csv(load_profiles_path, index_col=0)
    raw_voltage_frame = pd.read_csv(voltage_labels_path, index_col=0)
    current_frame = pd.read_csv(line_currents_path, index_col=0)

    if not load_frame.index.equals(raw_voltage_frame.index):
        raise ValueError("load_profiles.csv and bus_voltages.csv indices do not match")
    if not load_frame.index.equals(current_frame.index):
        raise ValueError("load_profiles.csv and line_currents.csv indices do not match")

    voltage_frame = _collapse_phase_voltages(raw_voltage_frame)

    return CSVTimeSeries(
        loads=_to_finite_tensor(load_frame, "load profiles"),
        voltages=_to_finite_tensor(voltage_frame, "bus voltages"),
        currents=_to_finite_tensor(current_frame, "line currents"),
        timesteps=load_frame.index.to_numpy(copy=True),
        load_names=[str(name) for name in load_frame.columns],
        voltage_bus_names=[str(name) for name in voltage_frame.columns],
        current_line_names=[str(name) for name in current_frame.columns],
    )


class LSTMForecastWindowDataset(Dataset):
    """
    Build (past CSV observation window -> future voltage/current horizon)
    examples.

    A target horizon must remain fully inside split_range. History may reach
    before a validation/test boundary because it contains only past inputs.
    """

    def __init__(
        self,
        normalized_loads: torch.Tensor,
        normalized_voltages: torch.Tensor,
        normalized_currents: torch.Tensor,
        window: int = 30,
        horizon: int = 15,
        split_range: tuple[int, int] | None = None,
    ) -> None:
        if window < 1 or horizon < 1:
            raise ValueError("window and horizon must both be positive")

        lengths = {
            normalized_loads.size(0),
            normalized_voltages.size(0),
            normalized_currents.size(0),
        }
        if len(lengths) != 1:
            raise ValueError("load, voltage, and current tensors must have equal lengths")

        self.loads = normalized_loads
        self.voltages = normalized_voltages
        self.currents = normalized_currents
        self.window = window
        self.horizon = horizon

        total = normalized_loads.size(0)
        lo, hi = split_range if split_range is not None else (0, total)
        if not (0 <= lo < hi <= total):
            raise ValueError(f"invalid split range {(lo, hi)} for {total} timesteps")

        self.valid_t: list[int] = []
        for t in range(window - 1, total - horizon):
            first_target = t + 1
            last_target = t + horizon
            if lo <= first_target and last_target < hi:
                self.valid_t.append(t)

        if not self.valid_t:
            raise ValueError(
                f"split {(lo, hi)} has no valid windows for window={window}, "
                f"horizon={horizon}"
            )

    def __len__(self) -> int:
        return len(self.valid_t)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        t = self.valid_t[index]
        history_slice = slice(t - self.window + 1, t + 1)
        target_slice = slice(t + 1, t + 1 + self.horizon)
        return {
            "load_history": self.loads[history_slice],
            "voltage_history": self.voltages[history_slice],
            "current_history": self.currents[history_slice],
            "voltage_targets": self.voltages[target_slice],
            "current_targets": self.currents[target_slice],
            "origin_index": torch.tensor(t, dtype=torch.long),
            "target_indices": torch.arange(t + 1, t + 1 + self.horizon, dtype=torch.long),
        }


def fit_standardizers(
    data: CSVTimeSeries,
    train_range: tuple[int, int],
) -> tuple[FeatureStandardizer, FeatureStandardizer, FeatureStandardizer]:
    """Fit input and target scaling exclusively on the training interval."""

    return (
        FeatureStandardizer.fit(data.loads, train_range),
        FeatureStandardizer.fit(data.voltages, train_range),
        FeatureStandardizer.fit(data.currents, train_range),
    )
