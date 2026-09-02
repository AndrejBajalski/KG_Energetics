"""Pure LSTM baseline for feeder CSV forecasting (no graph or Neo4j)."""

from __future__ import annotations

import torch
import torch.nn as nn


class CSVForecastLSTM(nn.Module):
    """Pure LSTM residual forecaster over aligned CSV observations.

    Loads, voltages, and currents are projected separately before fusion so
    the small load vector is not overwhelmed by the much wider target vectors.
    One shared horizon-conditioned output mapping replaces the previous giant
    independent output layer. Predictions are residual corrections to a
    training-fitted mean-reversion baseline built from the full configured
    input-window trend.
    """

    def __init__(
        self,
        load_input_dim: int,
        num_voltage_targets: int,
        num_current_targets: int,
        hidden: int = 192,
        horizon: int = 15,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        load_embedding: int = 64,
        voltage_embedding: int = 96,
        current_embedding: int = 96,
        voltage_basis: torch.Tensor | None = None,
        current_basis: torch.Tensor | None = None,
        voltage_trend_coefficients: torch.Tensor | None = None,
        current_trend_coefficients: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if load_input_dim < 1 or num_voltage_targets < 1 or num_current_targets < 1:
            raise ValueError("all input and output dimensions must be positive")

        self.load_input_dim = load_input_dim
        self.num_voltage_targets = num_voltage_targets
        self.num_current_targets = num_current_targets
        self.hidden = hidden
        self.horizon = horizon
        self.lstm_layers = lstm_layers
        self.dropout_rate = dropout
        self.load_embedding = load_embedding
        self.voltage_embedding = voltage_embedding
        self.current_embedding = current_embedding
        if voltage_basis is None or current_basis is None:
            raise ValueError("training-fitted voltage and current bases are required")
        if voltage_basis.size(0) != num_voltage_targets:
            raise ValueError("voltage basis has the wrong target dimension")
        if current_basis.size(0) != num_current_targets:
            raise ValueError("current basis has the wrong target dimension")
        self.voltage_latent_dim = voltage_basis.size(1)
        self.current_latent_dim = current_basis.size(1)
        self.register_buffer("voltage_basis", voltage_basis.detach().clone())
        self.register_buffer("current_basis", current_basis.detach().clone())
        voltage_trend_coefficients = (
            torch.zeros(horizon, num_voltage_targets)
            if voltage_trend_coefficients is None
            else voltage_trend_coefficients
        )
        current_trend_coefficients = (
            torch.zeros(horizon, num_current_targets)
            if current_trend_coefficients is None
            else current_trend_coefficients
        )
        if voltage_trend_coefficients.shape != (horizon, num_voltage_targets):
            raise ValueError("voltage trend coefficients have the wrong shape")
        if current_trend_coefficients.shape != (horizon, num_current_targets):
            raise ValueError("current trend coefficients have the wrong shape")
        self.register_buffer(
            "voltage_trend_coefficients", voltage_trend_coefficients.detach().clone()
        )
        self.register_buffer(
            "current_trend_coefficients", current_trend_coefficients.detach().clone()
        )

        # Each encoder receives both the standardized level and one-step
        # difference, making short-term trend information explicit.
        self.load_encoder = nn.Sequential(
            nn.Linear(2 * load_input_dim, load_embedding),
            nn.GELU(),
        )
        self.voltage_encoder = nn.Sequential(
            nn.Linear(2 * self.voltage_latent_dim, voltage_embedding),
            nn.GELU(),
        )
        self.current_encoder = nn.Sequential(
            nn.Linear(2 * self.current_latent_dim, current_embedding),
            nn.GELU(),
        )
        fused_dim = load_embedding + voltage_embedding + current_embedding
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.output_norm = nn.LayerNorm(hidden)
        self.horizon_embedding = nn.Parameter(torch.zeros(horizon, hidden))
        nn.init.normal_(self.horizon_embedding, mean=0.0, std=0.02)
        self.forecast_block = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.voltage_delta_head = nn.Linear(hidden, self.voltage_latent_dim)
        self.current_delta_head = nn.Linear(hidden, self.current_latent_dim)

        # Start at exact persistence. Training only has to learn corrections.
        nn.init.zeros_(self.voltage_delta_head.weight)
        nn.init.zeros_(self.voltage_delta_head.bias)
        nn.init.zeros_(self.current_delta_head.weight)
        nn.init.zeros_(self.current_delta_head.bias)

    @staticmethod
    def _levels_and_differences(history: torch.Tensor) -> torch.Tensor:
        differences = torch.zeros_like(history)
        differences[:, 1:, :] = history[:, 1:, :] - history[:, :-1, :]
        return torch.cat([history, differences], dim=-1)

    def trend_baseline(
        self,
        voltage_history: torch.Tensor,
        current_history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Training-fitted mean-reversion baseline from the input-window trend."""

        trend_steps = voltage_history.size(1) - 1
        voltage_slope = (
            voltage_history[:, -1, :] - voltage_history[:, 0, :]
        ) / trend_steps
        current_slope = (
            current_history[:, -1, :] - current_history[:, 0, :]
        ) / trend_steps
        voltage_baseline = (
            voltage_history[:, -1, :].unsqueeze(1)
            + voltage_slope.unsqueeze(1)
            * self.voltage_trend_coefficients.unsqueeze(0)
        )
        current_baseline = (
            current_history[:, -1, :].unsqueeze(1)
            + current_slope.unsqueeze(1)
            * self.current_trend_coefficients.unsqueeze(0)
        )
        return voltage_baseline, current_baseline

    def forward(
        self,
        load_history: torch.Tensor,
        voltage_history: torch.Tensor,
        current_history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        All histories have shape (batch, window, feature_count).

        Returns:
            voltage prediction: (batch, horizon, number_of_buses)
            current prediction: (batch, horizon, number_of_lines)
        """

        encoded_load = self.load_encoder(self._levels_and_differences(load_history))
        voltage_latent_history = voltage_history @ self.voltage_basis
        current_latent_history = current_history @ self.current_basis
        encoded_voltage = self.voltage_encoder(
            self._levels_and_differences(voltage_latent_history)
        )
        encoded_current = self.current_encoder(
            self._levels_and_differences(current_latent_history)
        )
        fused = self.fusion(torch.cat([encoded_load, encoded_voltage, encoded_current], dim=-1))
        sequence, _ = self.lstm(fused)
        last_hidden = self.output_norm(sequence[:, -1, :])
        future_hidden = self.forecast_block(
            last_hidden.unsqueeze(1) + self.horizon_embedding.unsqueeze(0)
        )

        voltage_delta = self.voltage_delta_head(future_hidden) @ self.voltage_basis.T
        current_delta = self.current_delta_head(future_hidden) @ self.current_basis.T
        voltage_baseline, current_baseline = self.trend_baseline(
            voltage_history, current_history
        )
        voltage = voltage_baseline + voltage_delta
        current = current_baseline + current_delta
        return voltage, current

    def configuration(self) -> dict[str, int | float]:
        return {
            "load_input_dim": self.load_input_dim,
            "num_voltage_targets": self.num_voltage_targets,
            "num_current_targets": self.num_current_targets,
            "hidden": self.hidden,
            "horizon": self.horizon,
            "lstm_layers": self.lstm_layers,
            "dropout": self.dropout_rate,
            "load_embedding": self.load_embedding,
            "voltage_embedding": self.voltage_embedding,
            "current_embedding": self.current_embedding,
            "voltage_latent_dim": self.voltage_latent_dim,
            "current_latent_dim": self.current_latent_dim,
            "residual_from_last_observation": 1,
            "training_fitted_trend_baseline": 1,
        }


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
