"""
computed_properties.py

Computes descriptive statistics (mean and standard deviation) for the two
OpenDSS-derived label sets used by the GNN pipeline:

    * Line current   (data/Computed/line_currents.csv)   -> per-line + global
    * Bus voltage    (data/Computed/bus_voltages.csv)    -> per-bus  + global

Bus voltages are reported per-*phase* by OpenDSS (columns like "1.1", "1.2",
"1.3"). They are collapsed to one value per physical Bus by taking the MINIMUM
across the present phases, identical to `_collapse_phase_columns_to_bus` in
hetero_gnn_dataset.py, so the per-bus statistics here line up with what the model sees.

Statistics are computed over the FULL dataset (all timesteps).

Standard deviation uses the POPULATION estimator (ddof=0), matching the
normalization in hetero_gnn_dataset.py (`std(..., unbiased=False)`), so these numbers
are directly comparable to the model's normalization statistics.

Results are printed to stdout (no files written), in fixed-point notation:
    * per-line current mean/std
    * per-bus voltage mean/std
    * global line-current and bus-voltage min/max/mean/variance (over all
      lines/buses and all timesteps)
    * averaged per-entity mean and variance (mean over lines/buses of each
      entity's own mean / own variance; the averaged variance is the
      trivial-baseline MSE the model's per-line/per-bus MSE is judged against)

Requires: pandas, numpy
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
COMPUTED_DIR = BASE_DIR / "data" / "Computed"

LINE_CURRENTS_PATH = COMPUTED_DIR / "line_currents.csv"
BUS_VOLTAGES_PATH = COMPUTED_DIR / "bus_voltages.csv"

# Population standard deviation (divide by N), to match the model's
# std(..., unbiased=False) normalization rather than pandas' default (N-1).
DDOF = 0


def collapse_phase_columns_to_bus(df):
    """Collapse per bus-phase columns ("1.1", "1.2", ...) to one column per
    Bus by taking the minimum across the present phases.

    Mirrors FeederHeteroSnapshotDataset._collapse_phase_columns_to_bus so the
    per-bus statistics computed here match the model's per-bus targets.
    """
    bus_of = {col: col.split(".")[0] for col in df.columns}
    out = {}
    for bus_name in sorted(set(bus_of.values()), key=lambda b: (len(b), b)):
        cols = [c for c, b in bus_of.items() if b == bus_name]
        out[bus_name] = df[cols].min(axis=1)
    return pd.DataFrame(out)


def per_column_stats(df):
    """Per-column (per-entity) mean and population std, computed over rows
    (timesteps). NaNs are skipped per column."""
    stats = pd.DataFrame({
        "mean": df.mean(axis=0, skipna=True),
        "std": df.std(axis=0, skipna=True, ddof=DDOF),
    })
    stats.index.name = df.columns.name or "name"
    return stats


def global_stats(df):
    """Min, max, mean and population variance over every value in the frame
    (all entities, all timesteps), NaN-aware."""
    values = df.to_numpy(dtype="float64").ravel()
    return {
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
        "mean": float(np.nanmean(values)),
        "variance": float(np.nanvar(values, ddof=DDOF)),
    }


def main():
    if not LINE_CURRENTS_PATH.exists():
        raise FileNotFoundError(f"Missing line currents file: {LINE_CURRENTS_PATH}")
    if not BUS_VOLTAGES_PATH.exists():
        raise FileNotFoundError(f"Missing bus voltages file: {BUS_VOLTAGES_PATH}")

    # ---- Line current: per-line + global ----
    line_currents = pd.read_csv(LINE_CURRENTS_PATH, index_col=0)
    line_currents.columns.name = "line"
    line_stats = per_column_stats(line_currents)
    line_global = global_stats(line_currents)

    # ---- Bus voltage: collapse phases -> per-bus + global ----
    raw_voltages = pd.read_csv(BUS_VOLTAGES_PATH, index_col=0)
    bus_voltages = collapse_phase_columns_to_bus(raw_voltages)
    bus_voltages.columns.name = "bus"
    bus_stats = per_column_stats(bus_voltages)
    bus_global = global_stats(bus_voltages)

    global_df = pd.DataFrame(
        [line_global, bus_global],
        index=pd.Index(["line_current", "bus_voltage"], name="quantity"),
        columns=["min", "max", "mean", "variance"],
    )

    # Averaged per-entity statistics: the mean of each line's/bus's own mean,
    # and the mean of each line's/bus's own variance (= std^2). The averaged
    # per-entity variance is the trivial-baseline MSE — the error you'd get by
    # predicting each line's/bus's own mean at every timestep — so it's the
    # right number to compare the model's per-line/per-bus MSE against.
    avg_df = pd.DataFrame(
        {
            "avg_per_entity_mean": [
                line_stats["mean"].mean(),
                bus_stats["mean"].mean(),
            ],
            "avg_per_entity_variance": [
                (line_stats["std"] ** 2).mean(),
                (bus_stats["std"] ** 2).mean(),
            ],
        },
        index=pd.Index(["line_current", "bus_voltage"], name="quantity"),
    )

    # ---- Print results (no files written) ----
    # Print every row rather than letting pandas truncate long frames, and use
    # plain fixed-point formatting instead of scientific notation.
    with pd.option_context("display.max_rows", None, "display.width", None,
                           "display.float_format", "{:.6f}".format):
        print(f"=== Per-line current stats "
              f"({len(line_stats)} lines over {len(line_currents)} timesteps) ===")
        print(line_stats.to_string())

        print(f"\n=== Per-bus voltage stats "
              f"({len(bus_stats)} buses, {raw_voltages.shape[1]} phase columns collapsed, "
              f"over {len(bus_voltages)} timesteps) ===")
        print(bus_stats.to_string())

        print("\n=== Global stats ===")
        print(global_df.to_string())

        print("\n=== Averaged per-entity stats "
              "(avg_per_entity_variance = trivial-baseline MSE) ===")
        print(avg_df.to_string())


if __name__ == "__main__":
    main()