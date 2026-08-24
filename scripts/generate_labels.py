"""
generate_labels.py

Builds the IEEE European LV feeder circuit in OpenDSS directly from the
project's CSVs (Source, Transformer, LineCodes, Lines, Loads, LoadShapes),
runs a 1-minute-resolution daily power flow, and exports per-timestep
bus voltages + line currents to Parquet. These become the ground-truth
labels for the GNN (voltages/currents can't be derived from topology
alone -- they require an actual power flow solve).

Requires: opendssdirect.py, pandas, pyarrow
    pip install opendssdirect.py pandas pyarrow

Assumes the load profile CSVs referenced in LoadShapes.csv ("File" column,
e.g. Load_profile_1.csv) live alongside the other CSVs in --data-dir, each
a single column of 1440 per-minute kW multipliers (OpenDSS "actual" loadshape
format, matching useactual=TRUE in LoadShapes.csv).

Output: two Parquet files under --out-dir:
    bus_voltages.parquet   -- rows=timestep, cols=node names, values=Vpu
    line_currents.parquet  -- rows=timestep, cols=line names, values=Amax (per phase max)
"""

import configparser
from pathlib import Path

import numpy as np
import pandas as pd


def parse_source_ini(path: Path) -> dict:
    """Source.csv is a small INI-style block: [Source]\nVoltage=11 kV\n..."""
    cp = configparser.ConfigParser()
    # utf-8-sig accepts files saved with a UTF-8 BOM as well as ordinary UTF-8.
    with path.open(encoding="utf-8-sig") as source_file:
        cp.read_file(source_file, source=str(path))
    if not cp.sections():
        raise ValueError(
            f"{path} must contain an INI section such as [Source]; "
            "check that --data-dir points to data/European_LV_CSV."
        )
    section = cp[cp.sections()[0]]
    # strip units like "11 kV" / "3000 A" down to the leading number
    def num(key, default=None):
        raw = section.get(key, fallback=None)
        if raw is None:
            return default
        return float(raw.strip().split()[0])
    return {
        "voltage_kv": num("Voltage"),
        "pu": num("pu", 1.0),
        "isc3": num("ISC3"),
        "isc1": num("ISC1"),
    }

import pandas as pd
from pathlib import Path


def read_profile(path: Path) -> pd.Series:
    """Robustly read a single Load_profile_N.csv (handles tabs/spaces/commas,
    and a header row like 'time, mult')."""
    prof = pd.read_csv(path, sep=None, engine="python")
    prof.columns = [c.strip().lower() for c in prof.columns]
    return prof["mult"].reset_index(drop=True)


def build_load_profiles(data_dir: Path, profiles_subdir: str = "Load_Profiles") -> pd.DataFrame:
    """
    Consolidates the raw per-load Load_profile_N.csv files into one wide
    DataFrame: rows = timestep, columns = Load names, values = kW at that
    minute. This is what hetero_gnn_dataset.py reads as the per-timestep dynamic
    node features for Load nodes.

    Requires Loads.csv (Name, Yearly, kW, ...) and LoadShapes.csv
    (Name, npts, File, useactual) in data_dir, and the raw profile files
    under data_dir/profiles_subdir.
    """
    df_loads = pd.read_csv(data_dir / "Loads.csv")
    df_loads.columns = [c.strip() for c in df_loads.columns]
    df_shapes = pd.read_csv(data_dir / "LoadShapes.csv")
    df_shapes.columns = [c.strip() for c in df_shapes.columns]

    shape_file = dict(zip(df_shapes["Name"], df_shapes["File"]))
    shape_npts = dict(zip(df_shapes["Name"], df_shapes["npts"]))
    shape_useactual = dict(zip(df_shapes["Name"], df_shapes["useactual"]))

    profiles_dir = data_dir / profiles_subdir

    series = {}
    for _, row in df_loads.iterrows():
        load_name = row["Name"]
        shape_name = row["Yearly"]

        if shape_name not in shape_file:
            print(f"WARNING: {load_name} references unknown shape {shape_name}, skipping")
            continue

        mult = read_profile(profiles_dir / shape_file[shape_name])
        expected = int(shape_npts[shape_name])
        if len(mult) != expected:
            print(f"WARNING: {load_name} ({shape_file[shape_name]}) has {len(mult)} points, "
                  f"expected {expected}")

        # useactual=TRUE -> the shape already stores actual kW, don't rescale.
        # useactual=FALSE -> the shape is a per-unit multiplier, scale by base kW.
        useactual = str(shape_useactual[shape_name]).strip().upper()
        kw_series = mult if useactual in ("TRUE", "1") else mult * float(row["kW"])

        series[load_name] = kw_series.reset_index(drop=True)

    df_out = pd.DataFrame(series)
    df_out.index.name = "timestep"
    return df_out


def build_circuit(dss, data_dir: Path):
    """Issue OpenDSS text commands built directly from the project CSVs."""
    src = parse_source_ini(data_dir / "Source.csv")

    df_buscoords = pd.read_csv(data_dir / "Buscoords.csv")
    df_buscoords.columns = [c.strip() for c in df_buscoords.columns]
    df_linecodes = pd.read_csv(data_dir / "LineCodes.csv")
    df_linecodes.columns = [c.strip() for c in df_linecodes.columns]
    df_lines = pd.read_csv(data_dir / "Lines.csv")
    df_lines.columns = [c.strip() for c in df_lines.columns]
    df_transformer = pd.read_csv(data_dir / "Transformer.csv")
    df_transformer.columns = [c.strip() for c in df_transformer.columns]
    df_loads = pd.read_csv(data_dir / "Loads.csv")
    df_loads.columns = [c.strip() for c in df_loads.columns]
    df_loadshapes = pd.read_csv(data_dir / "LoadShapes.csv")
    df_loadshapes.columns = [c.strip() for c in df_loadshapes.columns]

    dss.Text.Command("Clear")
    dss.Text.Command(
        f"New Circuit.EuropeanLV basekv={src['voltage_kv']} pu={src['pu']} "
        f"phases=3 bus1=SourceBus isc3={src['isc3']} isc1={src['isc1']}"
    )

    # LineCodes
    for _, r in df_linecodes.iterrows():
        dss.Text.Command(
            f"New LineCode.{r['Name']} nphases={int(r['nphases'])} "
            f"units={r['Units']} R1={r['R1']} X1={r['X1']} "
            f"R0={r['R0']} X0={r['X0']} C1={r['C1']} C0={r['C0']}"
        )

    # Transformer (row 0 is the single substation transformer)
    t = df_transformer.iloc[0]
    dss.Text.Command(
        f"New Transformer.{t['Name']} phases={int(t['phases'])} windings=2 "
        f"buses=[{t['bus1']}.1.2.3, {t['bus2']}.1.2.3] "
        f"conns=[{t['Conn_pri']}, {t['Conn_sec']}] "
        f"kvs=[{t['kV_pri']}, {t['kV_sec']}] kvas=[{float(t['MVA'])*1000}, {float(t['MVA'])*1000}] "
        f"xhl={t['%XHL']} %r={t['% resistance']}"
    )

    # Lines
    for _, r in df_lines.iterrows():
        nphases = len(r["Phases"])
        dss.Text.Command(
            f"New Line.{r['Name']} bus1={r['Bus1']} bus2={r['Bus2']} "
            f"phases={nphases} linecode={r['LineCode']} "
            f"length={r['Length']} units={r['Units']}"
        )

    # LoadShapes -- point directly at the per-minute profile CSVs

    for _, r in df_loadshapes.iterrows():
        profile_path = (data_dir / "Load_Profiles" / r["File"]).resolve()
        prof = pd.read_csv(profile_path)
        prof.columns = [c.strip().lower() for c in prof.columns]
        mult = prof["mult"].values
        if len(mult) != int(r["npts"]):
            print(f"WARNING: {r['Name']} has {len(mult)} points, expected {r['npts']}")
        dss.Text.Command(f"New Loadshape.{r['Name']} npts={int(r['npts'])} minterval={r['minterval']}")
        dss.LoadShape.Name(r["Name"])
        dss.LoadShape.PMult(mult.tolist())
        dss.LoadShape.UseActual(bool(r["useactual"]))

    # Loads
    for _, r in df_loads.iterrows():
        bus_phase_suffix = ".".join(_phase_nums(r["phases"]))
        dss.Text.Command(
            f"New Load.{r['Name']} bus1={r['Bus']}.{bus_phase_suffix} "
            f"phases={int(r['numPhases'])} conn={r['Connection']} "
            f"kV={r['kV']} kW={r['kW']} PF={r['PF']} model={int(r['Model'])} "
            f"daily={r['Yearly']}"
        )

    dss.Text.Command("Set VoltageBases=[11, 0.416, 0.23]")
    dss.Text.Command("CalcVoltageBases")


def _phase_nums(phase_letters: str):
    """'A' -> ['1'], 'AB' -> ['1','2'], 'ABC' -> ['1','2','3']"""
    mapping = {"A": "1", "B": "2", "C": "3"}
    return [mapping[c] for c in phase_letters.strip().upper()]


def run_daily_simulation(dss, n_steps: int, step_minutes: int = 1):
    dss.Text.Command("Set Mode=Daily")
    dss.Text.Command(f"Set stepsize={step_minutes}m")
    dss.Text.Command(f"Set number=1")  # step one interval at a time, manually
    dss.Solution.Number(1)

    node_names = None
    line_names = [n for n in dss.Lines.AllNames()]

    v_rows, i_rows = [], []
    for t in range(n_steps):
        dss.Solution.Solve()
        if not dss.Solution.Converged():
            print(f"WARNING: timestep {t} did not converge")

        if node_names is None:
            node_names = list(dss.Circuit.AllNodeNames())
        v_rows.append(list(dss.Circuit.AllBusMagPu()))

        currents = []
        for ln in line_names:
            dss.Circuit.SetActiveElement(f"Line.{ln}")
            mags = dss.CktElement.CurrentsMagAng()[0::2]  # every other value is magnitude
            currents.append(max(mags) if mags else np.nan)
        i_rows.append(currents)

    df_v = pd.DataFrame(v_rows, columns=node_names)
    df_i = pd.DataFrame(i_rows, columns=line_names)
    return df_v, df_i


def main():
    import opendssdirect as dss

    # Resolve from this file so the script behaves the same from any cwd.
    PROJECT_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_DIR / "data" / "European_LV_CSV"
    OUT_DIR = PROJECT_DIR / "data" / "Computed"
    N_STEPS = 1440

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_circuit(dss, DATA_DIR)

    df_lp = build_load_profiles(DATA_DIR)
    df_v, df_i = run_daily_simulation(dss, n_steps=N_STEPS)

    df_lp.to_csv(OUT_DIR / "load_profiles.csv")
    df_v.to_csv(OUT_DIR / "bus_voltages.csv")
    df_i.to_csv(OUT_DIR / "line_currents.csv")
    print(f"Wrote {len(df_v)} timesteps -> {OUT_DIR}")


if __name__ == "__main__":
    main()
