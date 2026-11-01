"""
Physics-informed synthetic sensor data generator for 220 kV GIS assets.

Generates multi-variate time series that mimic:
- Three-phase voltages & currents
- Current harmonics (THD)
- Phase imbalance
- Winding / ambient temperature
- SF6 gas pressure
- Vibration (proxy)

Normal operation + injected anomaly scenarios (SF6 leak, contact overheating,
partial discharge signature, CT saturation-like imbalance).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime, timedelta


# Nominal values (approximate for 220 kV GIS)
NOMINAL = {
    "v_rms": 220_000 / np.sqrt(3),          # phase-to-ground ~127 kV
    "i_rms": 800.0,                         # typical load current
    "freq": 50.0,
    "temp_winding": 65.0,                   # °C
    "temp_ambient": 32.0,
    "sf6_pressure": 6.5,                    # bar
    "vibration": 0.8,                       # mm/s RMS proxy
}


def _add_noise(x: np.ndarray, rel_std: float = 0.005) -> np.ndarray:
    return x + np.random.normal(0, rel_std * np.abs(x).mean() + 1e-6, size=x.shape)


def generate_normal_segment(
    n_samples: int,
    start_time: datetime,
    freq_hz: float = 1.0,          # 1 Hz sampling for demo (can be raised)
    load_factor: float = 0.85,
) -> pd.DataFrame:
    """Generate a healthy operating segment."""
    t = np.arange(n_samples) / freq_hz
    timestamps = [start_time + timedelta(seconds=float(s)) for s in t]

    # Slow load variation
    load = load_factor * (1.0 + 0.04 * np.sin(2 * np.pi * t / 3600))

    va = NOMINAL["v_rms"] * _add_noise(np.ones(n_samples), 0.002)
    vb = NOMINAL["v_rms"] * _add_noise(np.ones(n_samples), 0.002)
    vc = NOMINAL["v_rms"] * _add_noise(np.ones(n_samples), 0.002)

    ia = NOMINAL["i_rms"] * load * _add_noise(np.ones(n_samples), 0.01)
    ib = NOMINAL["i_rms"] * load * _add_noise(np.ones(n_samples), 0.01)
    ic = NOMINAL["i_rms"] * load * _add_noise(np.ones(n_samples), 0.01)

    # Small natural imbalance
    ib *= 0.995
    ic *= 1.005

    # Harmonics (low THD in healthy state)
    thd_i = 1.5 + 0.3 * np.random.randn(n_samples)
    thd_i = np.clip(thd_i, 0.5, 4.0)

    freq = NOMINAL["freq"] + 0.02 * np.random.randn(n_samples)

    temp_w = NOMINAL["temp_winding"] + 3 * load + 0.5 * np.random.randn(n_samples)
    temp_a = NOMINAL["temp_ambient"] + 0.3 * np.random.randn(n_samples)

    sf6 = NOMINAL["sf6_pressure"] + 0.02 * np.random.randn(n_samples)
    vib = NOMINAL["vibration"] * (0.9 + 0.2 * load) + 0.05 * np.random.randn(n_samples)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "va_kv": va / 1000,
        "vb_kv": vb / 1000,
        "vc_kv": vc / 1000,
        "ia_a": ia,
        "ib_a": ib,
        "ic_a": ic,
        "freq_hz": freq,
        "thd_i_pct": thd_i,
        "temp_winding_c": temp_w,
        "temp_ambient_c": temp_a,
        "sf6_pressure_bar": sf6,
        "vibration_mms": vib,
        "label": 0,                     # 0 = normal
        "anomaly_type": "normal",
    })
    return df


def inject_anomaly(
    df: pd.DataFrame,
    anomaly_type: str,
    start_idx: int,
    duration: int,
) -> pd.DataFrame:
    """Inject a realistic anomaly signature into a copy of the dataframe."""
    df = df.copy()
    end = min(start_idx + duration, len(df))
    sl = slice(start_idx, end)
    n = end - start_idx
    ramp = np.linspace(0, 1, n)

    if anomaly_type == "sf6_leak":
        # Gradual pressure drop + slight temperature change
        df.loc[df.index[sl], "sf6_pressure_bar"] -= ramp * 1.2
        df.loc[df.index[sl], "temp_winding_c"] += ramp * 2.0
        df.loc[df.index[sl], "label"] = 1
        df.loc[df.index[sl], "anomaly_type"] = "sf6_leak"

    elif anomaly_type == "contact_overheating":
        # Rising winding temperature + increasing vibration + mild imbalance
        df.loc[df.index[sl], "temp_winding_c"] += ramp * 25
        df.loc[df.index[sl], "vibration_mms"] += ramp * 1.8
        df.loc[df.index[sl], "ic_a"] *= (1 + ramp * 0.06)
        df.loc[df.index[sl], "thd_i_pct"] += ramp * 3.5
        df.loc[df.index[sl], "label"] = 1
        df.loc[df.index[sl], "anomaly_type"] = "contact_overheating"

    elif anomaly_type == "phase_imbalance":
        # Strong current imbalance (possible CT or connection issue)
        df.loc[df.index[sl], "ic_a"] *= (1 - ramp * 0.35)
        df.loc[df.index[sl], "thd_i_pct"] += ramp * 2.0
        df.loc[df.index[sl], "label"] = 1
        df.loc[df.index[sl], "anomaly_type"] = "phase_imbalance"

    elif anomaly_type == "partial_discharge":
        # Elevated high-frequency content proxy (THD) + small voltage dips
        df.loc[df.index[sl], "thd_i_pct"] += 4 + ramp * 6
        df.loc[df.index[sl], "va_kv"] -= ramp * 0.8
        df.loc[df.index[sl], "vibration_mms"] += ramp * 0.6
        df.loc[df.index[sl], "label"] = 1
        df.loc[df.index[sl], "anomaly_type"] = "partial_discharge"

    else:
        raise ValueError(f"Unknown anomaly_type: {anomaly_type}")

    return df


def generate_dataset(
    hours: float = 48,
    sample_rate_hz: float = 1.0,
    anomaly_scenarios: Optional[list] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a multi-hour dataset containing normal operation and several anomalies.
    """
    np.random.seed(seed)
    n_samples = int(hours * 3600 * sample_rate_hz)
    start = datetime(2026, 10, 1, 0, 0, 0)

    df = generate_normal_segment(n_samples, start, freq_hz=sample_rate_hz)

    if anomaly_scenarios is None:
        anomaly_scenarios = [
            ("sf6_leak", int(0.25 * n_samples), int(0.08 * n_samples)),
            ("contact_overheating", int(0.55 * n_samples), int(0.06 * n_samples)),
            ("phase_imbalance", int(0.75 * n_samples), int(0.05 * n_samples)),
            ("partial_discharge", int(0.88 * n_samples), int(0.04 * n_samples)),
        ]

    for atype, start_idx, duration in anomaly_scenarios:
        df = inject_anomaly(df, atype, start_idx, duration)

    return df.reset_index(drop=True)


def save_raw_dataset(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    # Also CSV for easy inspection
    df.to_csv(path.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[2] / "data" / "raw" / "gis_sensor_stream.parquet"
    print("Generating 48 h synthetic GIS sensor stream...")
    df = generate_dataset(hours=48, sample_rate_hz=1.0)
    save_raw_dataset(df, out)
    print(f"Saved {len(df):,} rows → {out}")
    print(df["anomaly_type"].value_counts())
