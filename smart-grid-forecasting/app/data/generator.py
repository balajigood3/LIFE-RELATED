"""
Synthetic industrial plant load + weather data generator.

Creates realistic multi-week demand profiles that include:
- Daily & weekly seasonality
- Production shift patterns
- Weather influence (temperature → cooling/heating load)
- Occasional maintenance / curtailment events
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional


def generate_weather(timestamps: pd.DatetimeIndex, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(timestamps)
    hour = timestamps.hour + timestamps.minute / 60.0
    day_of_year = timestamps.dayofyear

    # Seasonal baseline + daily cycle
    temp = (
        28
        + 8 * np.sin(2 * np.pi * (day_of_year - 80) / 365)   # seasonal
        + 6 * np.sin(2 * np.pi * (hour - 14) / 24)           # daily peak ~14:00
        + rng.normal(0, 1.5, n)
    )
    humidity = np.clip(
        65 - 0.4 * (temp - 28) + rng.normal(0, 5, n), 30, 95
    )
    return pd.DataFrame({
        "timestamp": timestamps,
        "temp_c": temp,
        "humidity_pct": humidity,
    })


def generate_industrial_load(
    start: datetime = datetime(2026, 1, 1),
    days: int = 60,
    freq: str = "h",          # hourly
    base_load_mw: float = 45.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate hourly industrial load (MW) with realistic patterns.
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start, periods=days * 24, freq=freq)
    n = len(timestamps)

    hour = timestamps.hour
    dayofweek = timestamps.dayofweek          # 0=Mon
    is_weekend = (dayofweek >= 5).astype(float)

    # Base + daily shape (higher during day shifts)
    daily_shape = (
        0.75
        + 0.25 * np.sin(2 * np.pi * (hour - 8) / 24)   # ramp up morning
        + 0.10 * ((hour >= 8) & (hour <= 20)).astype(float)
    )

    # Weekend reduction
    weekend_factor = 1.0 - 0.22 * is_weekend

    # Mild weekly production cycle
    weekly = 1.0 + 0.05 * np.sin(2 * np.pi * dayofweek / 7)

    # Weather influence (cooling load proxy)
    weather = generate_weather(timestamps, seed=seed)
    temp = weather["temp_c"].values
    cooling = np.clip((temp - 26) * 0.35, 0, None)   # MW extra when hot
    heating = np.clip((18 - temp) * 0.15, 0, None)

    # Random operational noise + occasional maintenance dips
    noise = rng.normal(0, 1.2, n)
    maintenance = np.zeros(n)
    # Inject 3-4 maintenance windows
    for _ in range(4):
        start_idx = rng.integers(0, n - 48)
        duration = rng.integers(8, 36)
        maintenance[start_idx:start_idx + duration] = -rng.uniform(8, 18)

    load = (
        base_load_mw
        * daily_shape
        * weekend_factor
        * weekly
        + cooling
        + heating
        + noise
        + maintenance
    )
    load = np.clip(load, base_load_mw * 0.35, None)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "load_mw": load,
        "temp_c": temp,
        "humidity_pct": weather["humidity_pct"].values,
        "hour": hour,
        "dayofweek": dayofweek,
        "is_weekend": is_weekend,
        "month": timestamps.month,
    })
    return df


def save_dataset(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    df.to_csv(path.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[2] / "data" / "raw" / "industrial_load.parquet"
    print("Generating 60 days of hourly industrial load + weather...")
    df = generate_industrial_load(days=60)
    save_dataset(df, out)
    print(f"Saved {len(df):,} rows → {out}")
    print(df[["load_mw", "temp_c"]].describe().round(2))
