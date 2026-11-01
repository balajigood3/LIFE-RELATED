"""Feature engineering for short-term industrial load forecasting."""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Tuple


FEATURE_COLS = [
    "hour",
    "dayofweek",
    "is_weekend",
    "month",
    "temp_c",
    "humidity_pct",
    "load_lag_1",
    "load_lag_2",
    "load_lag_3",
    "load_lag_24",
    "load_lag_48",
    "load_lag_168",          # 1 week
    "load_rolling_mean_24",
    "load_rolling_std_24",
    "load_rolling_mean_168",
    "temp_rolling_mean_24",
]


def add_lag_features(df: pd.DataFrame, target: str = "load_mw") -> pd.DataFrame:
    out = df.copy()
    out["load_lag_1"] = out[target].shift(1)
    out["load_lag_2"] = out[target].shift(2)
    out["load_lag_3"] = out[target].shift(3)
    out["load_lag_24"] = out[target].shift(24)
    out["load_lag_48"] = out[target].shift(48)
    out["load_lag_168"] = out[target].shift(168)

    out["load_rolling_mean_24"] = out[target].shift(1).rolling(24).mean()
    out["load_rolling_std_24"] = out[target].shift(1).rolling(24).std()
    out["load_rolling_mean_168"] = out[target].shift(1).rolling(168).mean()
    out["temp_rolling_mean_24"] = out["temp_c"].shift(1).rolling(24).mean()

    return out


def build_supervised(
    df: pd.DataFrame,
    horizon: int = 24,          # multi-horizon optional; we forecast step-ahead primarily
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create supervised learning table.
    For simplicity we train a model that predicts the next hour (horizon=1)
    and can be rolled forward for multi-step forecasts.
    """
    feat = add_lag_features(df)
    feat = feat.dropna().reset_index(drop=True)

    # Target is next-step load
    feat["target"] = feat["load_mw"].shift(-1)
    feat = feat.dropna().reset_index(drop=True)

    available = [c for c in FEATURE_COLS if c in feat.columns]
    return feat, available
