"""
Feature engineering for power-quality & thermal anomaly detection.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List


FEATURE_COLUMNS = [
    "v_rms_mean",
    "v_imbalance_pct",
    "i_rms_mean",
    "i_imbalance_pct",
    "thd_i_pct",
    "freq_hz",
    "temp_winding_c",
    "temp_rise_c",               # winding - ambient
    "sf6_pressure_bar",
    "vibration_mms",
    "i_rms_rolling_std",
    "temp_winding_rolling_mean",
    "sf6_pressure_slope",
]


def compute_imbalance(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Max deviation from average, in percent."""
    mean = (a + b + c) / 3.0
    max_dev = np.maximum.reduce([np.abs(a - mean), np.abs(b - mean), np.abs(c - mean)])
    return np.where(mean > 1e-6, (max_dev / mean) * 100.0, 0.0)


def build_feature_frame(
    df: pd.DataFrame,
    window: int = 30,           # samples (seconds if 1 Hz)
) -> pd.DataFrame:
    """
    Transform raw sensor stream into a feature matrix suitable for ML.
    Uses centered rolling statistics for smoothness.
    """
    out = pd.DataFrame(index=df.index)
    out["timestamp"] = df["timestamp"]

    # Basic aggregates
    out["v_rms_mean"] = (df["va_kv"] + df["vb_kv"] + df["vc_kv"]) / 3.0
    out["i_rms_mean"] = (df["ia_a"] + df["ib_a"] + df["ic_a"]) / 3.0

    out["v_imbalance_pct"] = compute_imbalance(
        df["va_kv"].values, df["vb_kv"].values, df["vc_kv"].values
    )
    out["i_imbalance_pct"] = compute_imbalance(
        df["ia_a"].values, df["ib_a"].values, df["ic_a"].values
    )

    out["thd_i_pct"] = df["thd_i_pct"]
    out["freq_hz"] = df["freq_hz"]
    out["temp_winding_c"] = df["temp_winding_c"]
    out["temp_rise_c"] = df["temp_winding_c"] - df["temp_ambient_c"]
    out["sf6_pressure_bar"] = df["sf6_pressure_bar"]
    out["vibration_mms"] = df["vibration_mms"]

    # Rolling features
    out["i_rms_rolling_std"] = out["i_rms_mean"].rolling(window, min_periods=5).std().fillna(0)
    out["temp_winding_rolling_mean"] = (
        out["temp_winding_c"].rolling(window, min_periods=5).mean().bfill()
    )

    # Simple slope of SF6 pressure (bar per window)
    out["sf6_pressure_slope"] = (
        out["sf6_pressure_bar"]
        .rolling(window, min_periods=5)
        .apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 2 else 0, raw=True)
        .fillna(0)
    )

    # Labels (for evaluation only – models are unsupervised)
    if "label" in df.columns:
        out["label"] = df["label"]
    if "anomaly_type" in df.columns:
        out["anomaly_type"] = df["anomaly_type"]

    # Drop initial NaNs from rolling
    out = out.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    return out


def get_feature_matrix(feat_df: pd.DataFrame) -> tuple[np.ndarray, List[str]]:
    """Return X matrix and feature names."""
    X = feat_df[FEATURE_COLUMNS].values.astype(np.float64)
    return X, FEATURE_COLUMNS
