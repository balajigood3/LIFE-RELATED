"""
Explainable AI layer using SHAP for Isolation Forest.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import shap

from app.models.anomaly import IsolationForestDetector


def compute_shap_values(
    detector: IsolationForestDetector,
    X: np.ndarray,
    max_samples: int = 200,
) -> tuple[np.ndarray, List[str]]:
    """
    Compute SHAP values for the Isolation Forest.
    Returns (shap_values, feature_names)
    """
    if not detector.is_fitted:
        raise RuntimeError("Detector must be fitted first")

    # Subsample for speed
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    Xs = detector.scaler.transform(X_sample)

    # TreeExplainer works well with IsolationForest
    explainer = shap.TreeExplainer(detector.model)
    shap_vals = explainer.shap_values(Xs)

    return shap_vals, detector.feature_names


def top_contributing_features(
    shap_row: np.ndarray,
    feature_names: List[str],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Return the top-k features that pushed the sample toward anomaly."""
    # For Isolation Forest, higher absolute SHAP often indicates contribution
    # We look at features with positive contribution to the anomaly score direction.
    order = np.argsort(-np.abs(shap_row))[:top_k]
    results = []
    for i in order:
        results.append({
            "feature": feature_names[i],
            "shap_value": float(shap_row[i]),
            "abs_impact": float(np.abs(shap_row[i])),
        })
    return results


def explain_anomalies(
    detector: IsolationForestDetector,
    X: np.ndarray,
    preds: np.ndarray,
    timestamps: Optional[pd.Series] = None,
    top_k: int = 4,
    max_anomalies_to_explain: int = 15,
) -> pd.DataFrame:
    """
    Produce a human-readable explanation table for detected anomalies.
    """
    anomaly_idx = np.where(preds == 1)[0]
    if len(anomaly_idx) == 0:
        return pd.DataFrame()

    # Limit for performance
    anomaly_idx = anomaly_idx[:max_anomalies_to_explain]

    Xs = detector.scaler.transform(X[anomaly_idx])
    explainer = shap.TreeExplainer(detector.model)
    shap_vals = explainer.shap_values(Xs)

    rows = []
    for i, idx in enumerate(anomaly_idx):
        top = top_contributing_features(shap_vals[i], detector.feature_names, top_k=top_k)
        row = {
            "sample_index": int(idx),
            "timestamp": timestamps.iloc[idx] if timestamps is not None else None,
            "top_features": " | ".join(f"{t['feature']} ({t['shap_value']:+.3f})" for t in top),
            "primary_driver": top[0]["feature"] if top else "unknown",
        }
        rows.append(row)

    return pd.DataFrame(rows)
