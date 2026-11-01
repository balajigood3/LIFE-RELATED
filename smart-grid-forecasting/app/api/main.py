"""
FastAPI inference service for industrial load forecasting.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.training.features import FEATURE_COLS

MODEL_PATH = ROOT / "data" / "models" / "best_model.joblib"
META_PATH = ROOT / "data" / "models" / "metadata.json"

app = FastAPI(
    title="Smart Grid Load Forecasting API",
    description="Industrial / microgrid short-term load forecast service",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ForecastRequest(BaseModel):
    """Minimal request: recent history + optional weather forecast."""
    recent_load_mw: List[float] = Field(
        ...,
        description="Recent hourly load values (most recent last). Prefer >= 168 points (1 week).",
        min_length=24,
    )
    recent_temp_c: Optional[List[float]] = Field(
        None,
        description="Recent hourly temperatures matching recent_load_mw length",
    )
    future_temp_c: Optional[List[float]] = Field(
        None,
        description="Forecast temperatures for the horizons to predict",
    )
    horizons: int = Field(24, ge=1, le=168, description="Number of hourly steps to forecast")
    start_time: Optional[datetime] = Field(
        None,
        description="Timestamp of the first forecast step (default = now + 1h)",
    )


class ForecastPoint(BaseModel):
    timestamp: datetime
    load_mw: float
    horizon: int


class ForecastResponse(BaseModel):
    model_name: str
    generated_at: datetime
    forecasts: List[ForecastPoint]
    metrics: Optional[Dict[str, float]] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str] = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_model_bundle: Optional[dict] = None


def get_model() -> dict:
    global _model_bundle
    if _model_bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Model not found. Run `python -m app.training.train` first.",
            )
        _model_bundle = joblib.load(MODEL_PATH)
    return _model_bundle


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    loaded = MODEL_PATH.exists()
    name = None
    if loaded:
        try:
            name = get_model().get("model_name")
        except Exception:
            loaded = False
    return HealthResponse(
        status="ok" if loaded else "model_missing",
        model_loaded=loaded,
        model_name=name,
    )


@app.get("/models")
def list_models():
    if not META_PATH.exists():
        return {"message": "No metadata found. Train models first."}
    import json
    with open(META_PATH) as f:
        meta = json.load(f)
    return meta


@app.post("/predict", response_model=ForecastResponse)
def predict(req: ForecastRequest):
    bundle = get_model()
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    model_name = bundle.get("model_name", "unknown")
    train_metrics = bundle.get("metrics")

    hist_load = list(req.recent_load_mw)
    n_hist = len(hist_load)

    # Temperatures: use provided or a constant mild value
    if req.recent_temp_c and len(req.recent_temp_c) == n_hist:
        hist_temp = list(req.recent_temp_c)
    else:
        hist_temp = [28.0] * n_hist

    future_temp = req.future_temp_c or [28.0] * req.horizons
    if len(future_temp) < req.horizons:
        future_temp = future_temp + [future_temp[-1]] * (req.horizons - len(future_temp))

    start = req.start_time or (datetime.utcnow() + timedelta(hours=1))
    start = start.replace(minute=0, second=0, microsecond=0)

    forecasts: List[ForecastPoint] = []
    load_window = hist_load.copy()
    temp_window = hist_temp.copy()

    for h in range(1, req.horizons + 1):
        ts = start + timedelta(hours=h - 1)
        hour = ts.hour
        dow = ts.weekday()
        is_we = 1.0 if dow >= 5 else 0.0
        month = ts.month
        temp = future_temp[h - 1]

        # Build feature vector from current window
        def lag(arr, k):
            return arr[-k] if len(arr) >= k else arr[0]

        def roll_mean(arr, w):
            return float(np.mean(arr[-w:])) if len(arr) >= w else float(np.mean(arr))

        def roll_std(arr, w):
            return float(np.std(arr[-w:])) if len(arr) >= w else 0.0

        feat = {
            "hour": hour,
            "dayofweek": dow,
            "is_weekend": is_we,
            "month": month,
            "temp_c": temp,
            "humidity_pct": 60.0,  # simplified
            "load_lag_1": lag(load_window, 1),
            "load_lag_2": lag(load_window, 2),
            "load_lag_3": lag(load_window, 3),
            "load_lag_24": lag(load_window, 24),
            "load_lag_48": lag(load_window, 48),
            "load_lag_168": lag(load_window, 168),
            "load_rolling_mean_24": roll_mean(load_window, 24),
            "load_rolling_std_24": roll_std(load_window, 24),
            "load_rolling_mean_168": roll_mean(load_window, 168),
            "temp_rolling_mean_24": roll_mean(temp_window, 24),
        }

        x = np.array([[feat[c] for c in feature_names]], dtype=np.float64)
        pred = float(model.predict(x)[0])
        pred = max(pred, 0.0)

        forecasts.append(ForecastPoint(timestamp=ts, load_mw=round(pred, 3), horizon=h))

        # Roll forward
        load_window.append(pred)
        temp_window.append(temp)

    return ForecastResponse(
        model_name=model_name,
        generated_at=datetime.utcnow(),
        forecasts=forecasts,
        metrics=train_metrics,
    )


@app.get("/")
def root():
    return {
        "service": "Smart Grid Load Forecasting API",
        "docs": "/docs",
        "health": "/health",
    }
