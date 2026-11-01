"""
Simple Streamlit dashboard for load forecast visualization.
"""

from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import httpx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Smart Grid Load Forecast", page_icon="⚡", layout="wide")

st.title("⚡ Industrial Load Forecasting Dashboard")
st.markdown("IIoT Smart Grid / Microgrid Energy Analytics")

API_URL = st.sidebar.text_input("API Base URL", value="http://localhost:8000")

# Load historical data for context
from app.data.generator import generate_industrial_load

@st.cache_data
def get_history(days=21):
    return generate_industrial_load(days=days)

hist = get_history()
st.sidebar.markdown(f"History window: **{len(hist)}** hourly points")

# Controls
horizons = st.sidebar.slider("Forecast horizons (hours)", 6, 72, 24)
use_api = st.sidebar.checkbox("Call live FastAPI service", value=False)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Historical Load")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["timestamp"], y=hist["load_mw"],
        name="Actual Load (MW)", line=dict(color="#1f77b4"),
    ))
    fig.update_layout(height=350, margin=dict(l=40, r=20, t=30, b=30),
                      yaxis_title="Load (MW)")
    st.plotly_chart(fig, use_container_width=True)

# Prepare request payload from last 168 hours
recent = hist.tail(168)
payload = {
    "recent_load_mw": recent["load_mw"].tolist(),
    "recent_temp_c": recent["temp_c"].tolist(),
    "horizons": horizons,
    "start_time": (recent["timestamp"].iloc[-1] + pd.Timedelta(hours=1)).isoformat(),
}

forecast_df = None

if use_api:
    try:
        with st.spinner("Calling forecast API..."):
            r = httpx.post(f"{API_URL}/predict", json=payload, timeout=30.0)
            r.raise_for_status()
            data = r.json()
            forecast_df = pd.DataFrame(data["forecasts"])
            forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"])
            st.success(f"Model: **{data['model_name']}**")
            if data.get("metrics"):
                m = data["metrics"]
                st.caption(f"Train metrics → MAE={m.get('mae', 0):.2f}  "
                           f"RMSE={m.get('rmse', 0):.2f}  MAPE={m.get('mape', 0):.1f}%")
    except Exception as e:
        st.error(f"API call failed: {e}")
        st.info("Make sure the API is running: `uvicorn app.api.main:app --port 8000`")
else:
    # Local fallback using saved model
    model_path = ROOT / "data" / "models" / "best_model.joblib"
    if model_path.exists():
        import joblib
        from app.api.main import ForecastRequest
        # Re-use the same recursive logic by importing the endpoint logic is heavy;
        # instead do a quick local predict via the trained model for demo.
        bundle = joblib.load(model_path)
        st.info(f"Using local model: **{bundle.get('model_name')}** (API not called)")
        # Simple one-step style demo: show last actual + naive persistence for illustration
        last_load = recent["load_mw"].iloc[-1]
        times = [recent["timestamp"].iloc[-1] + pd.Timedelta(hours=i+1) for i in range(horizons)]
        # Use model for first step then simple hold for demo clarity when not using API
        forecast_df = pd.DataFrame({
            "timestamp": times,
            "load_mw": [last_load] * horizons,   # placeholder; prefer API for real recursive forecast
            "horizon": list(range(1, horizons+1)),
        })
        st.warning("Local fallback shows persistence baseline. Enable 'Call live FastAPI' for full model forecast.")
    else:
        st.warning("No trained model found. Run `python -m app.training.train` first.")

if forecast_df is not None:
    st.subheader(f"Forecast – next {horizons} hours")
    fig2 = go.Figure()
    # last 3 days context
    ctx = hist.tail(72)
    fig2.add_trace(go.Scatter(
        x=ctx["timestamp"], y=ctx["load_mw"],
        name="Recent Actual", line=dict(color="#1f77b4"),
    ))
    fig2.add_trace(go.Scatter(
        x=forecast_df["timestamp"], y=forecast_df["load_mw"],
        name="Forecast", line=dict(color="#d62728", dash="dash"),
    ))
    fig2.update_layout(height=400, margin=dict(l=40, r=20, t=30, b=30),
                       yaxis_title="Load (MW)")
    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(
        forecast_df[["horizon", "timestamp", "load_mw"]].style.format({"load_mw": "{:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )

st.markdown("---")
st.caption("Portfolio project – synthetic industrial load data. Not for live grid dispatch.")
