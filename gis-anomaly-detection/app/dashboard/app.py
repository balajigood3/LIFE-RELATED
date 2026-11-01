"""
Streamlit dashboard – Real-Time Asset Health & Anomaly Explainability
for High-Voltage GIS Substations.
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.pipeline.features import build_feature_frame, get_feature_matrix, FEATURE_COLUMNS
from app.models.anomaly import IsolationForestDetector
from app.models.explain import explain_anomalies, compute_shap_values
from app.data.generator import generate_dataset

st.set_page_config(
    page_title="GIS Anomaly Detection | Asset Health",
    page_icon="⚡",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_PATH = ROOT / "data" / "raw" / "gis_sensor_stream.parquet"
FEAT_PATH = ROOT / "data" / "processed" / "features.parquet"
IF_MODEL_PATH = ROOT / "data" / "models" / "isolation_forest.joblib"
METRICS_PATH = ROOT / "data" / "models" / "metrics.json"


@st.cache_data
def load_or_generate_data():
    if RAW_PATH.exists():
        return pd.read_parquet(RAW_PATH)
    df = generate_dataset(hours=24, sample_rate_hz=1.0)  # shorter for dashboard speed
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_PATH, index=False)
    return df


@st.cache_resource
def load_model():
    if IF_MODEL_PATH.exists():
        return IsolationForestDetector.load(IF_MODEL_PATH)
    # Train on the fly if missing
    from app.pipeline.train import run_training
    run_training(use_lstm=False)
    return IsolationForestDetector.load(IF_MODEL_PATH)


def compute_health_index(scores: np.ndarray) -> np.ndarray:
    """Map anomaly score (0–1) to Asset Health Index (100–0)."""
    return np.clip(100 * (1 - scores), 0, 100)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚡ GIS Asset Health")
    st.markdown("**220 kV GIS – Anomaly Detection**")
    st.markdown("---")
    st.markdown("### Model")
    st.markdown("- Isolation Forest (unsupervised)")
    st.markdown("- SHAP explainability")
    st.markdown("---")
    show_explanations = st.checkbox("Show SHAP explanations", value=True)
    st.caption("Synthetic data for portfolio demonstration only.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.header("Real-Time Power Quality & Anomaly Detection")
st.markdown("High-Voltage GIS Switchgear & Transformer Health Monitoring")

with st.spinner("Loading data & model..."):
    raw_df = load_or_generate_data()
    detector = load_model()
    feat_df = build_feature_frame(raw_df, window=30)
    X, feature_names = get_feature_matrix(feat_df)
    preds = detector.predict(X)
    scores = detector.anomaly_score(X)
    health = compute_health_index(scores)

    feat_df = feat_df.copy()
    feat_df["anomaly"] = preds
    feat_df["anomaly_score"] = scores
    feat_df["health_index"] = health

# KPI row
n_anom = int(preds.sum())
avg_health = float(health.mean())
min_health = float(health.min())
current_health = float(health[-1])

k1, k2, k3, k4 = st.columns(4)
k1.metric("Current Asset Health Index", f"{current_health:.1f}", 
          delta=f"{current_health - avg_health:+.1f} vs avg")
k2.metric("Anomalies Detected", n_anom)
k3.metric("Average Health (period)", f"{avg_health:.1f}")
k4.metric("Lowest Health Observed", f"{min_health:.1f}")

st.markdown("---")

# Time series overview
st.subheader("Sensor Trends & Anomaly Flags")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Electrical", "Thermal / SF6", "Health Index"])

with tab1:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("Current RMS (A)", "SF6 Pressure (bar)", "Anomaly Score"))
    fig.add_trace(go.Scatter(x=feat_df["timestamp"], y=feat_df["i_rms_mean"],
                             name="I rms", line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=feat_df["timestamp"], y=feat_df["sf6_pressure_bar"],
                             name="SF6", line=dict(color="#2ca02c")), row=2, col=1)
    fig.add_trace(go.Scatter(x=feat_df["timestamp"], y=feat_df["anomaly_score"],
                             name="Score", line=dict(color="#d62728")), row=3, col=1)
    # Mark anomalies
    anom = feat_df[feat_df["anomaly"] == 1]
    if len(anom):
        fig.add_trace(go.Scatter(x=anom["timestamp"], y=anom["anomaly_score"],
                                 mode="markers", name="Anomaly",
                                 marker=dict(color="red", size=8, symbol="x")), row=3, col=1)
    fig.update_layout(height=650, showlegend=True, margin=dict(l=40, r=20, t=40, b=30))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                         subplot_titles=("Phase Current Imbalance %", "THD Current %"))
    fig2.add_trace(go.Scatter(x=feat_df["timestamp"], y=feat_df["i_imbalance_pct"],
                              name="Imbalance", line=dict(color="#ff7f0e")), row=1, col=1)
    fig2.add_trace(go.Scatter(x=feat_df["timestamp"], y=feat_df["thd_i_pct"],
                              name="THD", line=dict(color="#9467bd")), row=2, col=1)
    fig2.update_layout(height=500, margin=dict(l=40, r=20, t=40, b=30))
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                         subplot_titles=("Winding Temperature °C", "SF6 Pressure bar"))
    fig3.add_trace(go.Scatter(x=feat_df["timestamp"], y=feat_df["temp_winding_c"],
                              name="Temp", line=dict(color="#d62728")), row=1, col=1)
    fig3.add_trace(go.Scatter(x=feat_df["timestamp"], y=feat_df["sf6_pressure_bar"],
                              name="SF6", line=dict(color="#2ca02c")), row=2, col=1)
    fig3.update_layout(height=500, margin=dict(l=40, r=20, t=40, b=30))
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=feat_df["timestamp"], y=feat_df["health_index"],
        fill="tozeroy", name="Health Index",
        line=dict(color="#1f77b4"),
    ))
    fig4.add_hline(y=70, line_dash="dash", line_color="orange", annotation_text="Watch")
    fig4.add_hline(y=40, line_dash="dash", line_color="red", annotation_text="Critical")
    fig4.update_layout(height=400, yaxis_title="Asset Health Index (0-100)",
                       margin=dict(l=40, r=20, t=30, b=30))
    st.plotly_chart(fig4, use_container_width=True)

# Explanations
if show_explanations and n_anom > 0:
    st.markdown("---")
    st.subheader("🔍 Anomaly Explanations (SHAP)")
    with st.spinner("Computing SHAP values for detected anomalies..."):
        expl_df = explain_anomalies(
            detector, X, preds,
            timestamps=feat_df["timestamp"],
            top_k=4,
            max_anomalies_to_explain=12,
        )
    if len(expl_df):
        st.dataframe(
            expl_df[["timestamp", "primary_driver", "top_features"]],
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Primary driver = feature with highest absolute SHAP contribution to the anomaly score.")
    else:
        st.info("No anomalies to explain in the current window.")

# Footer
st.markdown("---")
st.caption(
    "Portfolio demonstration using physics-informed synthetic data for 220 kV GIS assets. "
    "Not for live substation protection or control decisions."
)
