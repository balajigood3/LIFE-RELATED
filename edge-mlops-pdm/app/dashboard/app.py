import streamlit as st
import numpy as np
from app.edge.inference import EdgeInferenceEngine

st.set_page_config(page_title="Edge MLOps PdM", page_icon="🛠️", layout="wide")
st.title("🛠️ Edge MLOps – Predictive Maintenance")

engine = EdgeInferenceEngine()

st.sidebar.header("Asset Sensor Inputs")
vib = st.sidebar.slider("Vibration (mm/s)", 0.2, 4.0, 1.2, 0.1)
temp = st.sidebar.slider("Temperature (°C)", 30.0, 120.0, 65.0, 1.0)
current = st.sidebar.slider("Current (A)", 20.0, 150.0, 80.0, 1.0)
oil = st.sidebar.slider("Oil Quality (0-1)", 0.2, 1.0, 0.8, 0.05)
hours = st.sidebar.slider("Operating Hours", 100, 25000, 8000, 100)

if st.button("Run Edge Inference", type="primary"):
    result = engine.predict({
        "vibration_mms": vib, "temp_c": temp, "current_a": current,
        "oil_quality": oil, "operating_hours": hours,
    })
    c1, c2, c3 = st.columns(3)
    c1.metric("Failure Risk", f"{result.get('failure_risk_proba', 0):.1%}")
    c2.metric("Health Index", f"{result.get('health_index', 0):.0f}")
    c3.metric("Model Version", result.get("model_version", "n/a"))
    st.json(result)

st.markdown("---")
st.caption(f"Drift score (recent window): {engine.drift_score():.4f}")
st.caption("Simulated edge device – portfolio demonstration.")
