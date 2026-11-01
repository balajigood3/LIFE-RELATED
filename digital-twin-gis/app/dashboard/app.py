import streamlit as st
import time
from app.twin.physics import GISBayTwin

st.set_page_config(page_title="GIS Digital Twin", page_icon="⚡", layout="wide")
st.title("⚡ 220 kV GIS Digital Twin + Agentic Control")

if "twin" not in st.session_state:
    st.session_state.twin = GISBayTwin()

twin = st.session_state.twin

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Step Simulation"):
        twin.step()
with col2:
    fault = st.selectbox("Inject Fault", ["sf6_leak", "contact_overheating", "mechanical_binding"])
    if st.button("Inject"):
        twin.inject_fault(fault, 1.5)
with col3:
    action = st.selectbox("Apply Action", ["trip_breaker", "open_disconnector", "close_earthing", "reset_faults"])
    if st.button("Execute Action"):
        msg = twin.apply_action(action)
        st.toast(msg)

st.markdown("---")
s = twin.state.to_dict()
k1, k2, k3, k4 = st.columns(4)
k1.metric("SF6 Pressure", f"{s['sf6_pressure_bar']:.2f} bar")
k2.metric("Contact Temp", f"{s['temp_contact_c']:.1f} °C")
k3.metric("Load Current", f"{s['load_current_a']:.0f} A")
k4.metric("State", s["state"].upper())

st.json(s)
st.caption("Digital Twin demo – not for live substation control.")
