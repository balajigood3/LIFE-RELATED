"""
Streamlit frontend for the Multi-Agent Industrial Assistant.
"""

from __future__ import annotations
import streamlit as st
from pathlib import Path
import sys

# Ensure package root is on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agents.graph import run_industrial_assistant
from app.utils.config import get_settings, DATA_DIR
from app.rag.vectorstore import build_or_load_vectorstore

st.set_page_config(
    page_title="Industrial AI Assistant | Plant Operations",
    page_icon="⚡",
    layout="wide",
)

# ---------- Sidebar ----------
with st.sidebar:
    st.title("⚡ Industrial AI Assistant")
    st.markdown("**GraphRAG + LangGraph Multi-Agent System**")
    st.markdown("---")
    st.markdown("### Agents")
    st.markdown("- 🛡️ **Safety Compliance** (GraphRAG + IEC 60079)")
    st.markdown("- 🔍 **Diagnostics** (SCADA/PLC logs)")
    st.markdown("- 📋 **Workflow** (LOTO / SOP drafting)")
    st.markdown("---")

    st.markdown("### Sample SCADA Logs")
    log_files = list((DATA_DIR / "sample_logs").glob("*.txt")) if (DATA_DIR / "sample_logs").exists() else []
    selected_log_name = st.selectbox(
        "Load a sample log (optional)",
        options=["None"] + [f.name for f in log_files],
    )
    scada_log_content = None
    if selected_log_name != "None":
        log_path = DATA_DIR / "sample_logs" / selected_log_name
        scada_log_content = log_path.read_text(encoding="utf-8")
        with st.expander("Preview log"):
            st.code(scada_log_content, language="text")

    st.markdown("---")
    st.caption("⚠️ Demo system only. Never use as sole authority for real LOTO / PTW.")

# ---------- Main ----------
st.header("Multi-Agent Industrial Assistant")
st.markdown(
    "Ask about equipment faults, safety requirements, isolation procedures, or paste a SCADA alarm description."
)

# Pre-populate example
example_query = (
    "GIS Circuit Breaker CB-220-01 is showing SF6 low pressure alarm (Stage 2) "
    "and phase current imbalance. There was also a protection trip. "
    "What are the safety constraints and draft a LOTO isolation list before inspection?"
)

query = st.text_area(
    "Your query / alarm description",
    value=example_query,
    height=140,
)

equipment_tags_input = st.text_input(
    "Equipment tags (comma-separated, optional)",
    value="CB-220-01, DS-220-01A, ES-220-01A",
)

col1, col2 = st.columns([1, 4])
with col1:
    run_btn = st.button("🚀 Run Multi-Agent Analysis", type="primary", use_container_width=True)

if run_btn and query.strip():
    tags = [t.strip() for t in equipment_tags_input.split(",") if t.strip()]

    with st.spinner("Running Safety → Diagnostics → Workflow agents..."):
        try:
            # Ensure vector store exists (first run builds it)
            build_or_load_vectorstore()

            result = run_industrial_assistant(
                user_query=query.strip(),
                scada_log=scada_log_content,
                equipment_tags=tags,
            )

            st.success("Analysis complete")

            # Final polished response
            if result.get("final_response"):
                st.markdown("## Final Assistant Response")
                st.markdown(result["final_response"])

            # Expandable raw agent outputs for transparency
            with st.expander("🔍 Diagnostics Agent Output (raw)"):
                st.json(result.get("diagnostics_result") or {})

            with st.expander("🛡️ Safety Agent Output (raw)"):
                st.json(result.get("safety_assessment") or {})

            with st.expander("📋 Workflow / LOTO Agent Output (raw)"):
                st.json(result.get("workflow_output") or {})

            with st.expander("🧠 Agent Messages / Trace"):
                for msg in result.get("messages", []):
                    st.markdown(f"**{msg.get('role', 'system')}**: {msg.get('content', '')}")

        except Exception as e:
            st.error(f"Error during execution: {e}")
            st.exception(e)

elif run_btn:
    st.warning("Please enter a query.")

# Footer
st.markdown("---")
st.caption(
    "Built to demonstrate the intersection of deep Industrial Electrical / OT domain expertise "
    "and modern Agentic AI (LangGraph + GraphRAG). Portfolio project – not for live plant control."
)
