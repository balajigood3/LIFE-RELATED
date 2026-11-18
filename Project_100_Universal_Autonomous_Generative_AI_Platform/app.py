import json
import streamlit as st
from agent_platform.engine import run_workflow, MemoryStore, safe_calculate, export_report

st.set_page_config(page_title="Universal Autonomous Generative AI Platform", layout="wide")
st.title("Universal Autonomous Generative AI Platform")
st.caption("Local-first multi-agent workflow prototype — திட்டமிடல், local knowledge retrieval, draft generation, review.")

store = MemoryStore("data/platform_memory.db")
tab1, tab2, tab3 = st.tabs(["Autonomous workflow", "Memory", "Safe calculator"])

with tab1:
    goal = st.text_area("Your objective", placeholder="Example: Create a launch plan for a local knowledge assistant", height=120)
    docs = st.text_area("Local knowledge documents (one document per line)", height=130)
    if st.button("Run agent workflow", type="primary"):
        try:
            result = run_workflow(goal, docs.splitlines(), store.list())
            st.session_state.workflow_result = result
        except Exception as exc:
            st.error(str(exc))
    if "workflow_result" in st.session_state:
        result = st.session_state.workflow_result
        st.metric("Workflow status", result["status"])
        st.subheader("Agent execution trace")
        for step in result["steps"]:
            st.write(f"**{step['agent']}** — {step['status']}: {step['task']}")
        st.subheader("Quality checks")
        st.json(result["quality_checks"])
        st.subheader("Final output")
        st.markdown(result["output"])
        st.download_button("Download workflow JSON", export_report(result),
                           "workflow_report.json", "application/json")

with tab2:
    memory_text = st.text_area("Add a reusable note", max_chars=5000)
    if st.button("Save note"):
        try:
            store.add(memory_text)
            st.success("Saved to local SQLite memory.")
        except Exception as exc:
            st.error(str(exc))
    st.write("Recent memory")
    for item in store.list(30):
        st.markdown(f"- {item}")

with tab3:
    expression = st.text_input("Arithmetic expression", "((120 + 30) * 2) / 5")
    if st.button("Calculate"):
        try:
            st.success(f"Result: {safe_calculate(expression)}")
        except Exception as exc:
            st.error(str(exc))

st.warning("Prototype safety boundary: agents do not execute arbitrary shell commands, send messages, spend money, or modify external systems. Review outputs before acting.")
