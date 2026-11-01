"""
LangGraph multi-agent orchestration for the Industrial Assistant.

Flow:
  START → classify_intent → (safety / diagnostics / both) → workflow (optional) → synthesize → END
"""

from __future__ import annotations
from typing import TypedDict, Annotated, Literal, Optional, List, Dict, Any
import operator

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from app.schemas.models import (
    AgentState, SafetyAssessment, DiagnosticsResult, WorkflowOutput
)
from app.agents.safety_agent import run_safety_agent
from app.agents.diagnostics_agent import run_diagnostics_agent
from app.agents.workflow_agent import run_workflow_agent
from app.utils.config import get_settings


# ---------- Graph State (using TypedDict for LangGraph compatibility) ----------

class GraphState(TypedDict):
    user_query: str
    scada_log: Optional[str]
    equipment_tags: List[str]
    needs_safety: bool
    needs_diagnostics: bool
    needs_workflow: bool
    safety_assessment: Optional[Dict[str, Any]]
    diagnostics_result: Optional[Dict[str, Any]]
    workflow_output: Optional[Dict[str, Any]]
    final_response: Optional[str]
    error: Optional[str]
    messages: Annotated[List[Dict[str, str]], operator.add]


# ---------- Nodes ----------

def classify_intent(state: GraphState) -> GraphState:
    """Decide which agents are required."""
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.0,
        api_key=settings.openai_api_key or None,
        base_url=settings.openai_base_url if settings.openai_base_url != "https://api.openai.com/v1" else None,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an intent classifier for an industrial plant assistant.\n"
         "Given the user query, answer with a JSON-like decision:\n"
         "needs_safety: true if the query involves standards, hazardous areas, IEC, zones, LOTO rules, isolation safety, PPE, SF6 hazards.\n"
         "needs_diagnostics: true if there is an alarm, fault, trip, abnormal reading, SCADA log, or symptom description.\n"
         "needs_workflow: true if the user wants a LOTO list, isolation steps, SOP, procedure, or PTW draft.\n"
         "Reply ONLY with three lines:\n"
         "needs_safety: true/false\n"
         "needs_diagnostics: true/false\n"
         "needs_workflow: true/false"),
        ("human", "{query}")
    ])

    resp = llm.invoke(prompt.format(query=state["user_query"])).content.lower()

    needs_safety = "needs_safety: true" in resp
    needs_diagnostics = "needs_diagnostics: true" in resp
    needs_workflow = "needs_workflow: true" in resp

    # Sensible defaults for industrial queries
    if not any([needs_safety, needs_diagnostics, needs_workflow]):
        needs_safety = True
        needs_diagnostics = bool(state.get("scada_log"))
        needs_workflow = True

    return {
        **state,
        "needs_safety": needs_safety,
        "needs_diagnostics": needs_diagnostics,
        "needs_workflow": needs_workflow,
        "messages": [{"role": "system", "content": f"Intent → safety={needs_safety}, diag={needs_diagnostics}, workflow={needs_workflow}"}],
    }


def safety_node(state: GraphState) -> GraphState:
    if not state.get("needs_safety"):
        return state
    result = run_safety_agent(
        user_query=state["user_query"],
        equipment_tags=state.get("equipment_tags") or [],
    )
    return {
        **state,
        "safety_assessment": result.model_dump(),
        "messages": [{"role": "assistant", "content": f"Safety assessment completed (confidence={result.confidence:.2f})"}],
    }


def diagnostics_node(state: GraphState) -> GraphState:
    if not state.get("needs_diagnostics"):
        return state
    result = run_diagnostics_agent(
        user_query=state["user_query"],
        scada_log=state.get("scada_log"),
    )
    return {
        **state,
        "diagnostics_result": result.model_dump(),
        "messages": [{"role": "assistant", "content": f"Diagnostics completed → {result.primary_fault.category} ({result.urgency})"}],
    }


def workflow_node(state: GraphState) -> GraphState:
    if not state.get("needs_workflow"):
        return state

    safety_obj = None
    if state.get("safety_assessment"):
        safety_obj = SafetyAssessment(**state["safety_assessment"])

    diag_obj = None
    if state.get("diagnostics_result"):
        diag_obj = DiagnosticsResult(**state["diagnostics_result"])

    result = run_workflow_agent(
        user_query=state["user_query"],
        safety=safety_obj,
        diagnostics=diag_obj,
        document_type="LOTO",
    )
    return {
        **state,
        "workflow_output": result.model_dump(),
        "messages": [{"role": "assistant", "content": "Workflow / LOTO document drafted"}],
    }


def synthesize(state: GraphState) -> GraphState:
    """Produce a clear, structured final response for the user."""
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.2,
        api_key=settings.openai_api_key or None,
        base_url=settings.openai_base_url if settings.openai_base_url != "https://api.openai.com/v1" else None,
    )

    parts = [f"**User Query**\n{state['user_query']}\n"]

    if state.get("diagnostics_result"):
        d = state["diagnostics_result"]
        parts.append("### Diagnostics Summary")
        parts.append(f"- Primary fault category: **{d['primary_fault']['category']}** "
                     f"(confidence {d['primary_fault']['confidence']:.0%})")
        if d['primary_fault'].get('subcategory'):
            parts.append(f"- Subcategory: {d['primary_fault']['subcategory']}")
        parts.append(f"- Urgency: **{d['urgency']}**")
        if d.get("recommended_checks"):
            parts.append("- Recommended checks:")
            for c in d["recommended_checks"][:5]:
                parts.append(f"  - {c}")
        parts.append("")

    if state.get("safety_assessment"):
        s = state["safety_assessment"]
        parts.append("### Safety & Compliance Assessment")
        parts.append(f"- Applicable standards: {', '.join(s.get('applicable_standards', []))}")
        if s.get("risk_flags"):
            parts.append("- **Risk Flags**:")
            for r in s["risk_flags"]:
                parts.append(f"  - ⚠️ {r}")
        if s.get("constraints"):
            parts.append("- Key Constraints:")
            for c in s["constraints"][:4]:
                parts.append(f"  - [{c['severity'].upper()}] {c['description']}")
        if s.get("recommendations"):
            parts.append("- Recommendations:")
            for r in s["recommendations"][:4]:
                parts.append(f"  - {r}")
        parts.append("")

    if state.get("workflow_output"):
        w = state["workflow_output"]
        parts.append("### Generated Work Document")
        parts.append(f"- Type: {w.get('document_type')}")
        parts.append(f"- Summary: {w.get('summary', '')}")
        if w.get("loto") and w["loto"].get("steps"):
            parts.append("\n**LOTO Isolation Steps (Draft – Human Review Required)**")
            for step in w["loto"]["steps"]:
                parts.append(
                    f"{step['step_number']}. **{step['action']}** "
                    f"[{step['equipment_tag']}] – Verify: {step['verification']}"
                )
            if w["loto"].get("special_precautions"):
                parts.append("\nSpecial Precautions:")
                for p in w["loto"]["special_precautions"]:
                    parts.append(f"- {p}")
        if w.get("warnings"):
            parts.append("\n**WARNINGS**")
            for warn in w["warnings"]:
                parts.append(f"- 🛑 {warn}")
        parts.append("")

    parts.append("---\n*This output is generated by an AI assistant for decision-support only. "
                 "All isolation, LOTO, and Permit-to-Work documents require validation by authorized plant personnel.*")

    final_text = "\n".join(parts)

    # Optional: one LLM polish pass for readability
    polish_prompt = (
        "You are a technical writer for industrial operations. "
        "Polish the following structured report for clarity and professional tone. "
        "Do NOT remove any warnings, risk flags, or the human-review disclaimer. "
        "Keep all technical tags and step numbers intact.\n\n"
        f"{final_text}"
    )
    try:
        polished = llm.invoke(polish_prompt).content
    except Exception:
        polished = final_text

    return {
        **state,
        "final_response": polished,
        "messages": [{"role": "assistant", "content": "Final response synthesized"}],
    }


# ---------- Build Graph ----------

def build_industrial_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("classify", classify_intent)
    workflow.add_node("safety", safety_node)
    workflow.add_node("diagnostics", diagnostics_node)
    workflow.add_node("workflow", workflow_node)
    workflow.add_node("synthesize", synthesize)

    workflow.set_entry_point("classify")

    # Linear but conditional execution is handled inside nodes via flags
    workflow.add_edge("classify", "safety")
    workflow.add_edge("safety", "diagnostics")
    workflow.add_edge("diagnostics", "workflow")
    workflow.add_edge("workflow", "synthesize")
    workflow.add_edge("synthesize", END)

    return workflow.compile()


# Convenience runner
def run_industrial_assistant(
    user_query: str,
    scada_log: Optional[str] = None,
    equipment_tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    graph = build_industrial_graph()
    initial: GraphState = {
        "user_query": user_query,
        "scada_log": scada_log,
        "equipment_tags": equipment_tags or [],
        "needs_safety": False,
        "needs_diagnostics": False,
        "needs_workflow": False,
        "safety_assessment": None,
        "diagnostics_result": None,
        "workflow_output": None,
        "final_response": None,
        "error": None,
        "messages": [],
    }
    result = graph.invoke(initial)
    return result
