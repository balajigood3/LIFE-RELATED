"""Workflow Agent – LOTO / SOP / PTW drafting."""

from __future__ import annotations
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from app.schemas.models import (
    WorkflowOutput, LOTODocument, IsolationStep, SOPDraft,
    SafetyAssessment, DiagnosticsResult
)
from app.utils.config import get_settings


WORKFLOW_SYSTEM_PROMPT = """You are a senior Operations & Maintenance Planning Engineer specializing in 
High-Voltage substations and industrial plants. You produce clear, conservative, and auditable 
LOTO isolation lists and SOP drafts.

Rules you MUST follow:
1. Always include residual energy checks (electrical, mechanical spring, pressure, thermal).
2. Isolation points must be specific (tag numbers when available).
3. Every LOTO document must state that it REQUIRES human validation by an authorized person 
   before use in the field.
4. Incorporate any critical constraints coming from the Safety Assessment.
5. Prefer the sequence: Open breaker → Open disconnectors → Close earth switches → Lock & Tag → Verify.
6. Never authorize live work.
"""


def run_workflow_agent(
    user_query: str,
    safety: Optional[SafetyAssessment] = None,
    diagnostics: Optional[DiagnosticsResult] = None,
    document_type: str = "LOTO",
) -> WorkflowOutput:
    """Execute the Workflow Agent to draft LOTO or SOP."""
    settings = get_settings()

    safety_summary = "No safety assessment provided."
    if safety:
        constraints = "\n".join(
            f"- [{c.severity.upper()}] {c.description} (Source: {c.source.document})"
            for c in safety.constraints
        )
        risk_flags = "\n".join(f"- {r}" for r in safety.risk_flags)
        safety_summary = (
            f"Applicable standards: {', '.join(safety.applicable_standards)}\n"
            f"Constraints:\n{constraints or 'None extracted'}\n"
            f"Risk flags:\n{risk_flags or 'None'}"
        )

    diag_summary = "No diagnostics result provided."
    if diagnostics:
        diag_summary = (
            f"Primary fault: {diagnostics.primary_fault.category} "
            f"({diagnostics.primary_fault.subcategory or 'n/a'}) "
            f"confidence={diagnostics.primary_fault.confidence:.2f}\n"
            f"Urgency: {diagnostics.urgency}\n"
            f"Evidence: {'; '.join(diagnostics.primary_fault.evidence[:3])}"
        )

    parser = PydanticOutputParser(pydantic_object=WorkflowOutput)

    prompt = ChatPromptTemplate.from_messages([
        ("system", WORKFLOW_SYSTEM_PROMPT + "\n\n{format_instructions}"),
        ("human",
         "User Request:\n{query}\n\n"
         "Desired document type: {doc_type}\n\n"
         "Safety Assessment Summary:\n{safety}\n\n"
         "Diagnostics Summary:\n{diag}\n\n"
         "Produce a complete WorkflowOutput. Prefer LOTO when isolation is involved.")
    ])

    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.2,
        api_key=settings.openai_api_key or None,
        base_url=settings.openai_base_url if settings.openai_base_url != "https://api.openai.com/v1" else None,
    )

    chain = prompt | llm | parser

    try:
        result: WorkflowOutput = chain.invoke({
            "query": user_query,
            "doc_type": document_type,
            "safety": safety_summary,
            "diag": diag_summary,
            "format_instructions": parser.get_format_instructions(),
        })
        result.human_review_required = True
        if not result.warnings:
            result.warnings = [
                "This document is AI-generated and MUST be reviewed, validated, and approved "
                "by an authorized competent person before field use."
            ]
        return result
    except Exception as e:
        return WorkflowOutput(
            document_type="LOTO",
            summary=f"Workflow generation failed: {str(e)[:150]}",
            human_review_required=True,
            warnings=["Generation error – do not use. Escalate to human planner."],
        )
