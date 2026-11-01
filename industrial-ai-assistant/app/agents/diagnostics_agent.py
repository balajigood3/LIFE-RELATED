"""Diagnostics Agent – SCADA / PLC log analysis."""

from __future__ import annotations
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from app.schemas.models import DiagnosticsResult, FaultClassification
from app.utils.config import get_settings


DIAG_SYSTEM_PROMPT = """You are an expert Power System Protection & High-Voltage Equipment Diagnostics Engineer 
with 15+ years of experience in GIS substations, transformers, and industrial plants.

Analyze the provided SCADA/PLC alarm & event log (and any free-text operator notes) and produce a 
structured DiagnosticsResult.

Focus on:
- Distinguishing electrical root causes (insulation, SF6, protection, CT/VT, secondary circuits) 
  from mechanical (mechanism, springs, linkages, bearings) or other categories.
- Identifying the most urgent issue.
- Recommending concrete next checks that a field engineer can perform safely.

Be concise and evidence-based. Quote specific alarm codes or values from the log.
"""


def run_diagnostics_agent(
    user_query: str,
    scada_log: Optional[str] = None,
) -> DiagnosticsResult:
    """Execute the Diagnostics Agent."""
    settings = get_settings()

    log_text = scada_log or "No SCADA log provided. Rely only on the user query description."

    parser = PydanticOutputParser(pydantic_object=DiagnosticsResult)

    prompt = ChatPromptTemplate.from_messages([
        ("system", DIAG_SYSTEM_PROMPT + "\n\n{format_instructions}"),
        ("human",
         "User Query / Symptom Description:\n{query}\n\n"
         "SCADA / Event Log:\n{log}\n\n"
         "Produce a complete DiagnosticsResult.")
    ])

    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.15,
        api_key=settings.openai_api_key or None,
        base_url=settings.openai_base_url if settings.openai_base_url != "https://api.openai.com/v1" else None,
    )

    chain = prompt | llm | parser

    try:
        result: DiagnosticsResult = chain.invoke({
            "query": user_query,
            "log": log_text,
            "format_instructions": parser.get_format_instructions(),
        })
        # Attach a short excerpt for traceability
        if scada_log and not result.raw_log_excerpt:
            result.raw_log_excerpt = scada_log[:600]
        return result
    except Exception as e:
        return DiagnosticsResult(
            alarm_summary=user_query[:200],
            primary_fault=FaultClassification(
                category="unknown",
                confidence=0.0,
                evidence=[f"Diagnostics agent error: {str(e)[:120]}"]
            ),
            recommended_checks=["Manual review of SCADA historian required."],
            urgency="urgent",
            raw_log_excerpt=scada_log[:400] if scada_log else None,
        )
