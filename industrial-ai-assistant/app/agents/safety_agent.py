"""Safety Compliance Agent – GraphRAG powered."""

from __future__ import annotations
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from app.schemas.models import SafetyAssessment, SafetyConstraint, SourceCitation
from app.rag.graph_rag import get_graph_rag
from app.utils.config import get_settings


SAFETY_SYSTEM_PROMPT = """You are a senior Hazardous Area and Electrical Safety Compliance Specialist 
with deep expertise in IEC 60079 series, SF6 GIS equipment, and industrial LOTO/PTW practices.

Your job is to analyze the user query + retrieved technical context and produce a structured 
Safety Assessment. Be precise, conservative, and always cite sources.

Rules:
- Never invent requirements that are not supported by the provided context or well-established IEC principles.
- Flag any situation that requires a formal Permit-to-Work or competent person.
- Prefer over-isolation and higher PPE rather than under-isolation.
- Clearly separate "mandatory from standards" vs "best practice recommendations".
"""


def run_safety_agent(user_query: str, equipment_tags: list[str] | None = None) -> SafetyAssessment:
    """Execute the Safety Compliance Agent."""
    settings = get_settings()
    graph_rag = get_graph_rag()

    # Retrieve hybrid context
    retrieval = graph_rag.retrieve(user_query, k=5)
    context_str = graph_rag.format_context_for_llm(retrieval)

    parser = PydanticOutputParser(pydantic_object=SafetyAssessment)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SAFETY_SYSTEM_PROMPT + "\n\n{format_instructions}"),
        ("human", 
         "User Query:\n{query}\n\n"
         "Equipment Tags (if any): {tags}\n\n"
         "Retrieved Context:\n{context}\n\n"
         "Produce a complete SafetyAssessment.")
    ])

    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.1,
        api_key=settings.openai_api_key or None,
        base_url=settings.openai_base_url if settings.openai_base_url != "https://api.openai.com/v1" else None,
    )

    chain = prompt | llm | parser

    try:
        result: SafetyAssessment = chain.invoke({
            "query": user_query,
            "tags": ", ".join(equipment_tags or []) or "None specified",
            "context": context_str,
            "format_instructions": parser.get_format_instructions(),
        })
        return result
    except Exception as e:
        # Graceful fallback
        return SafetyAssessment(
            query_summary=user_query[:200],
            applicable_standards=["IEC 60079 (retrieval or parsing failed)"],
            constraints=[],
            risk_flags=[f"Safety agent error: {str(e)[:150]}. Manual expert review mandatory."],
            recommendations=["Escalate to certified Hazardous Area competent person immediately."],
            citations=[],
            confidence=0.0,
        )
