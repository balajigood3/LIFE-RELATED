"""Agentic decision layer that recommends actions based on twin state."""
from __future__ import annotations
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.utils.config import get_settings


SYSTEM = """You are an expert HV Substation Operations AI.
Given the current digital-twin state of a 220 kV GIS bay, recommend the safest next actions.
Always prefer isolation over continued operation when critical thresholds are crossed.
Return a short ranked list of recommended actions and a one-sentence rationale.
Possible actions: trip_breaker, open_disconnector, close_earthing, continue_monitor, request_human.
"""

def recommend_actions(state: Dict[str, Any]) -> str:
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.1,
        api_key=settings.openai_api_key or None,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        ("human", "Current GIS Bay State:\n{state}\n\nRecommend actions:")
    ])
    chain = prompt | llm
    return chain.invoke({"state": str(state)}).content
