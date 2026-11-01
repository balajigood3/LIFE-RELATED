"""Pydantic models for structured agent inputs/outputs."""

from __future__ import annotations
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class SourceCitation(BaseModel):
    document: str = Field(..., description="Source document name or ID")
    section: Optional[str] = None
    excerpt: Optional[str] = None
    relevance_score: Optional[float] = None


class SafetyConstraint(BaseModel):
    constraint_id: str
    description: str
    severity: Literal["critical", "high", "medium", "low"]
    source: SourceCitation
    applicable_zones: List[str] = Field(default_factory=list)  # e.g. Zone 0, Zone 1, Zone 2


class SafetyAssessment(BaseModel):
    """Output of the Safety Compliance Agent."""
    query_summary: str
    applicable_standards: List[str]
    constraints: List[SafetyConstraint]
    risk_flags: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    citations: List[SourceCitation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class FaultClassification(BaseModel):
    category: Literal["electrical", "mechanical", "instrumentation", "process", "unknown"]
    subcategory: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list)


class DiagnosticsResult(BaseModel):
    """Output of the Diagnostics Agent."""
    alarm_summary: str
    primary_fault: FaultClassification
    secondary_suspects: List[FaultClassification] = Field(default_factory=list)
    recommended_checks: List[str] = Field(default_factory=list)
    urgency: Literal["emergency", "urgent", "scheduled", "monitor"] = "scheduled"
    raw_log_excerpt: Optional[str] = None


class IsolationStep(BaseModel):
    step_number: int
    action: str
    equipment_tag: str
    location: Optional[str] = None
    verification: str
    energy_type: Literal["electrical", "mechanical", "hydraulic", "pneumatic", "thermal", "chemical", "other"]
    notes: Optional[str] = None


class LOTODocument(BaseModel):
    """Lockout-Tagout isolation list."""
    equipment: str
    work_description: str
    prepared_by: str = "AI Assistant (requires human validation)"
    steps: List[IsolationStep]
    residual_energy_checks: List[str] = Field(default_factory=list)
    special_precautions: List[str] = Field(default_factory=list)
    references: List[str] = Field(default_factory=list)


class SOPDraft(BaseModel):
    title: str
    purpose: str
    scope: str
    responsibilities: List[str]
    procedure_steps: List[str]
    safety_notes: List[str]
    references: List[str] = Field(default_factory=list)


class WorkflowOutput(BaseModel):
    """Output of the Workflow Agent."""
    document_type: Literal["LOTO", "SOP", "PTW_DRAFT", "CHECKLIST"]
    loto: Optional[LOTODocument] = None
    sop: Optional[SOPDraft] = None
    summary: str
    human_review_required: bool = True
    warnings: List[str] = Field(default_factory=list)


class AgentState(BaseModel):
    """Shared state flowing through the LangGraph."""
    user_query: str
    scada_log: Optional[str] = None
    equipment_tags: List[str] = Field(default_factory=list)
    
    # Intermediate results
    safety_assessment: Optional[SafetyAssessment] = None
    diagnostics_result: Optional[DiagnosticsResult] = None
    workflow_output: Optional[WorkflowOutput] = None
    
    # Control
    current_agent: Optional[str] = None
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    final_response: Optional[str] = None
    error: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
