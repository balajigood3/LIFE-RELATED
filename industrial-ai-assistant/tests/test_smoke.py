"""Smoke tests that do not require an API key."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.rag.graph_rag import IndustrialGraphRAG
from app.schemas.models import SafetyAssessment, DiagnosticsResult, WorkflowOutput


def test_graph_rag_seed():
    rag = IndustrialGraphRAG()
    assert "IEC_60079-14" in rag.graph
    assert "LOTO" in rag.graph
    result = rag.retrieve("SF6 low pressure on CB-220-01 Zone 1")
    assert "vector_chunks" in result
    assert isinstance(result["detected_entities"], list)


def test_pydantic_models():
    # Just ensure models can be instantiated
    s = SafetyAssessment(
        query_summary="test",
        applicable_standards=["IEC 60079"],
        constraints=[],
        confidence=0.5,
    )
    assert s.confidence == 0.5
