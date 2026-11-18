import pytest
from agent_platform.engine import safe_calculate, make_plan, run_workflow, extract_local_knowledge, MemoryStore, export_report

def test_safe_calculator_basic_math():
    assert safe_calculate("(10 + 5) * 2") == 30

def test_safe_calculator_rejects_code_execution():
    with pytest.raises(ValueError):
        safe_calculate("__import__('os').system('echo unsafe')")

def test_safe_calculator_rejects_names():
    with pytest.raises(ValueError):
        safe_calculate("x + 1")

def test_safe_calculator_bounds_exponent():
    with pytest.raises(ValueError):
        safe_calculate("2 ** 20")

def test_plan_has_five_agent_roles():
    assert len(make_plan("Build an assistant")) == 5

def test_empty_goal_rejected():
    with pytest.raises(ValueError):
        make_plan(" ")

def test_local_knowledge_retrieval():
    docs = ["RAG retrieves relevant documents for grounded generation.", "Cooking pasta."]
    result = extract_local_knowledge("RAG document retrieval", docs)
    assert result and "RAG" in result[0]

def test_workflow_completes_with_checks():
    result = run_workflow("Build a RAG assistant", ["RAG uses document retrieval."])
    assert result["status"] == "completed"
    assert len(result["steps"]) == 5
    assert all(result["quality_checks"][k] for k in ["goal_present", "plan_has_five_agents", "output_non_empty"])

def test_workflow_empty_docs_is_supported():
    result = run_workflow("Plan a project")
    assert result["context"] == []
    assert "No matching local document" in result["output"]

def test_memory_store_round_trip(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.add("Use local models where possible.")
    assert store.list() == ["Use local models where possible."]

def test_memory_rejects_empty():
    store = MemoryStore(":memory:")
    with pytest.raises(ValueError):
        store.add(" ")

def test_report_is_json():
    result = run_workflow("Draft a plan")
    assert '"workflow_id"' in export_report(result)
