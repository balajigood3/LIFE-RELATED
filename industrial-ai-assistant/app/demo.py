"""CLI demo of the Multi-Agent Industrial Assistant."""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agents.graph import run_industrial_assistant
from app.rag.vectorstore import build_or_load_vectorstore
from app.utils.config import DATA_DIR


def main():
    print("=" * 70)
    print("  Multi-Agent Industrial Assistant – CLI Demo")
    print("=" * 70)

    # Build vector store once
    print("\n[1/3] Building / loading vector store from manuals...")
    build_or_load_vectorstore()
    print("      Done.")

    # Load a sample log
    log_path = DATA_DIR / "sample_logs" / "scada_alarm_log_01.txt"
    scada_log = log_path.read_text(encoding="utf-8") if log_path.exists() else None

    query = (
        "GIS Circuit Breaker CB-220-01 is showing SF6 low pressure alarm (Stage 2) "
        "and phase current imbalance after a protection trip. "
        "Provide safety constraints from IEC 60079 and draft a LOTO isolation list."
    )

    print("\n[2/3] Running multi-agent graph...")
    print(f"Query: {query[:100]}...")
    result = run_industrial_assistant(
        user_query=query,
        scada_log=scada_log,
        equipment_tags=["CB-220-01", "DS-220-01A", "ES-220-01A"],
    )

    print("\n[3/3] Final Response:\n")
    print("-" * 70)
    print(result.get("final_response") or "No final response generated.")
    print("-" * 70)

    print("\nAgent trace:")
    for m in result.get("messages", []):
        print(f"  • {m.get('content')}")


if __name__ == "__main__":
    main()
