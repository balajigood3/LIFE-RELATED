"""
Placeholder for future industrial tools (MCP-style or LangChain tools).

Examples of tools you can add later:
- Live SCADA tag reader (via OPC-UA / MQTT / historian API)
- Equipment hierarchy lookup
- Permit-to-Work system integration
- Real-time SF6 pressure trend fetcher
"""

from langchain_core.tools import tool


@tool
def get_equipment_metadata(tag: str) -> str:
    """Return basic metadata for a known equipment tag (demo lookup)."""
    catalog = {
        "CB-220-01": "220 kV SF6 GIS Circuit Breaker, Bay 01, Manufacturer: GenericGIS",
        "DS-220-01A": "220 kV Disconnector, side A of CB-220-01",
        "ES-220-01A": "220 kV Earthing Switch, side A of CB-220-01",
        "T-110-02": "110/33 kV Power Transformer, Bay 02",
    }
    return catalog.get(tag.upper(), f"No metadata found for tag '{tag}' in demo catalog.")


@tool
def list_sample_alarms() -> str:
    """List available sample SCADA alarm logs in the demo."""
    return (
        "Available demo logs:\n"
        "- scada_alarm_log_01.txt : CB-220-01 SF6 low pressure + phase imbalance + trip\n"
        "- scada_alarm_log_02.txt : Transformer T-110-02 winding temp + Buchholz + fan failure"
    )
