"""
Lightweight GraphRAG implementation for industrial technical documents.

Uses:
- NetworkX for the knowledge graph (entities + relations)
- Chroma for vector retrieval of text chunks
- LLM for entity extraction and final synthesis (called from agents)
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import networkx as nx
from langchain_core.documents import Document

from app.rag.vectorstore import similarity_search, build_or_load_vectorstore
from app.utils.config import get_settings


# Simple domain ontology for industrial electrical / hazardous areas
ENTITY_TYPES = [
    "Standard", "Zone", "Equipment", "ProtectionType", "Alarm",
    "Procedure", "Hazard", "EnergySource", "Document"
]

RELATION_TYPES = [
    "APPLIES_TO", "REQUIRES", "PROTECTS_AGAINST", "PART_OF",
    "REFERENCES", "TRIGGERS", "ISOLATES", "LOCATED_IN"
]


class IndustrialGraphRAG:
    """Graph-augmented retrieval for compliance and safety queries."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self._build_seed_graph()
        self.settings = get_settings()

    def _build_seed_graph(self):
        """Seed the graph with high-value domain knowledge from the demo corpus."""
        # Standards
        self.graph.add_node("IEC_60079-14", type="Standard", label="IEC 60079-14")
        self.graph.add_node("IEC_60079-17", type="Standard", label="IEC 60079-17")
        self.graph.add_node("IEC_60079-19", type="Standard", label="IEC 60079-19")

        # Zones
        for z in ["Zone_0", "Zone_1", "Zone_2"]:
            self.graph.add_node(z, type="Zone", label=z.replace("_", " "))

        # Equipment
        self.graph.add_node("CB-220-01", type="Equipment", label="220kV GIS Circuit Breaker CB-220-01")
        self.graph.add_node("SF6_GIS", type="Equipment", label="SF6 Gas Insulated Switchgear")

        # Hazards
        self.graph.add_node("Explosive_Atmosphere", type="Hazard", label="Explosive Atmosphere")
        self.graph.add_node("SF6_Asphyxiation", type="Hazard", label="SF6 Oxygen Displacement")
        self.graph.add_node("Arc_Byproducts", type="Hazard", label="Toxic SF6 Arc By-products")
        self.graph.add_node("Induced_Voltage", type="Hazard", label="Induced / Residual Voltage")

        # Procedures
        self.graph.add_node("LOTO", type="Procedure", label="Lockout-Tagout")
        self.graph.add_node("PTW", type="Procedure", label="Permit-to-Work")
        self.graph.add_node("Isolation_Verification", type="Procedure", label="Isolation Verification")

        # Relations
        edges = [
            ("IEC_60079-14", "Zone_1", "APPLIES_TO"),
            ("IEC_60079-14", "Zone_2", "APPLIES_TO"),
            ("IEC_60079-17", "LOTO", "REQUIRES"),
            ("IEC_60079-17", "Isolation_Verification", "REQUIRES"),
            ("CB-220-01", "SF6_GIS", "PART_OF"),
            ("SF6_GIS", "SF6_Asphyxiation", "PROTECTS_AGAINST"),  # actually creates the hazard; adjust semantically later
            ("LOTO", "Induced_Voltage", "PROTECTS_AGAINST"),
            ("LOTO", "Explosive_Atmosphere", "PROTECTS_AGAINST"),
            ("PTW", "LOTO", "REQUIRES"),
            ("CB-220-01", "LOTO", "REQUIRES"),
        ]
        for u, v, rel in edges:
            self.graph.add_edge(u, v, relation=rel)

    def retrieve(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Hybrid retrieval:
        1. Vector similarity on document chunks
        2. Graph neighborhood expansion around detected entities
        """
        # 1. Vector retrieval
        vector_docs = similarity_search(query, k=k)

        # 2. Simple entity mention detection (keyword based for demo robustness)
        detected_entities = self._detect_entities(query)

        # 3. Graph expansion
        graph_context = []
        for ent in detected_entities:
            if ent in self.graph:
                neighbors = list(self.graph.successors(ent)) + list(self.graph.predecessors(ent))
                for n in neighbors:
                    edge_data = self.graph.get_edge_data(ent, n) or self.graph.get_edge_data(n, ent) or {}
                    rel = edge_data.get("relation", "RELATED")
                    graph_context.append({
                        "entity": ent,
                        "related": n,
                        "relation": rel,
                        "related_label": self.graph.nodes[n].get("label", n)
                    })

        return {
            "vector_chunks": [
                {
                    "content": d.page_content,
                    "source": d.metadata.get("source", "unknown"),
                    "score": None
                }
                for d in vector_docs
            ],
            "detected_entities": detected_entities,
            "graph_relations": graph_context,
        }

    def _detect_entities(self, text: str) -> List[str]:
        """Lightweight keyword → entity mapping for the demo."""
        text_lower = text.lower()
        found = []

        mapping = {
            "iec 60079-14": "IEC_60079-14",
            "iec 60079-17": "IEC_60079-17",
            "iec 60079": "IEC_60079-14",
            "zone 0": "Zone_0",
            "zone 1": "Zone_1",
            "zone 2": "Zone_2",
            "cb-220-01": "CB-220-01",
            "circuit breaker": "CB-220-01",
            "sf6": "SF6_GIS",
            "gis": "SF6_GIS",
            "loto": "LOTO",
            "lockout": "LOTO",
            "tagout": "LOTO",
            "permit to work": "PTW",
            "ptw": "PTW",
            "isolation": "Isolation_Verification",
        }

        for keyword, ent in mapping.items():
            if keyword in text_lower and ent not in found:
                found.append(ent)
        return found

    def format_context_for_llm(self, retrieval_result: Dict[str, Any]) -> str:
        """Turn retrieval result into a clean context string for the Safety Agent."""
        parts = ["### Retrieved Document Chunks\n"]
        for i, chunk in enumerate(retrieval_result["vector_chunks"], 1):
            parts.append(f"[{i}] Source: {chunk['source']}\n{chunk['content'][:1200]}\n")

        if retrieval_result["graph_relations"]:
            parts.append("\n### Knowledge Graph Relations\n")
            for rel in retrieval_result["graph_relations"]:
                parts.append(f"- {rel['entity']} --[{rel['relation']}]--> {rel['related_label']}")

        if retrieval_result["detected_entities"]:
            parts.append(f"\n### Detected Domain Entities: {', '.join(retrieval_result['detected_entities'])}")

        return "\n".join(parts)


# Singleton for convenience
_graph_rag: Optional[IndustrialGraphRAG] = None


def get_graph_rag() -> IndustrialGraphRAG:
    global _graph_rag
    if _graph_rag is None:
        _graph_rag = IndustrialGraphRAG()
    return _graph_rag
