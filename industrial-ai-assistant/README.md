# Multi-Agent Industrial Assistant for Plant Operations & Maintenance

**GraphRAG + LangGraph Multi-Agent System** for Oil & Gas / Power Plant Operations

A production-oriented multi-agent AI system that assists plant engineers with:
- Safety compliance checks against IEC 60079 and vendor manuals (GraphRAG)
- Fault diagnostics from SCADA/PLC text logs
- Automated drafting of LOTO (Lockout-Tagout) isolation lists and SOPs

Designed for roles: **Industrial AI Engineer · Generative AI Engineer · AI/ML Engineer**

---

## Architecture Overview

```
User Query / Alarm
        │
        ▼
┌───────────────────┐
│  Orchestrator     │  (LangGraph State Machine)
│  (Supervisor)     │
└─────────┬─────────┘
          │
    ┌─────┴─────┬─────────────┐
    ▼           ▼             ▼
┌────────┐  ┌──────────┐  ┌────────────┐
│ Safety │  │Diagnostics│  │  Workflow  │
│ Agent  │  │  Agent    │  │   Agent    │
│(GraphRAG)│ │ (Logs)   │  │ (LOTO/SOP) │
└────────┘  └──────────┘  └────────────┘
    │           │             │
    └───────────┴─────────────┘
                │
                ▼
         Final Structured Response
         + Citations + Risk Flags
```

### Agents

| Agent | Responsibility | Key Tech |
|-------|----------------|----------|
| **Safety Compliance Agent** | Retrieves and reasons over IEC 60079, hazardous area guidelines, and vendor manuals using GraphRAG | NetworkX knowledge graph + Chroma vector store |
| **Diagnostics Agent** | Parses SCADA/PLC alarm & event logs to classify electrical vs mechanical root cause | Structured extraction + LLM reasoning |
| **Workflow Agent** | Generates LOTO isolation sequences, Permit-to-Work drafts, and SOPs | Template-guided generation + safety constraints from Safety Agent |

---

## Quick Start

### 1. Prerequisites
- Python 3.11+
- OpenAI API key (or any OpenAI-compatible endpoint, e.g. Azure, Groq, local vLLM/Ollama with OpenAI adapter)

### 2. Installation

```bash
cd industrial-ai-assistant
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configuration

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

### 4. Run the Streamlit UI

```bash
streamlit run app/main.py
```

### 5. Run as a pure Python demo (no UI)

```bash
python -m app.demo
```

---

## Project Structure

```
industrial-ai-assistant/
├── app/
│   ├── main.py                 # Streamlit frontend
│   ├── demo.py                 # CLI demo
│   ├── agents/
│   │   ├── safety_agent.py
│   │   ├── diagnostics_agent.py
│   │   ├── workflow_agent.py
│   │   └── graph.py            # LangGraph orchestration
│   ├── rag/
│   │   ├── graph_rag.py        # GraphRAG implementation
│   │   └── vectorstore.py
│   ├── tools/
│   │   └── industrial_tools.py
│   ├── schemas/
│   │   └── models.py           # Pydantic models
│   ├── data/
│   │   ├── sample_manuals/     # Mock IEC & vendor excerpts
│   │   ├── sample_logs/        # Mock SCADA/PLC logs
│   │   └── knowledge/          # Pre-built graph artifacts
│   └── utils/
│       └── config.py
├── tests/
├── requirements.txt
├── .env.example
└── README.md
```

---

## Sample Queries to Try

1. **"GIS Circuit Breaker CB-220-01 is showing SF6 low pressure alarm and phase imbalance. What are the safety steps and LOTO requirements before inspection?"**

2. **"Analyze this SCADA log and tell me if the fault is electrical or mechanical, then draft a LOTO list."**

3. **"What does IEC 60079 say about working on equipment in Zone 1 with potential explosive atmosphere?"**

---

## Safety & Disclaimer

This is a **demonstration / portfolio project**.  
It must **never** be used as the sole source of truth for real plant safety decisions, LOTO, or Permit-to-Work without human expert review and official plant procedures.

The system is deliberately designed with:
- Explicit citation of source documents
- Risk flagging
- Human-in-the-loop checkpoints in the graph

---

## Extending the Project

- Replace mock manuals with real (redacted) vendor PDFs + proper chunking
- Swap Chroma for a production vector DB + Neo4j for full GraphRAG
- Add MCP (Model Context Protocol) tools for live SCADA connectivity
- Deploy agents as separate services with FastAPI + Docker
- Add evaluation harness (faithfulness, citation accuracy, safety constraint adherence)

---

## Author Positioning

This project demonstrates the rare combination of:
- 11+ years heavy Industrial Electrical / OT domain expertise
- Modern Agentic AI (LangGraph)
- GraphRAG for technical compliance documents
- Structured output & safety-aware generation

Ideal for **Industrial AI Engineer**, **Generative AI Engineer**, and **AI Solutions** roles in Energy, Oil & Gas, and Smart Manufacturing.
