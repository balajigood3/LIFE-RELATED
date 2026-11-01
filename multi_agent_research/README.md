# Multi-Agent Research System

A complete, production-style multi-agent research pipeline built with **CrewAI**.

Multiple specialized AI agents collaborate to research a topic, analyze findings, review quality, and produce a polished report — demonstrating the same patterns companies use for automated analysis, information gathering, and decision support.

---

## Architecture Overview

```
┌─────────────────────┐
│  Senior Researcher  │  ← Web search (Serper)
│  (gathers facts)    │
└──────────┬──────────┘
           │ context hand-off
┌──────────▼──────────┐
│  Research Analyst   │  ← Synthesizes insights
│  (analysis)         │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  QA Reviewer        │  ← Critiques accuracy & bias
│  (quality gate)     │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Report Writer      │  ← Final polished Markdown
│  (deliverable)      │
└─────────────────────┘
```

### Agents

| Agent                    | Role                                      | Tools          |
|--------------------------|-------------------------------------------|----------------|
| Senior Research Analyst  | Gather recent, credible information       | SerperDevTool  |
| Research Analyst         | Synthesize findings into prioritized insights | —            |
| Quality Assurance Reviewer | Check accuracy, completeness, bias      | —              |
| Research Report Writer   | Produce stakeholder-ready Markdown report | —              |

### Orchestration Modes

- **Sequential** (default) – fixed pipeline. Predictable, easy to debug and explain.
- **Hierarchical** – a manager agent dynamically delegates tasks and validates outputs. More adaptive for complex topics.

Agents share intermediate results via CrewAI’s `context` mechanism and optional short-term memory.

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- OpenAI API key
- Serper.dev API key (free tier is sufficient) → https://serper.dev

### 2. Install

```bash
cd multi_agent_research
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and add your keys
```

### 4. Run

```bash
# Interactive
python main.py

# With arguments
python main.py --topic "Agentic AI adoption in software engineering 2025-2026"
python main.py --topic "RAG vs long-context models" --process hierarchical
```

The final report is printed to the console **and** saved to:

```
output/research_report.md
```

---

## Project Structure

```
multi_agent_research/
├── main.py              # Full system – agents, tasks, crew, CLI
├── requirements.txt
├── .env.example
├── README.md
└── output/
    └── research_report.md   # Generated on each run
```

---

## Interview Talking Points

This project is designed so you can clearly explain:

1. **Specialization**  
   Each agent has a narrow, high-quality role instead of one overloaded LLM prompt.

2. **Task delegation & communication**  
   Explicit `context=[previous_task]` hand-offs make intermediate results visible and controllable.

3. **Orchestration patterns**  
   Sequential vs Hierarchical – when to choose each and the trade-offs.

4. **Tool use**  
   Only the Researcher has search capability; the other agents focus on reasoning and quality.

5. **Quality loops**  
   The Reviewer acts as a critic, reducing hallucinations and bias before the final report is written.

6. **Observability & debugging**  
   `verbose=True` + saved Markdown output make the system transparent.

7. **Extensibility ideas** (great for follow-up questions)
   - Parallel research agents for sub-topics
   - Pydantic structured outputs between agents
   - Human-in-the-loop approval gates
   - Cost / latency tracking per agent
   - Long-term memory or knowledge base
   - Switching to AutoGen for conversational debate style

---

## Comparison: CrewAI vs AutoGen

| Aspect              | CrewAI (this project)              | AutoGen                              |
|---------------------|------------------------------------|--------------------------------------|
| Mental model        | Role-based team + tasks            | Conversational group chat            |
| Best for            | Structured research pipelines      | Open-ended debate / iterative refinement |
| Learning curve      | Lower                              | Medium                               |
| Debugging           | Linear task logs                   | Message traces across agents         |
| Production fit      | Excellent for predictable workflows| Strong for research & coding agents  |

You can implement the same research workflow in AutoGen if interviewers ask for a comparison.

---

## Possible Extensions

- Add a second researcher specialized in academic / scholarly sources (`SerperDevTool` with scholar endpoint).
- Use Pydantic models for structured inter-agent messages.
- Add a final “Fact-Checker” agent that re-searches the most critical claims.
- Wrap the system in Streamlit, Gradio, or FastAPI.
- Track token usage and cost per agent/run.
- Add hierarchical process with a custom manager agent that has a specific decision style.

---

## License

MIT – feel free to use and adapt for portfolios, interviews, or production prototypes.
