# Project 100 — Universal Autonomous Generative AI Platform

A local-first, bounded multi-agent workflow prototype. It coordinates planner, researcher, builder, critic and reporter roles; retrieves relevant snippets from user-provided local text; stores reusable notes in SQLite; and exports an execution report.

## Features
- Five-role agent workflow and execution trace
- Local text knowledge retrieval (bounded to top 3 snippets)
- SQLite memory store
- Safe arithmetic tool using a restricted AST evaluator (no `eval`)
- Workflow quality checks and JSON export
- Streamlit UI
- Dockerfile and Docker Compose
- No paid API keys or external model dependency

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Open `http://localhost:8501`.

## Docker
```bash
docker compose up --build
```

## Tests
```bash
pytest -q
```

## Architecture
`User Goal -> Planner -> Researcher -> Builder -> Critic -> Reporter -> JSON Report`

The Researcher searches only the local documents supplied in the UI. Memory is local SQLite. The current Builder produces a structured draft rather than calling a hosted LLM.

## Safety and limitations
- This is a runnable orchestration prototype, not a fully general autonomous intelligence.
- Agent roles are bounded deterministic components; no paid API, external browsing, or model inference is performed.
- Local keyword retrieval can miss semantically relevant content.
- Quality checks confirm workflow structure, not factual correctness.
- No arbitrary shell execution, external side effects, payment, or automatic deployment is enabled.
- Human review is required before consequential use.
- If connecting a local model later, add timeouts, output validation, prompt-injection defenses, access controls and audit logs.
