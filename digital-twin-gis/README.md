# Digital Twin + Agentic Control for 220 kV GIS Bay

Physics-informed digital twin of a Gas-Insulated Switchgear bay combined with an agentic decision layer that recommends safe operational actions.

## Features
- Real-time simulation of SF6 pressure, temperature, contact wear, load current
- Fault injection (SF6 leak, contact overheating, pole discrepancy)
- Agentic AI layer (LangGraph) that proposes isolation / mitigation steps
- FastAPI control interface + Streamlit operator dashboard

## Quick Start
```bash
pip install -r requirements.txt
python -m app.twin.simulate          # run standalone simulation
uvicorn app.api.main:app --reload    # API
streamlit run app/dashboard/app.py   # Dashboard
```
