from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
from app.twin.physics import GISBayTwin

app = FastAPI(title="GIS Digital Twin API", version="1.0")
twin = GISBayTwin()

class ActionRequest(BaseModel):
    action: str

class FaultRequest(BaseModel):
    fault_type: str
    severity: float = 1.0

@app.get("/state")
def get_state():
    return twin.state.to_dict()

@app.post("/step")
def step(load_a: Optional[float] = None):
    st = twin.step(load_a)
    return st.to_dict()

@app.post("/action")
def action(req: ActionRequest):
    msg = twin.apply_action(req.action)
    return {"message": msg, "state": twin.state.to_dict()}

@app.post("/inject_fault")
def inject(req: FaultRequest):
    twin.inject_fault(req.fault_type, req.severity)
    return {"message": f"Injected {req.fault_type}", "state": twin.state.to_dict()}

@app.get("/health")
def health():
    return {"status": "ok"}
