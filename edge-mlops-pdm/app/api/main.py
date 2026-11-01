from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Optional
from app.edge.inference import EdgeInferenceEngine
from pathlib import Path
import json

app = FastAPI(title="Edge MLOps PdM API", version="1.0")
engine = EdgeInferenceEngine()

class PredictRequest(BaseModel):
    vibration_mms: float
    temp_c: float
    current_a: float
    oil_quality: float
    operating_hours: float

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": engine.bundle is not None,
        "model_version": engine.bundle.get("version") if engine.bundle else None,
        "drift_score": engine.drift_score(),
    }

@app.post("/predict")
def predict(req: PredictRequest):
    return engine.predict(req.model_dump())

@app.get("/model/info")
def model_info():
    meta_path = Path(__file__).resolve().parents[2] / "models" / "metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {"message": "No metadata"}

@app.post("/model/reload")
def reload_model():
    engine.load_model()
    return {"status": "reloaded", "version": engine.bundle.get("version") if engine.bundle else None}
