"""Simulated edge inference + simple drift monitor."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import joblib
from typing import Dict, Any, List

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "pdm_model.joblib"

class EdgeInferenceEngine:
    def __init__(self):
        self.bundle = None
        self.recent_scores: List[float] = []
        self.load_model()

    def load_model(self):
        if MODEL_PATH.exists():
            self.bundle = joblib.load(MODEL_PATH)
            print(f"Edge loaded model v{self.bundle.get('version')} stage={self.bundle.get('stage')}")
        else:
            print("No model found on edge – run training first")

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        if self.bundle is None:
            return {"error": "model_not_loaded"}
        names = self.bundle["feature_names"]
        x = np.array([[features[n] for n in names]])
        model = self.bundle["model"]
        proba = float(model.predict_proba(x)[0, 1])
        pred = int(proba >= 0.5)
        self.recent_scores.append(proba)
        if len(self.recent_scores) > 200:
            self.recent_scores = self.recent_scores[-200:]
        return {
            "failure_risk_proba": round(proba, 4),
            "prediction": pred,
            "health_index": round(100 * (1 - proba), 1),
            "model_version": self.bundle.get("version"),
        }

    def drift_score(self) -> float:
        """Very simple population drift proxy: std of recent scores."""
        if len(self.recent_scores) < 30:
            return 0.0
        return float(np.std(self.recent_scores))
