"""Cloud training pipeline with MLflow."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import joblib

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
MLRUNS = ROOT / "mlruns"

def make_synthetic_pdm(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    # Features: vibration, temperature, current, oil_quality, hours
    vib = rng.normal(1.2, 0.4, n)
    temp = rng.normal(65, 8, n)
    current = rng.normal(80, 12, n)
    oil = rng.uniform(0.4, 1.0, n)
    hours = rng.uniform(100, 20000, n)

    # Label: failure risk
    risk = (vib > 2.0).astype(int) + (temp > 85).astype(int) + (oil < 0.55).astype(int) + (hours > 15000).astype(int)
    y = (risk >= 2).astype(int)

    df = pd.DataFrame({
        "vibration_mms": vib, "temp_c": temp, "current_a": current,
        "oil_quality": oil, "operating_hours": hours, "label": y
    })
    return df

def train():
    print("Training PdM classifier...")
    df = make_synthetic_pdm()
    X = df.drop(columns=["label"])
    y = df["label"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

    mlflow.set_tracking_uri(f"file://{MLRUNS.resolve()}")
    mlflow.set_experiment("edge_pdm")

    with mlflow.start_run(run_name="gb_classifier"):
        model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
        model.fit(Xtr, ytr)
        preds = model.predict(Xte)
        proba = model.predict_proba(Xte)[:, 1]
        f1 = f1_score(yte, preds)
        auc = roc_auc_score(yte, proba)
        mlflow.log_metrics({"f1": f1, "auc": auc})
        mlflow.sklearn.log_model(model, "model")
        print(f"F1={f1:.3f}  AUC={auc:.3f}")

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model": model,
            "feature_names": list(X.columns),
            "metrics": {"f1": float(f1), "auc": float(auc)},
            "version": "1.0.0",
            "stage": "production",
        }
        joblib.dump(bundle, MODEL_DIR / "pdm_model.joblib")
        with open(MODEL_DIR / "metadata.json", "w") as f:
            json.dump({"version": "1.0.0", "stage": "production", "metrics": bundle["metrics"]}, f, indent=2)
    print(f"Saved → {MODEL_DIR / 'pdm_model.joblib'}")
    return bundle

if __name__ == "__main__":
    train()
