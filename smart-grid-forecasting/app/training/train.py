"""
Training pipeline with MLflow tracking for industrial load forecasting.
Trains XGBoost, LightGBM, and CatBoost; logs metrics and registers the best model.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import joblib

from app.data.generator import generate_industrial_load, save_dataset
from app.training.features import build_supervised, FEATURE_COLS

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "industrial_load.parquet"
MODEL_DIR = ROOT / "data" / "models"
MLRUNS = ROOT / "mlruns"


def metrics(y_true, y_pred) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-3, None))) * 100
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape), "r2": float(r2)}


def train_models(days: int = 60) -> Dict[str, Any]:
    print("=" * 60)
    print("Smart Grid Load Forecasting – Training + MLflow")
    print("=" * 60)

    # Data
    if RAW_PATH.exists():
        print(f"[1/4] Loading data from {RAW_PATH}")
        df = pd.read_parquet(RAW_PATH)
    else:
        print(f"[1/4] Generating {days} days of synthetic load...")
        df = generate_industrial_load(days=days)
        save_dataset(df, RAW_PATH)

    print(f"      Rows: {len(df):,}")

    # Features
    print("[2/4] Building supervised features...")
    feat_df, feature_names = build_supervised(df)
    X = feat_df[feature_names]
    y = feat_df["target"]

    # Time-based split (last 20% as test)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    print(f"      Train: {len(X_train):,}  Test: {len(X_test):,}")

    # MLflow setup
    mlflow.set_tracking_uri(f"file://{MLRUNS.resolve()}")
    mlflow.set_experiment("industrial_load_forecasting")

    candidates = {
        "xgboost": XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1,
        ),
        "catboost": CatBoostRegressor(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            random_seed=42,
            verbose=0,
        ),
    }

    results = {}
    best_name = None
    best_rmse = float("inf")
    best_model = None

    print("[3/4] Training models with MLflow tracking...")
    for name, model in candidates.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            m = metrics(y_test.values, preds)

            mlflow.log_params({
                "model_type": name,
                "n_features": len(feature_names),
                "train_rows": len(X_train),
            })
            mlflow.log_metrics(m)

            # Log model
            if name == "xgboost":
                mlflow.xgboost.log_model(model, "model")
            elif name == "lightgbm":
                mlflow.lightgbm.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")

            print(f"      {name:12s}  MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  "
                  f"MAPE={m['mape']:.2f}%  R2={m['r2']:.4f}")

            results[name] = m
            if m["rmse"] < best_rmse:
                best_rmse = m["rmse"]
                best_name = name
                best_model = model

    # Persist best model for serving
    print(f"\n[4/4] Best model: {best_name} (RMSE={best_rmse:.3f})")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "feature_names": feature_names,
            "model_name": best_name,
            "metrics": results[best_name],
        },
        MODEL_DIR / "best_model.joblib",
    )

    # Also save a small metadata file
    meta = {
        "best_model": best_name,
        "metrics": results,
        "feature_names": feature_names,
    }
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"      Saved → {MODEL_DIR / 'best_model.joblib'}")
    print("=" * 60)
    return meta


if __name__ == "__main__":
    train_models()
