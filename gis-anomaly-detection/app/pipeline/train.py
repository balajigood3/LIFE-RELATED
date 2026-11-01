"""
End-to-end training pipeline:
1. Generate (or load) synthetic GIS sensor data
2. Engineer features
3. Train Isolation Forest
4. (Optional) Train LSTM Autoencoder
5. Evaluate against injected labels
6. Persist models + metrics
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from app.data.generator import generate_dataset, save_raw_dataset
from app.pipeline.features import build_feature_frame, get_feature_matrix, FEATURE_COLUMNS
from app.models.anomaly import IsolationForestDetector, LSTMAutoencoderDetector, TORCH_AVAILABLE

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "gis_sensor_stream.parquet"
FEAT_PATH = ROOT / "data" / "processed" / "features.parquet"
IF_MODEL_PATH = ROOT / "data" / "models" / "isolation_forest.joblib"
METRICS_PATH = ROOT / "data" / "models" / "metrics.json"


def run_training(use_lstm: bool = False, hours: float = 48) -> dict:
    print("=" * 60)
    print("GIS Anomaly Detection – Training Pipeline")
    print("=" * 60)

    # 1. Data
    if RAW_PATH.exists():
        print(f"[1/5] Loading existing data from {RAW_PATH}")
        df = pd.read_parquet(RAW_PATH)
    else:
        print(f"[1/5] Generating {hours}h synthetic sensor stream...")
        df = generate_dataset(hours=hours, sample_rate_hz=1.0)
        save_raw_dataset(df, RAW_PATH)
        print(f"      Saved {len(df):,} rows")

    print(f"      Anomaly distribution:\n{df['anomaly_type'].value_counts().to_string()}")

    # 2. Features
    print("\n[2/5] Engineering features...")
    feat_df = build_feature_frame(df, window=30)
    FEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_parquet(FEAT_PATH, index=False)
    X, feature_names = get_feature_matrix(feat_df)
    y = feat_df["label"].values
    print(f"      Feature matrix: {X.shape}")

    # 3. Isolation Forest
    print("\n[3/5] Training Isolation Forest...")
    if_det = IsolationForestDetector(contamination=0.05, n_estimators=200)
    if_det.fit(X, feature_names)
    if_det.save(IF_MODEL_PATH)
    print(f"      Saved → {IF_MODEL_PATH}")

    preds_if = if_det.predict(X)
    scores_if = if_det.anomaly_score(X)

    # 4. Optional LSTM
    preds_lstm = None
    if use_lstm and TORCH_AVAILABLE and LSTMAutoencoderDetector is not None:
        print("\n[4/5] Training LSTM Autoencoder (this may take a minute)...")
        lstm_det = LSTMAutoencoderDetector(seq_len=30, epochs=25, device="cpu")
        # Train only on normal data for proper autoencoder behaviour
        X_normal = X[y == 0]
        lstm_det.fit(X_normal, feature_names)
        preds_lstm = lstm_det.predict(X)
        lstm_path = ROOT / "data" / "models" / "lstm_ae.pt"
        # Simple save of threshold + scaler; full model state can be added later
        import joblib
        joblib.dump({"threshold": lstm_det.threshold, "scaler": lstm_det.scaler,
                     "feature_names": feature_names, "seq_len": lstm_det.seq_len},
                    lstm_path.with_suffix(".joblib"))
        print(f"      LSTM threshold = {lstm_det.threshold:.6f}")
    else:
        print("\n[4/5] Skipping LSTM (set use_lstm=True and ensure torch is installed)")

    # 5. Evaluation (against synthetic labels)
    print("\n[5/5] Evaluation against injected anomaly labels")
    metrics = {
        "isolation_forest": {
            "precision": float(precision_score(y, preds_if, zero_division=0)),
            "recall": float(recall_score(y, preds_if, zero_division=0)),
            "f1": float(f1_score(y, preds_if, zero_division=0)),
            "n_predicted_anomalies": int(preds_if.sum()),
            "n_true_anomalies": int(y.sum()),
        }
    }
    print(f"      Isolation Forest  P={metrics['isolation_forest']['precision']:.3f}  "
          f"R={metrics['isolation_forest']['recall']:.3f}  "
          f"F1={metrics['isolation_forest']['f1']:.3f}")

    if preds_lstm is not None:
        metrics["lstm_ae"] = {
            "precision": float(precision_score(y, preds_lstm, zero_division=0)),
            "recall": float(recall_score(y, preds_lstm, zero_division=0)),
            "f1": float(f1_score(y, preds_lstm, zero_division=0)),
            "n_predicted_anomalies": int(preds_lstm.sum()),
        }
        print(f"      LSTM Autoencoder  P={metrics['lstm_ae']['precision']:.3f}  "
              f"R={metrics['lstm_ae']['recall']:.3f}  "
              f"F1={metrics['lstm_ae']['f1']:.3f}")

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved → {METRICS_PATH}")
    print("=" * 60)
    return metrics


if __name__ == "__main__":
    run_training(use_lstm=False)
