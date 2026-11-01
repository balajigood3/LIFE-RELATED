import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.data.generator import generate_dataset
from app.pipeline.features import build_feature_frame, get_feature_matrix
from app.models.anomaly import IsolationForestDetector

def test_generator_and_features():
    df = generate_dataset(hours=1, sample_rate_hz=1.0, seed=0)
    assert len(df) > 100
    assert "label" in df.columns
    feat = build_feature_frame(df, window=10)
    X, names = get_feature_matrix(feat)
    assert X.shape[1] == len(names)
    assert X.shape[0] > 50

def test_isolation_forest_fit_predict():
    df = generate_dataset(hours=2, sample_rate_hz=1.0, seed=1)
    feat = build_feature_frame(df, window=15)
    X, names = get_feature_matrix(feat)
    det = IsolationForestDetector(contamination=0.05, n_estimators=50)
    det.fit(X, names)
    preds = det.predict(X)
    scores = det.anomaly_score(X)
    assert len(preds) == len(X)
    assert scores.min() >= 0 and scores.max() <= 1.01
