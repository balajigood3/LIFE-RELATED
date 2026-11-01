"""
Anomaly detection models for GIS sensor streams.

Primary: Isolation Forest (fast, robust, easy SHAP)
Optional: LSTM Autoencoder (for sequential patterns)
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class IsolationForestDetector:
    """Production-friendly unsupervised detector."""

    def __init__(
        self,
        contamination: float = 0.04,
        n_estimators: int = 200,
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
            verbose=0,
        )
        self.feature_names: list[str] = []
        self.is_fitted = False

    def fit(self, X: np.ndarray, feature_names: list[str]) -> "IsolationForestDetector":
        self.feature_names = feature_names
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns 1 for anomaly, 0 for normal (sklearn convention inverted)."""
        Xs = self.scaler.transform(X)
        raw = self.model.predict(Xs)          # 1 = normal, -1 = anomaly
        return (raw == -1).astype(int)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Lower (more negative) = more anomalous."""
        Xs = self.scaler.transform(X)
        return self.model.decision_function(Xs)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Higher score = more anomalous (0–1 range approx)."""
        df = self.decision_function(X)
        # Invert and min-max to [0, 1] for interpretability
        scores = -df
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        return scores

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "contamination": self.contamination,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "IsolationForestDetector":
        data = joblib.load(path)
        obj = cls(contamination=data["contamination"])
        obj.model = data["model"]
        obj.scaler = data["scaler"]
        obj.feature_names = data["feature_names"]
        obj.is_fitted = True
        return obj


# ---------------------------------------------------------------------------
# Optional LSTM Autoencoder (PyTorch) – used when sequential context matters
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class LSTMAutoencoder(nn.Module):
        def __init__(self, n_features: int, latent_dim: int = 16, hidden: int = 64):
            super().__init__()
            self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
            self.latent = nn.Linear(hidden, latent_dim)
            self.decoder_input = nn.Linear(latent_dim, hidden)
            self.decoder = nn.LSTM(hidden, n_features, batch_first=True)

        def forward(self, x):
            # x: (batch, seq, features)
            _, (h, _) = self.encoder(x)
            z = self.latent(h[-1])
            dec_in = self.decoder_input(z).unsqueeze(1).repeat(1, x.size(1), 1)
            out, _ = self.decoder(dec_in)
            return out

    class LSTMAutoencoderDetector:
        """Sequence-aware anomaly detector using reconstruction error."""

        def __init__(
            self,
            seq_len: int = 30,
            latent_dim: int = 16,
            epochs: int = 30,
            lr: float = 1e-3,
            device: str = "cpu",
        ):
            self.seq_len = seq_len
            self.latent_dim = latent_dim
            self.epochs = epochs
            self.lr = lr
            self.device = device
            self.scaler = StandardScaler()
            self.model: Optional[LSTMAutoencoder] = None
            self.threshold: float = 0.0
            self.feature_names: list[str] = []

        def _create_sequences(self, X: np.ndarray) -> np.ndarray:
            seqs = []
            for i in range(len(X) - self.seq_len + 1):
                seqs.append(X[i : i + self.seq_len])
            return np.array(seqs, dtype=np.float32)

        def fit(self, X: np.ndarray, feature_names: list[str]) -> "LSTMAutoencoderDetector":
            self.feature_names = feature_names
            Xs = self.scaler.fit_transform(X)
            seqs = self._create_sequences(Xs)

            self.model = LSTMAutoencoder(n_features=X.shape[1], latent_dim=self.latent_dim)
            self.model.to(self.device)
            opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            loss_fn = nn.MSELoss()

            dataset = torch.utils.data.TensorDataset(torch.from_numpy(seqs))
            loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

            self.model.train()
            for epoch in range(self.epochs):
                total = 0.0
                for (batch,) in loader:
                    batch = batch.to(self.device)
                    opt.zero_grad()
                    recon = self.model(batch)
                    loss = loss_fn(recon, batch)
                    loss.backward()
                    opt.step()
                    total += loss.item()
                if (epoch + 1) % 10 == 0:
                    print(f"  LSTM-AE epoch {epoch+1}/{self.epochs}  loss={total/len(loader):.6f}")

            # Set threshold at 99th percentile of training reconstruction error
            self.model.eval()
            with torch.no_grad():
                recon = self.model(torch.from_numpy(seqs).to(self.device)).cpu().numpy()
            errors = np.mean((seqs - recon) ** 2, axis=(1, 2))
            self.threshold = float(np.percentile(errors, 99))
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            scores = self.anomaly_score(X)
            # Align length: first seq_len-1 points get score 0
            preds = (scores >= self.threshold).astype(int)
            return preds

        def anomaly_score(self, X: np.ndarray) -> np.ndarray:
            Xs = self.scaler.transform(X)
            seqs = self._create_sequences(Xs)
            self.model.eval()
            with torch.no_grad():
                recon = self.model(torch.from_numpy(seqs).to(self.device)).cpu().numpy()
            errors = np.mean((seqs - recon) ** 2, axis=(1, 2))
            # Pad front to match original length
            pad = np.zeros(self.seq_len - 1)
            return np.concatenate([pad, errors])

else:
    LSTMAutoencoderDetector = None  # type: ignore
