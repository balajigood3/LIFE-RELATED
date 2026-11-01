# Edge-to-Cloud Real-Time Power Quality & Anomaly Detection Pipeline
### for High-Voltage GIS Substations

**Industrial AI / Predictive Maintenance Portfolio Project**

A complete pipeline that ingests high-frequency sensor streams from 220 kV / 110 kV GIS switchgear and power transformers, detects anomalies in real time using unsupervised models, and explains the root electrical parameters via SHAP.

---

## Key Capabilities

| Layer | Technology | Purpose |
|-------|------------|---------|
| Data Generation | Physics-informed synthetic sensors | Realistic voltage, current harmonics, temperature, SF6 pressure |
| ETL / Feature Pipeline | Pandas + rolling statistics | Windowed power-quality features |
| Anomaly Detection | Isolation Forest + optional LSTM Autoencoder | Unsupervised outlier detection |
| Explainability | SHAP (TreeExplainer) | Which electrical parameter drove the alert |
| Dashboard | Streamlit + Plotly | Real-time Asset Health Index & alerts |

---

## Architecture

```
Sensor Stream (simulated) 
        │
        ▼
┌───────────────────┐
│  Feature Engine   │  → rolling RMS, THD, imbalance, thermal gradients
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Anomaly Detector  │  Isolation Forest (primary) / LSTM-AE (optional)
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  XAI Layer (SHAP) │  Feature contribution → root-cause hints
└─────────┬─────────┘
          │
          ▼
   Streamlit Dashboard
   • Asset Health Index
   • Live anomaly flags
   • SHAP force / waterfall plots
```

---

## Quick Start

```bash
cd gis-anomaly-detection
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Generate synthetic data + train models
python -m app.pipeline.train

# Launch dashboard
streamlit run app/dashboard/app.py
```

---

## Project Structure

```
gis-anomaly-detection/
├── app/
│   ├── data/               # Synthetic data generator
│   ├── pipeline/           # Feature engineering + training
│   ├── models/             # Isolation Forest / LSTM wrappers
│   ├── dashboard/          # Streamlit UI
│   └── utils/
├── data/
│   ├── raw/                # Generated sensor streams
│   ├── processed/          # Feature matrices
│   └── models/             # Saved .joblib / .pt files
├── notebooks/
├── tests/
├── requirements.txt
└── README.md
```

---

## Industrial Context (Why this matters)

- GIS (Gas-Insulated Switchgear) failures are rare but catastrophic (long outages, high repair cost, SF6 environmental impact).
- Early detection of insulation degradation, contact overheating, or SF6 leakage via power-quality + thermal signatures is high-value.
- Explainable alerts are essential for operator trust and root-cause analysis.

---

## Disclaimer

This is a **portfolio / demonstration system** using synthetic data.  
It is not certified for live substation protection or control.
