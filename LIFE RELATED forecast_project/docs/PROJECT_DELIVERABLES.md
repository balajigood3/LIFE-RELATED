# Complete Project Deliverables  
## Predictive Forecasting of Care Load & Placement Demand

---

### 1. Codebase

| Path | Description |
|------|-------------|
| `src/data_prep.py` | Data loading, interpolation, lag/rolling/calendar feature engineering |
| `src/models.py` | Naïve, Moving Average, ExpSmoothing, SARIMA, Random Forest, Gradient Boosting |
| `src/train_evaluate.py` | End-to-end training, evaluation, model persistence |
| `src/eda.py` | Exploratory analysis + plot generation |
| `app/streamlit_app.py` | Interactive dashboard (horizon, model toggle, CI, stress indicators) |
| `notebooks/01_full_pipeline.py` | Reproducible end-to-end script (notebook-style) |
| `requirements.txt` | Python dependencies |
| `run_dashboard.sh` | One-command dashboard launcher |
| `README.md` | Project overview & quick-start |

### 2. Data

| Path | Description |
|------|-------------|
| `data/uac_daily_cleaned.csv` | Continuous daily series (2021-01-11 → 2026-09-20) |

### 3. Trained Models

| Path | Description |
|------|-------------|
| `models/rf_care_load.joblib` | Production Random Forest for Children in HHS Care |
| `models/rf_discharge.joblib` | Random Forest for daily discharge demand |

### 4. Evaluation Outputs

| Path | Description |
|------|-------------|
| `outputs/model_metrics.csv` | Care Load MAE / RMSE / MAPE |
| `outputs/model_metrics_discharge.csv` | Discharge demand metrics |
| `outputs/forecast_vs_actual.csv` | 60-day hold-out predictions |
| `outputs/summary.json` | Machine-readable summary |
| `outputs/eda_*.png` | Six EDA visualizations |

### 5. Documentation

| Path | Description |
|------|-------------|
| `docs/Full_Research_Paper.md` | Complete research paper (Abstract → Conclusion) |
| `docs/Research_Paper_Outline.md` | Structured outline |
| `docs/Executive_Summary.md` | English executive summary for stakeholders |
| `docs/Executive_Summary_Tamil.md` | Tamil executive summary |
| `docs/PROJECT_DELIVERABLES.md` | This checklist |

### 6. Streamlit Dashboard Capabilities

- Future Care Load Forecast Chart  
- Discharge Demand Forecast Panel  
- Model Selection & Comparison  
- Approximate Confidence Interval bands  
- Forecast horizon slider (7–90 days)  
- Capacity stress indicators  
- Historical dual-axis view  

---

### How to Reproduce Everything

```bash
# 1. Install
pip install -r requirements.txt

# 2. EDA
python src/eda.py

# 3. Train & evaluate
python src/train_evaluate.py

# 4. Launch dashboard
streamlit run app/streamlit_app.py
# or
./run_dashboard.sh
```

---

### Submission Package Contents

All files above constitute the complete deliverable set required by the project specification:

1. Research paper (EDA, insights, recommendations)  
2. Streamlit dashboard (live analytics)  
3. Executive summary for government stakeholders (English + Tamil)  
4. Full reproducible code + trained models + metrics  

**Status: COMPLETE**
