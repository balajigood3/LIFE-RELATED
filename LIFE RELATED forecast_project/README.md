# Predictive Forecasting of Care Load & Placement Demand

**UAC Program · U.S. Department of Health and Human Services**  
Unified Mentor Capstone Project

## Project Overview

This project builds short-term forecasting models for:

1. **Number of children in HHS care** (Care Load)
2. **Daily discharge / placement demand**

It enables proactive capacity planning instead of reactive responses.

## Dataset

Official public dataset from HHS Data Hub:

- **HHS Unaccompanied Alien Children Program**  
- Columns: Date, Children apprehended (CBP), Children in CBP custody, Transfers to HHS, Children in HHS Care, Children discharged from HHS  
- Period: 2021-01-11 → 2026-09-20 (daily, interpolated for continuity)

Source: https://healthdata.gov/National/HHS-Unaccompanied-Alien-Children-Program/ehpz-xc9n

## Project Structure

```
uac_forecast_project/
├── data/
│   └── uac_daily_cleaned.csv
├── src/
│   ├── data_prep.py          # Loading, lag/rolling/calendar features
│   ├── models.py             # Naïve, MA, ExpSmoothing, SARIMA, RF, GBM
│   └── train_evaluate.py     # Walk-forward style evaluation
├── app/
│   └── streamlit_app.py      # Interactive dashboard
├── models/                   # Saved joblib models
├── outputs/                  # Metrics & forecast CSVs
├── docs/                     # Research paper & executive summary
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models & generate metrics
cd src
python train_evaluate.py

# 3. Launch dashboard
cd ../app
streamlit run streamlit_app.py
```

## Models Implemented

| Category            | Models                              |
|---------------------|-------------------------------------|
| Baseline            | Naïve Persistence, 7-day Moving Avg |
| Statistical         | Exponential Smoothing, SARIMA       |
| Machine Learning    | Random Forest, Gradient Boosting    |

**Evaluation metrics**: MAE, RMSE, MAPE  
**Validation**: Strict time-based hold-out (last 60 days)

## Streamlit Features

- Future Care Load Forecast Chart
- Discharge Demand view
- Model selection / comparison
- Approximate confidence intervals
- Horizon slider (7–90 days)
- Capacity stress indicators

## Deliverables

1. Research-style documentation (EDA + insights + recommendations)
2. Live Streamlit dashboard
3. Executive summary for stakeholders

## Author Notes

- Feature engineering follows the project specification (lags 1/7/14/30, rolling means/std, calendar effects, net pressure).
- Random Forest is the default production model for the dashboard because it balances accuracy and stability.
- CBP intake columns contain substantial missing values; primary forecasting focuses on the complete HHS Care & Discharge series.

---

*Educational project – not an official HHS operational tool.*
