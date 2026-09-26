# Executive Summary  
## Predictive Forecasting of Care Load & Placement Demand  
**UAC Program – U.S. Department of Health and Human Services**

---

### Purpose
The UAC Program operates in a high-uncertainty environment. Sudden changes in border activity or policy can rapidly increase the number of children entering federal care. This project delivers **forward-looking intelligence** so HHS decision-makers can:

- Anticipate how many children will be under HHS care in the coming days/weeks  
- Estimate whether discharge capacity will offset incoming transfers  
- Scale shelters, medical staff, and caseworkers **in advance**

### Key Results (Hold-out Test: last 60 days)

**Care Load (Children in HHS Care)**

| Model              | MAE   | RMSE  | MAPE  |
|--------------------|-------|-------|-------|
| **Random Forest**  | 12.8  | 16.1  | 0.68% |
| Gradient Boosting  | 13.3  | 16.1  | 0.71% |
| Naïve Persistence  | 13.5  | 16.3  | 0.72% |
| 7-day Moving Avg   | 13.7  | 17.0  | 0.73% |

**Discharge Demand**

| Model              | MAE  | RMSE | MAPE   |
|--------------------|------|------|--------|
| **Gradient Boosting** | 2.7 | 3.6  | 29%   |
| Random Forest      | 3.0  | 4.0  | 30%   |
| 7-day Moving Avg   | 3.3  | 4.2  | 32%   |

Random Forest is the recommended production model for Care Load forecasts (lowest error, stable).

### Operational Value
- **Surge Lead Time**: Models provide 7–30+ day advance visibility.  
- **Capacity Breach Probability**: Can be derived from forecast confidence bands.  
- **Forecast Stability**: Tree-based models remain robust under recent low-volatility conditions.

### Recommendations for Stakeholders
1. Deploy the Streamlit dashboard for daily operational review.  
2. Use Random Forest 14–30 day forecasts for staffing and bed planning.  
3. Monitor the 7-day change in Care Load + daily discharge rate as an early-warning composite indicator.  
4. Retrain models monthly (or after any major policy shift) using the latest ORR data.

### Deliverables Completed
- Cleaned daily time-series dataset (2021–2026)  
- Full forecasting pipeline (6 models)  
- Interactive Streamlit dashboard  
- Model metrics & forecast comparison files  
- This executive summary + technical documentation

---

*Educational / research project using publicly available HHS data. Not an official HHS operational system.*
