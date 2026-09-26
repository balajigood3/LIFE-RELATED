# Research Paper Outline  
## Predictive Forecasting of Care Load & Placement Demand for the UAC Program

### 1. Introduction & Background
- High-uncertainty environment of the Unaccompanied Alien Children (UAC) Program  
- Limitations of purely descriptive analytics  
- Need for proactive capacity planning (shelters, medical staff, caseworkers)

### 2. Problem Statement
- Lack of short-term forecasts of children in HHS care  
- Lack of predictive estimates of discharge (placement) demand  
- Delayed operational responses → overcrowding risk, staff burnout, longer length of stay

### 3. Objectives
**Primary**  
- Forecast number of children in HHS care  
- Estimate future imbalance between intake and exits  
- Predict short-term discharge demand  

**Secondary**  
- Early-warning indicators for planners  
- Quantify forecast uncertainty  
- Compare statistical vs machine-learning approaches

### 4. Dataset
- Source: HHS Data Hub – Unaccompanied Alien Children Program  
- Period: 11 Jan 2021 – 20 Sep 2026  
- Key series: Children in HHS Care, Children discharged from HHS Care  
- Pre-processing: continuous daily index, linear interpolation of primary targets, lag/rolling/calendar features

### 5. Methodology
- Time-series preparation & feature engineering (lags 1/7/14/30, rolling mean/std 7/14, day-of-week, month, net pressure)  
- Strict chronological train/test split (last 60 days hold-out)  
- Models:  
  - Baseline: Naïve, Moving Average  
  - Statistical: Exponential Smoothing, SARIMA  
  - Machine Learning: Random Forest, Gradient Boosting  
- Metrics: MAE, RMSE, MAPE

### 6. Results
(Insert tables from `outputs/model_metrics.csv` and `model_metrics_discharge.csv`)

- Random Forest achieved the lowest error on Care Load (MAE ≈ 13 children, MAPE < 1%).  
- For daily discharges, Gradient Boosting / Random Forest slightly outperformed simple baselines.  
- Statistical models (especially SARIMA) struggled under the recent low-volatility regime.

### 7. Discussion & Insights
- Recent Care Load series is relatively smooth → persistence and short-window averages already strong.  
- Tree-based models still add value by capturing weekly seasonality and multi-lag interactions.  
- Discharge series is noisier → higher relative error; still useful for capacity stress monitoring.  
- Confidence intervals (approximate residual-based) provide practical uncertainty bands for planners.

### 8. Streamlit Application
- Live Care Load & Discharge forecast charts  
- Model toggle & horizon selector  
- Approximate 95% confidence bands  
- Capacity stress indicators

### 9. Recommendations
1. Operationalize Random Forest for 14–30 day Care Load forecasts.  
2. Combine forecast with real-time discharge rate for early-warning composite KPI.  
3. Retrain monthly and after major policy changes.  
4. Extend future work to hierarchical forecasts (by age/gender/facility type) when richer data become available.

### 10. Conclusion
The project successfully elevates the public UAC dataset from historical reporting to predictive intelligence, enabling data-driven foresight for child-welfare capacity planning.

### References
- HHS / ORR public data releases  
- Project specification (Unified Mentor)  
- Classic time-series & ensemble forecasting literature
