# Predictive Forecasting of Care Load & Placement Demand  
## A Time-Series and Machine-Learning Approach for the UAC Program

**Author:** Unified Mentor Capstone Project  
**Organization Context:** U.S. Department of Health and Human Services – Office of Refugee Resettlement (ORR)  
**Date:** September 2026

---

## Abstract

The Unaccompanied Alien Children (UAC) Program operates under high uncertainty. Sudden shifts in border activity, enforcement policy, or humanitarian conditions can rapidly change the number of children requiring federal care. While descriptive reporting explains the past, decision-makers need forward-looking intelligence. This project develops short-term forecasting models for (1) the number of children in HHS care and (2) daily discharge (placement) demand. Using the official public HHS daily time-series (2021–2026), we implement baseline, statistical, and machine-learning models under strict chronological validation. Random Forest achieves the lowest error on Care Load (MAE ≈ 13 children, MAPE < 1%). The accompanying Streamlit dashboard enables interactive horizon selection, model comparison, and approximate uncertainty visualization, supporting proactive rather than reactive capacity planning.

---

## 1. Introduction and Background

The UAC Program places unaccompanied children who lack lawful immigration status and a parent/guardian available to provide care into the least restrictive setting that serves the child’s best interest. Daily volumes of referrals from Customs and Border Protection (CBP) and subsequent discharges to vetted sponsors fluctuate substantially.  

Descriptive analytics answer “what happened.” Operational leaders additionally need answers to:

- How many children will be under HHS care in the next 7–30 days?  
- Will discharge capacity offset incoming transfers?  
- When should shelters, medical staff, and caseworkers be scaled up in advance?

This project introduces predictive modeling to convert the existing high-quality daily time series into actionable foresight.

---

## 2. Problem Statement

Despite the availability of daily operational data, the program currently lacks:

- Short-term forecasts of the in-care population  
- Predictive estimates of placement (discharge) demand  
- Early-warning indicators of capacity stress  

Consequently, responses remain largely reactive, elevating risks of overcrowding, staff burnout, and prolonged length of stay for children.

---

## 3. Objectives

**Primary**  
1. Forecast the number of children in HHS care.  
2. Estimate future imbalance between intake and exits.  
3. Predict short-term discharge demand.

**Secondary**  
4. Provide early-warning signals for healthcare and logistics planners.  
5. Quantify forecast uncertainty.  
6. Compare statistical versus machine-learning forecasting approaches.

---

## 4. Dataset Description

**Source:** HHS Data Hub – “HHS Unaccompanied Alien Children Program”  
(https://healthdata.gov/National/HHS-Unaccompanied-Alien-Children-Program/ehpz-xc9n)

| Column | Description |
|--------|-------------|
| Date | Reporting date |
| Children apprehended and placed in CBP custody | Daily intake volume |
| Children in CBP custody | Active CBP care load |
| Children transferred out of CBP custody | Flow into HHS system |
| Children in HHS Care | Active HHS care load (primary target) |
| Children discharged from HHS Care | Successful sponsor placements (secondary target) |

**Coverage:** 11 January 2021 – 20 September 2026  
**Pre-processing:**  
- Conversion to continuous daily index  
- Linear interpolation of the two primary series to fill reporting gaps  
- CBP-related columns retained where available (substantial missingness in early years)

---

## 5. Analytical Methodology

### 5.1 Time-Series Preparation
- Datetime indexing and chronological sorting  
- Continuity enforcement via re-indexing to daily frequency  
- Interpolation of primary targets only  

### 5.2 Feature Engineering
- **Lag features:** t-1, t-7, t-14, t-30  
- **Rolling statistics:** 7-day and 14-day mean and standard deviation  
- **Calendar effects:** day-of-week, month, day-of-month, weekend flag, quarter  
- **Flow signal:** Transfers − Discharges (net pressure) when both series are observed  

### 5.3 Train–Test Strategy
- Strict time-based split (no random sampling)  
- Hold-out = final 60 days  
- Walk-forward style evaluation via recursive multi-step prediction for ML models  

### 5.4 Forecasting Models

| Category | Models |
|----------|--------|
| Baseline | Naïve Persistence, 7-day Moving Average |
| Statistical | Exponential Smoothing (additive trend + weekly seasonality), SARIMA |
| Machine Learning | Random Forest Regressor, Gradient Boosting Regressor |

### 5.5 Evaluation Metrics
- MAE – absolute accuracy  
- RMSE – penalizes large errors  
- MAPE – relative error understanding  

---

## 6. Results

### 6.1 Care Load (Children in HHS Care) – 60-day hold-out

| Model              | MAE   | RMSE  | MAPE (%) |
|--------------------|-------|-------|----------|
| **Random Forest**  | 12.8  | 16.1  | 0.68     |
| Gradient Boosting  | 13.3  | 16.1  | 0.71     |
| Naïve              | 13.5  | 16.3  | 0.72     |
| Moving Average (7) | 13.7  | 17.0  | 0.73     |
| Exp. Smoothing     | 47.0  | 56.3  | 2.52     |
| SARIMA             | 91.3  | 103.7 | 4.89     |

Random Forest is the strongest performer. Under the recent relatively smooth regime, even simple persistence is competitive; tree-based models still extract additional weekly structure.

### 6.2 Discharge Demand

| Model              | MAE  | RMSE | MAPE (%) |
|--------------------|------|------|----------|
| **Gradient Boosting** | 2.7 | 3.6  | 29.0     |
| Random Forest      | 3.0  | 4.0  | 30.1     |
| Moving Average (7) | 3.3  | 4.2  | 32.1     |
| SARIMA             | 3.8  | 4.8  | 30.4     |
| Naïve              | 4.2  | 5.3  | 34.3     |
| Exp. Smoothing     | 7.1  | 8.1  | 54.3     |

Daily discharges are inherently noisier; relative errors are higher, yet absolute errors remain small (≈ 3 children per day).

---

## 7. Discussion and Insights

1. **Low recent volatility** of the Care Load series makes naïve and short-window averages already accurate. Machine-learning models still improve on them by capturing multi-lag interactions and calendar effects.  
2. **Statistical models** (especially SARIMA with the chosen order) under-performed on the hold-out window; richer seasonal specifications or rolling re-estimation may improve them.  
3. **Discharge series** exhibits higher day-to-day variance; ensemble methods remain the preferred choice.  
4. **Operational implication:** A 14–30 day Random Forest forecast of Care Load, combined with the observed 7-day discharge rate, supplies a practical early-warning composite for capacity stress.  
5. **Uncertainty:** Approximate residual-based confidence bands widen with horizon and provide planners with a transparent range rather than a point estimate.

---

## 8. Streamlit Web Application

Core modules implemented:

- Future Care Load Forecast Chart  
- Discharge Demand Forecast Panel  
- Model Selection & Comparison  
- Approximate Confidence Interval Visualization  
- Horizon selector (7–90 days)  
- Capacity stress indicators (current load, 7-day change, stress level)

The dashboard is designed for non-technical stakeholders while exposing full model comparison for analysts.

---

## 9. Recommendations

1. **Adopt Random Forest** as the primary Care Load forecasting engine for 14–30 day horizons.  
2. **Monitor a composite KPI**: 7-day Care Load change + 7-day average discharges.  
3. **Retrain monthly** (or after any major policy or border-flow regime change).  
4. **Extend** future work to hierarchical forecasts (age, gender, facility type) when richer micro-data become available.  
5. **Integrate** the dashboard into existing operational review cadence.

---

## 10. Conclusion

This project elevates the public UAC dataset from historical reporting to predictive intelligence. By combining rigorous time-series preparation, classical statistical models, and modern ensemble machine learning under strict chronological validation, it supplies HHS stakeholders with the foresight needed to allocate resources proactively and strengthen child-welfare outcomes.

---

## Deliverables Checklist

- [x] Cleaned daily time-series dataset  
- [x] Full feature-engineering and modeling pipeline  
- [x] Six forecasting models with quantitative evaluation  
- [x] Interactive Streamlit dashboard  
- [x] Executive summary (English + Tamil)  
- [x] Complete research paper  
- [x] Reproducible code repository  

---

*Data source: publicly released HHS/ORR statistics. This work is an educational and research demonstration and is not an official operational system of the U.S. Department of Health and Human Services.*
