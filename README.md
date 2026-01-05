# Predicting Branch Wait Times from Transaction Volumes

This project examines how daily transaction volumes and operational conditions influence customer wait times across a network of service branches. Using anonymized, aggregated data, the analysis focuses on a common real-world constraint: wait times must be predicted without direct information on staffing levels.

Because staffing data are unavailable, simple assumptions—such as wait times increasing linearly with transaction volume—do not hold. Instead, the project shows how wait times emerge from nonlinear demand–capacity dynamics, where small increases in volume can have little effect on some days but cause sharp increases once operational limits are reached.

The goal is to develop a practical forecasting approach that helps answer “what-if” questions, such as how population growth, policy changes, or shifts in customer behavior might impact wait times at different branches. By combining transaction volumes, calendar effects, and branch-level characteristics, the models provide actionable estimates of wait-time impacts even in the absence of staffing information.

---

## Problem Statment

Operational service networks often experience fluctuating customer demand, leading to unpredictable wait times. Understanding and forecasting these delays is critical for capacity planning, staffing decisions, and service-level management.
This project addresses the question:

To what extent can high-percentile daily wait times be predicted from transaction volume, seasonality, and branch-level characteristics?

---

## Data Description

- Unit of analysis: Branch-day (one row per branch per day)
- Target variable: 80th percentile daily wait time (p80_wait_min)
- Core predictors:
  - Daily transaction volume
  - Calendar seasonality (day-of-week, week-of-year)
  - Branch deployment model and visit mix
  - Branch identifier (anonymized)

⚠️ All identifiers, locations, and operational labels have been anonymized or transformed.
The data used in this repository do not represent real organizations or customers.

---

## Project Structure
```
wait-times-prediction/
│
├── notebooks/
│ ├── 01_eda_baseline.ipynb
│ └── 02_model_exploration.ipynb
│
├── data/
│ ├── raw/
│ └── processed/
│
├── outputs/ # Exported visualizations
│
├── README.md
├── LICENSE
└── .gitignore
```
---

### Exploratory Data Analysis (Notebook 01)

The EDA focuses on identifying why naive approaches fail rather than exhaustively profiling the data.

Key findings:
- Daily transaction volume is strongly associated with wait times, but
- Substantial variability exists at similar volume levels
- Branch-level effects and operational context materially influence delays
- High-percentile wait times are more stable and operationally meaningful than averages

These findings motivate the use of nonlinear models and robust targets.

---
### Modeling Approach (Notebook 02)

Baseline Model
- Model: Regularized linear regression
- Features: Volume, seasonality, categorical context
- Result: Captures broad trends but underfits nonlinear behavior

Nonlinear Model
- Model: Random Forest Regressor
- Enhancements:
  - Calendar seasonality features
  - Capacity proxies
  - Target encoding for branch identifiers to capture persistent site-level effects without leakage
- Validation: Time-based train/test split

---

### Key Takeaways

- High correlation does not imply high predictive accuracy
- Nonlinear interactions and site-specific baselines matter
- Leakage-aware target encoding can substantially improve performance
- Even strong models retain uncertainty due to unobserved constraints (e.g., staffing)


