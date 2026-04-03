# Branch Wait-Time Prediction

Predicting daily p80 customer wait times across a network of 92 bank branches using transaction-level visit data, time-series feature engineering, and gradient boosting. The primary output is a 30-day branch-level alert report that flags locations forecast to exceed a 25-minute service target before the period opens.

---

## Background

Bank branches handle a mix of appointment-based and walk-in visits across Internal and External deployment models. Wait times vary substantially by branch, day of week, transaction volume, and visit type - and long queues directly affect customer satisfaction scores and teller retention. When pressure builds at a location, adjustments to staffing or appointment capacity typically happen reactively - after the queues form - because operations planners lack forward-looking wait-time signals.

This project builds a predictive pipeline that produces a ranked weekly alert report: which branches are forecast to breach their service-level threshold in the next 30 working days, and by how much. High-risk branches can be flagged for proactive staffing review before the forecast window opens rather than after complaints arrive.

---

## Dataset

| Field | Detail |
|---|---|
| Source | Anonymized transaction-level branch visit log |
| Raw records | ~2.7 million visits |
| Date range | January 2024 - November 2025 |
| Branches | 92 locations across 3 regions (Central, East, West) |
| Deployment models | Internal and External |
| Key fields | Branch ID, Region, Site Type, Visit Type, Arrival Time, Call Time, End Treatment Time |
| Processed records | ~16,800 daily branch-day aggregates (after removing low-activity days) |
| Target variable | p80 wait time (minutes) - 80th percentile of customer wait per branch per day |

Raw and processed data files are excluded from this repository. See `.gitignore` and the note on data privacy below.

---

## Pipeline

```
data/raw/visits_2024_2025_adj.csv
         |
         v
+-- 01_eda_baseline.ipynb
|       Parse timestamps, derive wait times, aggregate to daily branch level,
|       explore volume-wait relationships, temporal patterns, regional variation
|       -> data/processed/daily_branch_waits.csv
|
+-- 02_feature_engineering.ipynb
|       Lag features (lag-1 p80, lag-1 volume), 5-day rolling averages,
|       cyclical calendar encoding, smoothed branch target encoding,
|       time-based 80/20 train-test split, Ridge and Random Forest baselines
|       -> data/processed/daily_branch_features.csv
|       -> models/rf_wait_time_pipeline.joblib
|
+-- 03_model_comparison.ipynb
|       TimeSeriesSplit cross-validation across Ridge, Random Forest,
|       and Gradient Boosting; residual analysis by volume band; SHAP
|       feature contributions
|       -> models/gbm_wait_time_pipeline.joblib
|
+-- 04_forecasting.ipynb
        Recursive 30-day branch-level forecast, horizon-stratified
        uncertainty bands, network-wide alert tiering
        -> outputs/tables/04_branch_alert_report.csv
```

---

## Notebooks

| Notebook | Description |
|---|---|
| `01_eda_baseline.ipynb` | Data preparation, wait-time distribution, volume-wait relationship, temporal heatmap, regional and visit-type breakdowns |
| `02_feature_engineering.ipynb` | Lag and rolling features, branch target encoding, train-test split, Ridge and Random Forest baseline models |
| `03_model_comparison.ipynb` | Time-series cross-validation, three-model comparison, residual analysis by volume band, SHAP explainability |
| `04_forecasting.ipynb` | Recursive 30-day forecast, confidence bands, network-wide alert report |

---

## Methods summary

**Feature engineering:** Lag-1 and 5-day rolling mean of both p80 wait and daily volume provide the autoregressive signal that dominates model performance. Day-of-week is encoded as sine-cosine pairs to preserve cyclical continuity. Branch identity is represented via smoothed target encoding fit strictly on the training split to prevent leakage.

**Train-test split:** Chronological 80/20 split on unique dates (cutoff 2025-07-16). All lag features use `.shift(1)` to ensure no future information enters any training observation.

**Model selection:** TimeSeriesSplit cross-validation (5 folds) on the training period. Ridge, Random Forest, and Gradient Boosting achieve comparable cross-validated R² (~0.80-0.81); Random Forest holds a marginal edge on MAE.

**Forecasting:** Recursive single-step prediction using median historical volume per day of week as the volume input. Uncertainty bands are 80% confidence intervals derived from horizon-stratified test-set residuals.

---

## Key results

| Metric | Value |
|---|---|
| Best test R² (Random Forest) | 0.73 |
| Best test MAE (Random Forest) | 6.00 min |
| Top permutation feature | 5-day rolling p80 wait (importance 0.91) |
| High-risk branches (30-day horizon) | 5 branches - #149, #003, #044, #076 (Central), #206 (East) |
| Forecast MAE at 1-7 day horizon | ~5.5 min |
| Forecast MAE at 61-120 day horizon | ~7.1 min |
| Walk-in vs appointment wait gap | 22 min vs 3 min mean p80 |

---

## Repo structure

```
wait-times-prediction/
+-- notebooks/
|   +-- 01_eda_baseline.ipynb
|   +-- 02_feature_engineering.ipynb
|   +-- 03_model_comparison.ipynb
|   +-- 04_forecasting.ipynb
+-- data/
|   +-- raw/               # excluded from git - see .gitignore
|   |   +-- .gitkeep
|   +-- processed/         # excluded from git - see .gitignore
|       +-- .gitkeep
+-- models/                # excluded from git - see .gitignore
|   +-- .gitkeep
+-- outputs/
|   +-- figures/           # excluded from git - see .gitignore
|   |   +-- .gitkeep
|   +-- tables/            # excluded from git - see .gitignore
|       +-- .gitkeep
+-- tweak_raw_data.py      # excluded from git - local data prep only
+-- requirements.txt
+-- .gitignore
+-- README.md
```

---

## Dependencies

Python 3.10+

```
pandas>=2.0
numpy
matplotlib
scikit-learn
joblib
shap
```

Install with:

```bash
pip install -r requirements.txt
```

---

## How to run

1. Place the raw visit log at `data/raw/visits_2024_2025.csv`
2. Run `tweak_raw_data.py` to produce `data/raw/visits_2024_2025_adj.csv`
3. Run notebooks in order: 01 - 02 - 03 - 04
4. Alert report is written to `outputs/tables/04_branch_alert_report.csv`

Each notebook reads from `data/processed/` and writes its outputs there or to `outputs/`. No notebook modifies a file produced by a later notebook.

---

## Data privacy

The raw transaction file contains anonymized branch and visit records from a financial services network and is not included in this repository. Branch identifiers have been replaced with numeric codes. No personally identifiable customer information is present in any file. The `tweak_raw_data.py` script used to prepare the modeling dataset is also excluded from the public repository.

---

## Author

Caio Correa
