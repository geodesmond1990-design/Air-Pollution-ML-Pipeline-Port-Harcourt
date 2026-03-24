# Geospatial Machine Learning for Air Pollution Prediction
### Multivariate Environmental Data — Port Harcourt, Nigeria (2023–2025)

> **Paper:** *Geospatial Machine Learning for Air Pollution Prediction Using Multivariate Environmental Data: A Spatiotemporal Analysis of Port Harcourt, Nigeria*  
> **Target Journal:** Environmental Science and Pollution Research (Springer)

---

## Overview

This repository contains the full reproducible analysis pipeline for predicting air pollutant concentrations (CO₂, N₂O, CH₄, O₃, CO) across 48 monitoring stations in Port Harcourt, Rivers State, Nigeria, using geospatial, temporal, and meteorological features.

**Dataset:** 576 station-month observations (48 stations × 12 months: August 2023 – January 2025)

---

## Repository Structure

```
air_pollution_ml/
│
├── main.py              # Master pipeline — run this
├── data_loader.py       # Data ingestion, coordinate fixing, feature engineering
├── eda.py               # Descriptive statistics, spatial & temporal summaries
├── stats_tests.py       # Shapiro-Wilk, Kruskal-Wallis, Mann-Whitney U, Spearman
├── ml_models.py         # 5-fold CV for 5 ML models; feature importance
├── visualization.py     # All 6 publication figures
├── requirements.txt     # Python dependencies
└── Data_set.xlsx        # ← place your dataset here
```

**Outputs** (auto-created in `outputs/`):

| File | Contents |
|------|----------|
| `cleaned_data.csv` | Standardised, combined dataset |
| `descriptive_stats.csv` | Summary statistics for all variables |
| `spatial_summary.csv` | Mean ± SD per monitoring station |
| `temporal_summary.csv` | Monthly mean per pollutant |
| `year_over_year.csv` | 2023 vs 2024-25 comparison |
| `spearman_correlation.csv` | Full correlation matrix |
| `statistical_tests.xlsx` | All formal tests (4 sheets) |
| `cv_results.csv` | 5-fold CV R², RMSE, MAE per model & target |
| `feature_importance.csv` | MDI importances (Random Forest) |
| `oof_predictions_gb.csv` | Out-of-fold GB predictions |
| `figures/fig1_*.png … fig6_*.png` | 6 publication-quality figures |

---

## Models Evaluated

| Model | Framework |
|-------|-----------|
| Linear Regression | `sklearn.linear_model` |
| Ridge Regression (α=1.0) | `sklearn.linear_model` |
| Random Forest (100 trees) | `sklearn.ensemble` |
| **Gradient Boosting** ← best | `sklearn.ensemble` |
| SVR (RBF, C=100) | `sklearn.svm` |

---

## Key Results

| Pollutant | Best Model | R² (CV) | RMSE (CV) |
|-----------|-----------|---------|-----------|
| CO₂ | Gradient Boosting | **0.701** | 18.18 ppm |
| N₂O | Gradient Boosting | **0.688** | 0.0066 ppm |
| CH₄ | Gradient Boosting | **0.867** | 1.507 ppm |
| CO  | Gradient Boosting | **0.814** | 2.438 ppm |

### Year-over-year increase (2023 → 2024-25)
- CH₄: **+224%** (p < 0.001) 🚨
- CO:  **+19%**  (p < 0.001)
- CO₂: **+2.7%** (p < 0.001)

---

## Quickstart

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/air-pollution-ml-portharcourt.git
cd air-pollution-ml-portharcourt

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your dataset in the project root
cp /path/to/Data_set.xlsx .

# 4. Run the full pipeline
python main.py

# Optional: specify a custom dataset path
python main.py --data /path/to/Data_set.xlsx
```

---

## Features Used

| Feature | Description |
|---------|-------------|
| `latitude` | Station latitude (°N, WGS84) |
| `longitude` | Station longitude (°E, WGS84) |
| `month_num` | Temporal index (1–12 ordinal) |
| `Temperature(°C)` | Ambient temperature |
| `Humidity(%)` | Relative humidity |
| `Wind Speed(m/s)` | Wind speed |
| `Hour Rainfall(mm)` | Hourly rainfall |

---

## Citation

If you use this code or dataset, please cite:

```
[Authors]. (2025). Geospatial Machine Learning for Air Pollution Prediction
Using Multivariate Environmental Data: A Spatiotemporal Analysis of Port
Harcourt, Nigeria. Environmental Science and Pollution Research, Springer.
```

---

## License

[MIT License](LICENSE) — free to use with attribution.
