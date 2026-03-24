"""
eda.py
──────
Exploratory Data Analysis:
  - Descriptive statistics
  - Spatial aggregation (mean per station)
  - Temporal aggregation (mean per month)
  - Year-over-year summary
  - Inter-pollutant Spearman correlation matrix
"""

import os
import pandas as pd
import numpy as np
from data_loader import POLLUTANTS, METEO, MONTH_ORDER

# Grouping constants
YEARS_2023    = [m for m in MONTH_ORDER if "2023" in m]
YEARS_2024_25 = [m for m in MONTH_ORDER if "2024" in m or "2025" in m]


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Full descriptive statistics for pollutants + meteorological variables."""
    cols = POLLUTANTS + METEO
    stats = df[cols].agg(["count", "mean", "std", "min",
                          lambda x: x.quantile(0.25),
                          "median",
                          lambda x: x.quantile(0.75),
                          "max"])
    stats.index = ["N", "Mean", "SD", "Min", "Q1", "Median", "Q3", "Max"]
    return stats.T.round(4)


def spatial_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± SD of each pollutant per monitoring station."""
    rows = []
    for station, grp in df.groupby("station"):
        row = {"station": station,
               "latitude":  grp["latitude"].mean(),
               "longitude": grp["longitude"].mean()}
        for p in POLLUTANTS:
            row[f"{p}_mean"] = grp[p].mean()
            row[f"{p}_std"]  = grp[p].std()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("CO2__mean", ascending=False).reset_index(drop=True)


def temporal_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly mean values for pollutants + meteorological variables."""
    monthly = (
        df.groupby("month")[POLLUTANTS + METEO]
          .mean()
          .reindex(MONTH_ORDER)
          .round(4)
    )
    return monthly


def year_over_year(df: pd.DataFrame) -> pd.DataFrame:
    """Compare 2023 vs 2024-2025 period means for each pollutant."""
    yr23   = df[df["month"].isin(YEARS_2023)]
    yr2425 = df[df["month"].isin(YEARS_2024_25)]
    rows = []
    for p in POLLUTANTS:
        m23   = yr23[p].mean()
        m2425 = yr2425[p].mean()
        pct   = (m2425 - m23) / m23 * 100
        rows.append({"pollutant": p,
                     "mean_2023": round(m23, 4),
                     "mean_2024_25": round(m2425, 4),
                     "pct_change": round(pct, 2)})
    return pd.DataFrame(rows)


def spearman_corr(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman rank correlation matrix for pollutants + meteorological vars."""
    cols = POLLUTANTS + METEO
    return df[cols].corr(method="spearman").round(3)


def run_eda(df: pd.DataFrame, output_dir: str) -> None:
    """Run full EDA suite and save results to CSV."""
    os.makedirs(output_dir, exist_ok=True)

    desc = descriptive_stats(df)
    desc.to_csv(f"{output_dir}/descriptive_stats.csv")
    print(f"\n  Descriptive Statistics:\n{desc.to_string()}")

    spatial = spatial_summary(df)
    spatial.to_csv(f"{output_dir}/spatial_summary.csv", index=False)
    print(f"\n  Top 5 CO2 stations:\n{spatial[['station','CO2__mean','CO2__std']].head().to_string(index=False)}")

    temporal = temporal_summary(df)
    temporal.to_csv(f"{output_dir}/temporal_summary.csv")
    print(f"\n  Monthly means (CO2, CH4, CO):\n{temporal[['CO2_','CH4','CO']].to_string()}")

    yoy = year_over_year(df)
    yoy.to_csv(f"{output_dir}/year_over_year.csv", index=False)
    print(f"\n  Year-over-Year Change:\n{yoy.to_string(index=False)}")

    corr = spearman_corr(df)
    corr.to_csv(f"{output_dir}/spearman_correlation.csv")
    print(f"\n  Spearman Correlation Matrix saved.")
