"""
stats_tests.py
──────────────
Formal statistical tests used in the paper:
  1. Shapiro-Wilk normality test
  2. Kruskal-Wallis H-test (seasonal differences across months)
  3. Mann-Whitney U test (2023 vs 2024-25 year-over-year)
  4. Spearman correlations with significance stars
"""

import os
import warnings
import pandas as pd
import numpy as np
from scipy import stats
from data_loader import POLLUTANTS, METEO, MONTH_ORDER

warnings.filterwarnings("ignore")

YEARS_2023    = [m for m in MONTH_ORDER if "2023" in m]
YEARS_2024_25 = [m for m in MONTH_ORDER if "2024" in m or "2025" in m]


def _sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def shapiro_wilk_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Test normality for each pollutant and meteorological variable."""
    rows = []
    for col in POLLUTANTS + METEO:
        sample = df[col].dropna().values
        # Shapiro-Wilk is valid for n ≤ 5000; use up to 500 samples
        sample = sample[:500] if len(sample) > 500 else sample
        w, p = stats.shapiro(sample)
        rows.append({"variable": col, "W": round(w, 4), "p_value": round(p, 6),
                     "normal": "Yes" if p >= 0.05 else "No",
                     "significance": _sig_stars(p)})
    return pd.DataFrame(rows)


def kruskal_wallis_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test whether pollutant concentrations differ significantly across the
    12 monthly observation periods (non-parametric one-way ANOVA).
    """
    rows = []
    for pollutant in POLLUTANTS:
        groups = [df[df["month"] == m][pollutant].dropna().values
                  for m in MONTH_ORDER]
        groups = [g for g in groups if len(g) > 0]
        h, p = stats.kruskal(*groups)
        rows.append({"pollutant": pollutant, "H_statistic": round(h, 4),
                     "p_value": round(p, 8), "significance": _sig_stars(p)})
    return pd.DataFrame(rows)


def mann_whitney_yoy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mann-Whitney U test comparing 2023 vs 2024-2025 concentrations
    for each pollutant.
    """
    yr23   = df[df["month"].isin(YEARS_2023)]
    yr2425 = df[df["month"].isin(YEARS_2024_25)]
    rows = []
    for p in POLLUTANTS:
        u, pval = stats.mannwhitneyu(yr23[p].dropna(), yr2425[p].dropna(),
                                     alternative="two-sided")
        mean23   = yr23[p].mean()
        mean2425 = yr2425[p].mean()
        pct      = (mean2425 - mean23) / mean23 * 100
        rows.append({
            "pollutant":   p,
            "mean_2023":   round(mean23,   4),
            "mean_2024_25": round(mean2425, 4),
            "pct_change":  round(pct,      2),
            "U_statistic": round(u,        2),
            "p_value":     round(pval,     8),
            "significance": _sig_stars(pval),
        })
    return pd.DataFrame(rows)


def spearman_significance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Spearman r and two-tailed p-value for every pollutant ×
    meteorological variable pair.
    """
    rows = []
    for pollutant in POLLUTANTS:
        for meteo in METEO:
            r, p = stats.spearmanr(df[pollutant].dropna(),
                                   df[meteo].dropna())
            rows.append({
                "pollutant":   pollutant,
                "meteo_var":   meteo,
                "spearman_r":  round(r, 4),
                "p_value":     round(p, 6),
                "significance": _sig_stars(p),
            })
    return pd.DataFrame(rows)


def run_statistical_tests(df: pd.DataFrame, output_dir: str) -> None:
    """Run all statistical tests and save results."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n  ── Shapiro-Wilk Normality Tests ──")
    sw = shapiro_wilk_tests(df)
    print(sw.to_string(index=False))
    sw.to_csv(f"{output_dir}/shapiro_wilk.csv", index=False)

    print("\n  ── Kruskal-Wallis Tests (seasonal differences) ──")
    kw = kruskal_wallis_tests(df)
    print(kw.to_string(index=False))
    kw.to_csv(f"{output_dir}/kruskal_wallis.csv", index=False)

    print("\n  ── Mann-Whitney U Tests (2023 vs 2024-25) ──")
    mw = mann_whitney_yoy(df)
    print(mw.to_string(index=False))
    mw.to_csv(f"{output_dir}/mann_whitney_yoy.csv", index=False)

    print("\n  ── Spearman Correlations (pollutant × meteorology) ──")
    sc = spearman_significance(df)
    sig_only = sc[sc["p_value"] < 0.05]
    print(sig_only.to_string(index=False))
    sc.to_csv(f"{output_dir}/spearman_significance.csv", index=False)

    # Combine all into one file for convenience
    all_tests = {
        "shapiro_wilk":       sw,
        "kruskal_wallis":     kw,
        "mann_whitney_yoy":   mw,
        "spearman_corr":      sc,
    }
    with pd.ExcelWriter(f"{output_dir}/statistical_tests.xlsx") as writer:
        for sheet, frame in all_tests.items():
            frame.to_excel(writer, sheet_name=sheet, index=False)
    print(f"\n  All test results saved to {output_dir}/statistical_tests.xlsx")
