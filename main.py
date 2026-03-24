"""
=============================================================================
Geospatial Machine Learning for Air Pollution Prediction
Using Multivariate Environmental Data — Port Harcourt, Nigeria (2023–2025)
=============================================================================
Authors  : [Your Name(s)]
Journal  : Environmental Science and Pollution Research (Springer)
Dataset  : 48 monitoring stations × 12 months = 576 station-month observations
Pollutants: CO2, N2O, CH4, O3, CO
=============================================================================
"""

import warnings
warnings.filterwarnings('ignore')

from data_loader   import load_and_clean
from eda           import run_eda
from ml_models     import run_ml_pipeline
from visualization import generate_all_figures
from stats_tests   import run_statistical_tests

import argparse, os

DATA_PATH = "Data_set.xlsx"
OUTPUT_DIR = "outputs"

def main(data_path=DATA_PATH):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  AIR POLLUTION ML PIPELINE  —  Port Harcourt, Nigeria")
    print("=" * 70)

    # 1 ── Load & clean
    print("\n[1/5] Loading and cleaning data …")
    df = load_and_clean(data_path)
    df.to_csv(f"{OUTPUT_DIR}/cleaned_data.csv", index=False)
    print(f"      ✓  {len(df)} records loaded, saved to {OUTPUT_DIR}/cleaned_data.csv")

    # 2 ── EDA
    print("\n[2/5] Running exploratory data analysis …")
    run_eda(df, OUTPUT_DIR)
    print(f"      ✓  EDA tables saved to {OUTPUT_DIR}/")

    # 3 ── Statistical tests
    print("\n[3/5] Running statistical tests …")
    run_statistical_tests(df, OUTPUT_DIR)
    print(f"      ✓  Statistical results saved to {OUTPUT_DIR}/statistical_tests.csv")

    # 4 ── Machine learning
    print("\n[4/5] Training and evaluating ML models …")
    results, feat_imp = run_ml_pipeline(df, OUTPUT_DIR)
    print(f"      ✓  CV results saved to {OUTPUT_DIR}/cv_results.csv")

    # 5 ── Figures
    print("\n[5/5] Generating publication figures …")
    generate_all_figures(df, results, feat_imp, OUTPUT_DIR)
    print(f"      ✓  All figures saved to {OUTPUT_DIR}/figures/")

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Air Pollution ML Pipeline")
    parser.add_argument("--data", default=DATA_PATH, help="Path to Excel dataset")
    args = parser.parse_args()
    main(args.data)
