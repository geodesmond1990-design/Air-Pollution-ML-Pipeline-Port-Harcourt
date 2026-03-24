"""
data_loader.py
──────────────
Loads all 12 monthly sheets from the Excel dataset, standardises column
names, fixes the Lat/Long swap present in sheets from October 2023 onward,
and adds derived temporal features.
"""

import pandas as pd
import numpy as np

MONTH_ORDER = [
    "AUGUST 2023", "SEPTEMBER 2023", "OCTOBER 2023", "NOVEMBER 2023",
    "DECEMBER 2023", "JANUARY 2024", "AUGUST 2024", "SEPTEMBER 2024",
    "OCTOBER 2024", "NOVEMBER 2024", "DECEMBER 2024", "JANUARY 2025",
]

POLLUTANTS   = ["CO2_", "N2O", "CH4", "O3", "CO"]
METEO        = ["Temperature(°C)", "Humidity(%)", "Wind Speed(m/s)", "Hour Rainfall(mm)"]
GEO          = ["latitude", "longitude"]
FEATURES     = GEO + ["month_num"] + METEO
ALL_VARS     = POLLUTANTS + METEO


def _standardise_coords(df: pd.DataFrame) -> pd.DataFrame:
    """
    August & September 2023 sheets use column names  Lat / Long
    with Lat ≈ 4.7–5.0 (correct latitude) and Long ≈ 6.9–7.1 (correct longitude).

    From October 2023 onward the columns are swapped in the source file:
    LONG column contains the latitude values (≈ 4.7–5.0)
    LAT  column contains the longitude values (≈ 6.9–7.1)

    We detect the swap by checking whether the value in the first column
    (whichever name it has) is < 5 (latitude) or > 6 (longitude).
    """
    cols = {c.strip().upper(): c for c in df.columns}

    if "LAT" in cols and "LONG" in cols:
        lat_col  = cols["LAT"]
        long_col = cols["LONG"]

        # Detect swap: true latitude for Port Harcourt ≈ 4.7–5.0
        first_lat_val = df[lat_col].dropna().iloc[0]
        if first_lat_val > 6:          # values are longitude → columns are swapped
            df = df.rename(columns={lat_col: "longitude", long_col: "latitude"})
        else:
            df = df.rename(columns={lat_col: "latitude",  long_col: "longitude"})

    elif "Lat" in df.columns and "Long" in df.columns:
        df = df.rename(columns={"Lat": "latitude", "Long": "longitude"})

    return df


def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Parameters
    ----------
    filepath : str
        Path to the multi-sheet Excel file (Data_set.xlsx).

    Returns
    -------
    pd.DataFrame
        Combined, cleaned dataset (576 rows × 14 columns).
    """
    all_sheets = pd.read_excel(filepath, sheet_name=None)
    frames = []

    for sheet_name, raw in all_sheets.items():
        df = raw.copy()
        df.columns = [c.strip() for c in df.columns]

        # Standardise coordinate columns
        df = _standardise_coords(df)

        # Drop the serial-number column if present
        df = df.drop(columns=[c for c in df.columns if c.lower() in ("s/n", "sn", "s.n.")],
                     errors="ignore")

        # Rename station code column
        code_col = next((c for c in df.columns if c.upper() == "CODE"), None)
        if code_col and code_col != "station":
            df = df.rename(columns={code_col: "station"})

        df["month"] = sheet_name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # ── Derived features ──────────────────────────────────────────────────────
    combined["month_num"] = pd.Categorical(
        combined["month"], categories=MONTH_ORDER, ordered=True
    ).codes + 1                          # 1–12 ordinal index

    # Validate coordinate ranges (Port Harcourt bounding box)
    assert combined["latitude"].between(4.5, 5.2).all(),  \
        "Unexpected latitude values — check coordinate assignment."
    assert combined["longitude"].between(6.8, 7.2).all(), \
        "Unexpected longitude values — check coordinate assignment."

    # ── Type safety ───────────────────────────────────────────────────────────
    numeric_cols = ALL_VARS + GEO + ["month_num"]
    for col in numeric_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # ── Missing value report ──────────────────────────────────────────────────
    missing = combined[numeric_cols].isnull().sum()
    if missing.any():
        print("[WARNING] Missing values detected:")
        print(missing[missing > 0])

    print(f"  Dataset shape : {combined.shape}")
    print(f"  Stations      : {combined['station'].nunique()}")
    print(f"  Months        : {combined['month'].nunique()}  ({MONTH_ORDER[0]} → {MONTH_ORDER[-1]})")

    return combined
