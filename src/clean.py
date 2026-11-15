"""Load the raw Bengaluru listings CSV, clean it, and save data/processed/clean.csv.

Run:  python src/clean.py
Every rule that drops rows is logged with a count, so the cleaning is auditable.
"""
import re
import sys

import numpy as np
import pandas as pd

from config import (
    MAX_BATH_OVER_BHK,
    MAX_BHK,
    MIN_SQFT_PER_BHK,
    PPSF_PCT_HIGH,
    PPSF_PCT_LOW,
    PROCESSED_CSV,
    RAW_CSV,
)

# Multipliers to convert the odd area units in the data into square feet.
UNIT_TO_SQFT = {
    "Sq. Meter": 10.7639,
    "Sq. Yards": 9.0,
    "Acres": 43560.0,
    "Perch": 272.25,
    "Cents": 435.6,
    "Guntha": 1089.0,
    "Grounds": 2400.0,
}


def to_snake(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", name.strip().lower()).strip("_")


def parse_sqft(value) -> float:
    """'1200' -> 1200, '1133 - 1384' -> midpoint, '34.46Sq. Meter' -> converted."""
    s = str(value).strip()
    if " - " in s:
        parts = [parse_sqft(p) for p in s.split(" - ")]
        return float(np.mean(parts)) if not any(np.isnan(parts)) else np.nan
    try:
        return float(s)
    except ValueError:
        pass
    for unit, mult in UNIT_TO_SQFT.items():
        if s.endswith(unit):
            try:
                return float(s[: -len(unit)]) * mult
            except ValueError:
                return np.nan
    return np.nan


class DropLog:
    """Tiny helper that records how many rows each cleaning rule removes."""

    def __init__(self, df):
        self.start = len(df)
        self.rows = []

    def apply(self, df, mask_keep, reason):
        removed = int((~mask_keep).sum())
        self.rows.append((reason, removed))
        return df[mask_keep].copy()

    def report(self, df):
        print(f"Raw rows: {self.start:,}")
        for reason, n in self.rows:
            print(f"  - {n:>5,}  {reason}")
        print(f"Clean rows: {len(df):,}  ({len(df) / self.start:.0%} kept)")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=to_snake)
    log = DropLog(df)

    df = log.apply(df, ~df.duplicated(), "exact duplicate rows")

    # Parse fields into usable numbers / categories.
    df["location"] = df["location"].str.strip().str.replace(r"\s+", " ", regex=True)
    df["area_type"] = df["area_type"].str.replace(r"\s+", " ", regex=True).str.replace(" Area", "").str.strip()
    df["bhk"] = df["size"].str.extract(r"(\d+)").astype(float)[0]
    df["area_sqft"] = df["total_sqft"].map(parse_sqft)
    df["ready_to_move"] = df["availability"].isin(["Ready To Move", "Immediate Possession"]).astype(int)
    df["price_lakhs"] = df["price"]

    required = ["price_lakhs", "area_sqft", "location", "bhk", "bath"]
    df = log.apply(df, df[required].notna().all(axis=1), "missing price, area, locality, BHK or bathrooms")
    df = log.apply(df, (df["area_sqft"] > 0) & (df["price_lakhs"] > 0), "non-positive area or price")

    # Physically implausible listings (likely typos or mis-entered units).
    df = log.apply(df, df["bhk"] <= MAX_BHK, f"more than {MAX_BHK} BHK")
    df = log.apply(df, df["area_sqft"] / df["bhk"] >= MIN_SQFT_PER_BHK, f"under {MIN_SQFT_PER_BHK} sq ft per bedroom")
    df = log.apply(df, df["bath"] <= df["bhk"] + MAX_BATH_OVER_BHK, f"bathrooms > BHK + {MAX_BATH_OVER_BHK}")

    # Price-per-sqft outliers. Used ONLY to filter rows here; never a model input (leakage).
    ppsf = df["price_lakhs"] * 1e5 / df["area_sqft"]
    lo, hi = ppsf.quantile([PPSF_PCT_LOW, PPSF_PCT_HIGH])
    df = log.apply(df, ppsf.between(lo, hi),
                   f"price/sq ft outside {PPSF_PCT_LOW:.0%}-{PPSF_PCT_HIGH:.0%} percentile (Rs {lo:,.0f}-{hi:,.0f})")

    # Balcony is missing for ~5% of rows: fill with the median for that BHK rather than dropping.
    n_bal = int(df["balcony"].isna().sum())
    df["balcony"] = df["balcony"].fillna(df.groupby("bhk")["balcony"].transform("median")).fillna(1)
    log.report(df)
    print(f"Filled {n_bal:,} missing balcony values with the median for that BHK.")

    cols = ["location", "area_type", "ready_to_move", "area_sqft", "bhk", "bath", "balcony", "price_lakhs"]
    out = df[cols].rename(columns={"bath": "bathrooms", "balcony": "balconies"})
    out[["bhk", "bathrooms", "balconies"]] = out[["bhk", "bathrooms", "balconies"]].astype(int)
    out["area_sqft"] = out["area_sqft"].round(0)
    return out.reset_index(drop=True)


def main():
    if not RAW_CSV.exists():
        sys.exit(f"Raw data not found at {RAW_CSV}. Put Bengaluru_House_Data.csv in data/raw/.")
    df = clean(pd.read_csv(RAW_CSV))
    PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_CSV, index=False)
    print(f"Saved {PROCESSED_CSV.relative_to(PROCESSED_CSV.parents[2])}")


if __name__ == "__main__":
    main()
