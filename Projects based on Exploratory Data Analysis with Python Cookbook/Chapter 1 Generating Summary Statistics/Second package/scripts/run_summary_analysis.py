#!/usr/bin/env python3
"""
Run full summary statistics analysis on the COVID-19 sample dataset.

Usage:
    python run_summary_analysis.py [--data PATH]

Expects covid-data.csv with at least the columns:
    iso_code, continent, location, date, total_cases, new_cases
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow running from project root or scripts/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from summary_statistics import (
    compute_mean,
    compute_median,
    compute_mode,
    compute_variance,
    compute_std,
    compute_range,
    compute_percentile,
    compute_quartile,
    compute_iqr,
    full_summary,
    summary_by_group,
    print_summary_report,
)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["iso_code", "continent", "location", "date", "total_cases", "new_cases"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    # Keep only the relevant columns (matches original notebooks)
    return df[required].copy()


def main(data_path: Path) -> None:
    print(f"Loading data from: {data_path}")
    covid = load_data(data_path)
    print(f"Shape: {covid.shape}")
    print(f"Columns: {list(covid.columns)}")
    print("\nFirst 5 rows:")
    print(covid.head())
    print("\nData types:")
    print(covid.dtypes)

    col = "new_cases"
    print(f"\n{'='*60}")
    print(f"ANALYSIS OF COLUMN: {col}")
    print(f"{'='*60}")

    # Individual statistics (mirroring original notebooks + enhancements)
    print("\n1. MEAN")
    print(f"   np / pandas mean: {compute_mean(covid[col]):,.4f}")

    print("\n2. MEDIAN")
    print(f"   median: {compute_median(covid[col]):,.4f}")

    print("\n3. MODE")
    mode_info = compute_mode(covid[col])
    print(f"   mode: {mode_info['mode']} (count={mode_info['count']})")

    print("\n4. VARIANCE")
    var0 = compute_variance(covid[col], ddof=0)
    var1 = compute_variance(covid[col], ddof=1)
    print(f"   population (ddof=0): {var0['numpy_var']:,.4f}")
    print(f"   sample     (ddof=1): {var1['pandas_var']:,.4f}")

    print("\n5. STANDARD DEVIATION")
    std0 = compute_std(covid[col], ddof=0)
    std1 = compute_std(covid[col], ddof=1)
    print(f"   population (ddof=0): {std0['numpy_std']:,.4f}")
    print(f"   sample     (ddof=1): {std1['pandas_std']:,.4f}")

    print("\n6. RANGE")
    rng = compute_range(covid[col])
    print(f"   min={rng['min']:,.0f}, max={rng['max']:,.0f}, range={rng['range']:,.0f}")

    print("\n7. PERCENTILES")
    for p in [25, 50, 60, 75, 90, 95]:
        print(f"   {p}th: {compute_percentile(covid[col], p):,.4f}")

    print("\n8. QUARTILES")
    print(f"   Q1 (0.25): {compute_quartile(covid[col], 0.25):,.4f}")
    print(f"   Q2 (0.50): {compute_quartile(covid[col], 0.50):,.4f}")
    print(f"   Q3 (0.75): {compute_quartile(covid[col], 0.75):,.4f}")

    print("\n9. INTERQUARTILE RANGE (IQR)")
    iqr = compute_iqr(covid[col])
    print(f"   IQR (midpoint): {iqr['iqr']:,.4f}")
    print(f"   Q1={iqr['q1']:,.4f}, Q3={iqr['q3']:,.4f}")

    # Full summary
    print("\n")
    full = full_summary(covid[col])
    print_summary_report(full, f"Full Summary – {col}")

    # Group-by enhancement
    print("\n=== Summary by Continent (enhancement) ===")
    by_cont = summary_by_group(covid, "new_cases", "continent")
    print(by_cont.to_string(index=False))

    print("\n=== Summary by Location (enhancement) ===")
    by_loc = summary_by_group(covid, "new_cases", "location")
    print(by_loc.to_string(index=False))

    # Save group summaries for spreadsheet use
    out_dir = ROOT / "data"
    by_cont.to_csv(out_dir / "summary_by_continent.csv", index=False)
    by_loc.to_csv(out_dir / "summary_by_location.csv", index=False)
    print(f"\nGroup summaries saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run COVID summary statistics")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "covid-data.csv",
        help="Path to covid-data.csv",
    )
    args = parser.parse_args()
    if not args.data.exists():
        print(f"ERROR: Data file not found: {args.data}")
        print("Place covid-data.csv in the data/ folder or pass --data PATH")
        sys.exit(1)
    main(args.data)
