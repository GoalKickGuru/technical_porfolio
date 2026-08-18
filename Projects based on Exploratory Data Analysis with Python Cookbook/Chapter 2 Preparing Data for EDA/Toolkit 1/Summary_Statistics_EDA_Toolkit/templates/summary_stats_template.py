#!/usr/bin/env python3
"""
Reusable Template Script – Summary Statistics
=============================================
Copy this file, edit the CONFIG section, and run:

    python summary_stats_template.py

Or import the functions from scripts/summary_statistics.py
into your own pipelines.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG – edit these lines for your dataset
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "covid-data.csv"
VALUE_COL = "new_cases"
GROUP_COL = "continent"          # set to None to skip group-by
PERCENTILES = [10, 25, 50, 75, 90, 95, 99]
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from summary_statistics import (
    full_summary,
    summary_by_group,
    print_summary_report,
)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {df.shape[0]:,} rows from {DATA_PATH.name}")
    assert VALUE_COL in df.columns, f"Column '{VALUE_COL}' not found"

    series = df[VALUE_COL]
    summary = full_summary(series, percentiles=PERCENTILES)
    print_summary_report(summary, f"Summary of `{VALUE_COL}`")

    if GROUP_COL and GROUP_COL in df.columns:
        print(f"\n=== Grouped by `{GROUP_COL}` ===")
        grouped = summary_by_group(df, VALUE_COL, GROUP_COL)
        print(grouped.to_string(index=False))


if __name__ == "__main__":
    main()
