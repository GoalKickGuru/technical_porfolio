#!/usr/bin/env python3
"""
Reusable Template Script – Data Preparation for EDA
===================================================
Copy this file, edit the CONFIG section, and run:

    python data_prep_template.py

Or import the functions from scripts/data_preparation.py
into your own pipelines.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG – edit these lines for your dataset
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "marketing_campaign.csv"
GROUP_BY = "Kidhome"                 # categorical column for groupby
VALUE_COL = "NumStorePurchases"      # numeric column to aggregate
SORT_BY = "NumStorePurchases"
BINS = [0, 4, 8, 13]
BIN_LABELS = ["Low", "Moderate", "High"]
FILL_STRATEGY = "median"             # mean | median | mode | constant | ffill | bfill
FILL_COLUMNS = ["Income"]
DROP_DUPLICATES_SUBSET = ["Education", "Marital_Status", "Kidhome", "Teenhome"]
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from data_preparation import (
    group_mean,
    group_and_aggregate,
    sort_dataframe,
    categorize_numeric,
    drop_duplicates,
    duplicate_report,
    missing_report,
    fill_missing,
    prepare_pipeline,
    print_audit,
)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns from {DATA_PATH.name}")

    # 1. Grouping
    if GROUP_BY in df.columns and VALUE_COL in df.columns:
        print(f"\n=== Mean of `{VALUE_COL}` by `{GROUP_BY}` ===")
        print(group_mean(df, GROUP_BY, VALUE_COL))

    # 2. Sorting
    if SORT_BY in df.columns:
        print(f"\n=== Top 5 rows sorted by `{SORT_BY}` (desc) ===")
        print(sort_dataframe(df, by=SORT_BY, ascending=False)[[ "ID", SORT_BY ] if "ID" in df.columns else [SORT_BY]].head())

    # 3. Binning
    if VALUE_COL in df.columns:
        print(f"\n=== Category counts for `{VALUE_COL}` ===")
        cats = categorize_numeric(df[VALUE_COL], bins=BINS, labels=BIN_LABELS)
        print(cats.value_counts().sort_index())

    # 4. Duplicates
    subset = [c for c in DROP_DUPLICATES_SUBSET if c in df.columns]
    if subset:
        print(f"\n=== Duplicate report on {subset} ===")
        print(duplicate_report(df, subset=subset))

    # 5. Missing values
    print("\n=== Missing-value report ===")
    print(missing_report(df).query("missing_count > 0"))

    # 6. Full pipeline
    print("\n=== Running prepare_pipeline ===")
    cleaned, audit = prepare_pipeline(
        df,
        fill_strategy=FILL_STRATEGY,
        fill_columns=[c for c in FILL_COLUMNS if c in df.columns],
        drop_duplicates_subset=subset or None,
        sort_by=SORT_BY if SORT_BY in df.columns else None,
    )
    print_audit(audit)
    print(f"Cleaned sample:\n{cleaned.head(3)}")


if __name__ == "__main__":
    main()
