#!/usr/bin/env python3
"""
Run a full data-preparation demonstration on the Marketing Campaign sample.

Usage:
    python run_data_preparation.py [--data PATH]

Expects marketing_campaign.csv (and optional append/concat/merge variants)
in the data/ folder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from data_preparation import (
    group_and_aggregate,
    group_mean,
    append_dataframes,
    concatenate_dataframes,
    merge_dataframes,
    sort_dataframe,
    categorize_numeric,
    drop_duplicates,
    duplicate_report,
    drop_columns,
    change_dtype,
    replace_values,
    missing_report,
    drop_missing,
    fill_missing,
    prepare_pipeline,
    print_audit,
)


def load_main(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Core columns used across most recipes
    cols = [
        "ID", "Year_Birth", "Education", "Marital_Status", "Income",
        "Kidhome", "Teenhome", "Dt_Customer", "Recency",
        "NumStorePurchases", "NumWebVisitsMonth",
    ]
    available = [c for c in cols if c in df.columns]
    return df[available].copy()


def main(data_dir: Path) -> None:
    main_path = data_dir / "marketing_campaign.csv"
    if not main_path.exists():
        raise FileNotFoundError(f"Expected {main_path}")

    print("\n>>> 1. Loading main marketing campaign data")
    df = load_main(main_path)
    print(f"Shape: {df.shape}")
    print(df.head(2).T)

    print("\n>>> 2. Grouping – average store purchases by Kidhome")
    g = group_mean(df, "Kidhome", "NumStorePurchases")
    print(g)

    print("\n>>> 3. Multi-agg groupby example")
    multi = group_and_aggregate(
        df,
        by=["Education"],
        agg={"NumStorePurchases": ["mean", "median"], "Income": "mean"},
    )
    print(multi.head())

    # Append demo
    a1 = data_dir / "marketing_campaign_append1.csv"
    a2 = data_dir / "marketing_campaign_append2.csv"
    if a1.exists() and a2.exists():
        print("\n>>> 4. Appending two sample files")
        s1 = pd.read_csv(a1)[df.columns.intersection(pd.read_csv(a1).columns)]
        s2 = pd.read_csv(a2)[df.columns.intersection(pd.read_csv(a2).columns)]
        appended = append_dataframes([s1, s2])
        print(f"Appended shape: {appended.shape}")

    # Concat horizontal
    c1 = data_dir / "marketing_campaign_concat1.csv"
    c2 = data_dir / "marketing_campaign_concat2.csv"
    if c1.exists() and c2.exists():
        print("\n>>> 5. Horizontal concatenation")
        left = pd.read_csv(c1)
        right = pd.read_csv(c2)
        concat = concatenate_dataframes([left, right], axis=1)
        print(f"Concat shape: {concat.shape}")
        print(concat.head(2))

    # Merge
    m1 = data_dir / "marketing_campaign_merge1.csv"
    m2 = data_dir / "marketing_campaign_merge2.csv"
    if m1.exists() and m2.exists():
        print("\n>>> 6. Merging on ID")
        left = pd.read_csv(m1)
        right = pd.read_csv(m2)
        merged = merge_dataframes(left, right, on="ID", how="inner")
        print(f"Merged shape: {merged.shape}")
        print(merged.head(2))

    print("\n>>> 7. Sorting by NumStorePurchases descending")
    sorted_df = sort_dataframe(df, by="NumStorePurchases", ascending=False)
    print(sorted_df[["ID", "NumStorePurchases"]].head())

    print("\n>>> 8. Categorizing NumStorePurchases")
    cats = categorize_numeric(
        df["NumStorePurchases"],
        bins=[0, 4, 8, 13],
        labels=["Low", "Moderate", "High"],
    )
    print(cats.value_counts())

    print("\n>>> 9. Duplicate report (Education + Marital_Status + Kidhome + Teenhome)")
    subset = ["Education", "Marital_Status", "Kidhome", "Teenhome"]
    if all(c in df.columns for c in subset):
        rep = duplicate_report(df, subset=subset)
        print(rep)
        deduped = drop_duplicates(df[subset], subset=subset)
        print(f"After drop_duplicates: {deduped.shape}")

    print("\n>>> 10. Missing-value report")
    print(missing_report(df).head(10))

    print("\n>>> 11. Fill missing Income with median then cast to int")
    filled = fill_missing(df, strategy="median", columns=["Income"])
    filled = change_dtype(filled, "Income", "int64", errors="ignore")
    print(filled[["Income"]].dtypes)
    print(filled[["Income"]].head(3))

    print("\n>>> 12. Replace Teenhome values")
    replaced = replace_values(
        df, "Teenhome", to_replace=[0, 1, 2], value=["has no teen", "has teen", "has teen"]
    )
    print(replaced[["Teenhome"]].head())

    print("\n>>> 13. Full prepare_pipeline demo")
    cleaned, audit = prepare_pipeline(
        df,
        drop_cols=None,
        drop_duplicates_subset=None,
        fill_strategy="median",
        fill_columns=["Income"],
        sort_by="Recency",
    )
    print_audit(audit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data-preparation demos")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data",
        help="Directory containing the marketing_campaign*.csv files",
    )
    args = parser.parse_args()
    main(args.data)
