"""
General-Purpose CSV Import & Cleaning Utility
Works with any .csv file.

Usage:
    python csv_cleaner.py                          # interactive prompt
    python csv_cleaner.py data/myfile.csv           # direct path
    python csv_cleaner.py data/myfile.csv --sep ";"  # custom delimiter
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def configure_display():
    """Set pandas display options for comfortable reading."""
    pd.options.display.float_format = '{:,.2f}'.format
    pd.set_option('display.width', 120)
    pd.set_option('display.max_columns', 25)
    pd.set_option('display.max_rows', 60)


def load_csv(path: str, sep: str = ',') -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    - Auto-detects whether the first row is a header.
    - Attempts to parse columns that look like dates.
    - Sets low_memory=False for consistent dtype inference.
    """
    df = pd.read_csv(
        path,
        sep=sep,
        low_memory=False,
        parse_dates=True,       # let pandas auto-detect date-like columns
        infer_datetime_format=True,
        encoding_errors='replace',
    )
    print(f"Loaded: {path}")
    print(f"Shape : {df.shape[0]:,} rows × {df.shape[1]} columns\n")
    return df


def audit_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary table of missing values per column."""
    counts = df.isnull().sum()
    pct = (counts / len(df) * 100).round(2)
    summary = pd.DataFrame({'missing_count': counts, 'missing_pct': pct})
    return summary[summary['missing_count'] > 0].sort_values(
        'missing_pct', ascending=False
    )


def audit_dtypes(df: pd.DataFrame) -> None:
    """Print the inferred dtypes for each column."""
    print("─── Column data types ───")
    for col in df.columns:
        print(f"  {col:<35} {str(df[col].dtype)}")


def drop_missing_by_threshold(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Drop *columns* that are more than `threshold` (fraction) empty,
    then drop *rows* that are entirely empty.
    """
    before_cols = df.shape[1]
    df = df.dropna(axis=1, thresh=int(len(df) * (1 - threshold)))
    dropped_cols = before_cols - df.shape[1]
    if dropped_cols:
        print(f"Dropped {dropped_cols} column(s) exceeding "
              f"{threshold*100:.0f}% missing.")

    before_rows = df.shape[0]
    df = df.dropna(how='all')
    dropped_rows = before_rows - df.shape[0]
    if dropped_rows:
        print(f"Dropped {dropped_rows:,} fully-empty row(s).")

    return df


def remove_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame:
    """Remove exact duplicate rows (optionally on a subset of columns)."""
    dup_count = df.duplicated(subset=subset).sum()
    if dup_count:
        df = df.drop_duplicates(subset=subset)
        print(f"Removed {dup_count:,} duplicate row(s).")
    else:
        print("No duplicates found.")
    return df


def flag_numeric_outliers_iqr(
    df: pd.DataFrame, cols: list[str] | None = None
) -> pd.DataFrame:
    """
    Add a boolean `is_outlier_<col>` column for each numeric column,
    using the IQR (1.5×IQR) method. Non-destructive — rows are flagged,
    not removed.
    """
    if cols is None:
        cols = df.select_dtypes(include='number').columns.tolist()

    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        flag_col = f"is_outlier_{col}"
        df[flag_col] = ~df[col].between(lo, hi)
        n_outliers = df[flag_col].sum()
        print(f"  {col}: {n_outliers} outliers flagged (IQR bounds: "
              f"{lo:.2f} – {hi:.2f})")

    return df


def full_report(df: pd.DataFrame) -> None:
    """Print a comprehensive overview of the DataFrame."""
    print("\n" + "=" * 70)
    print(" DATAFRAME OVERVIEW")
    print("=" * 70)

    audit_dtypes(df)

    print("\n─── First 5 rows ───")
    print(df.head())

    print("\n─── Numeric summary statistics ───")
    print(df.describe())

    print("\n─── Categorical summary (top categories) ───")
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols[:5]:  # show up to 5 categorical columns
        vc = df[col].value_counts()
        print(f"\n  {col} ({df[col].nunique()} unique values)")
        print(vc.head(5).to_string())

    missing = audit_missing(df)
    if missing.empty:
        print("\n─── Missing values: none ───")
    else:
        print("\n─── Missing values ───")
        print(missing.to_string())


def run_pipeline(filepath: str, sep: str = ',', drop_pct: float = 0.5) -> None:
    """Run the full import → clean → report pipeline."""
    configure_display()
    df = load_csv(filepath, sep=sep)

    full_report(df)

    print("\n" + "=" * 70)
    print(" CLEANING STEPS")
    print("=" * 70)

    df = drop_missing_by_threshold(df, threshold=drop_pct)
    df = remove_duplicates(df)
    df = flag_numeric_outliers_iqr(df)

    # Re-report missing after cleaning
    print("\n─── Remaining missing values after cleaning ───")
    remaining = audit_missing(df)
    print(remaining if not remaining.empty else "  None.")

    print(f"\n✅ Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Persist
    stem = Path(filepath).stem
    out_path = f"{stem}_cleaned.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved cleaned data to: {out_path}")


# ── CLI entry point ───────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Import and clean any CSV file with pandas.'
    )
    parser.add_argument('filepath', nargs='?', help='Path to the CSV file.')
    parser.add_argument('--sep', default=',', help='Column delimiter (default: ",").')
    parser.add_argument(
        '--drop-pct', type=float, default=0.5,
        help='Drop columns with more than this fraction missing (default: 0.5).'
    )
    args = parser.parse_args()

    if args.filepath is None:
        filepath = input("Enter the path to your CSV file: ").strip()
        if not filepath:
            print("No file path provided. Exiting.")
            sys.exit(1)
    else:
        filepath = args.filepath

    run_pipeline(filepath, sep=args.sep, drop_pct=args.drop_pct)