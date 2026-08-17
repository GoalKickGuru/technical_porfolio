"""
General-Purpose CSV Import & Cleaning Utility (with Logging & Profiling)

Usage:
    python csv_cleaner.py data/myfile.csv
    python csv_cleaner.py data/myfile.csv --sep ";"
    python csv_cleaner.py data/myfile.csv --sep "|" --drop-pct 0.7 --no-outliers

Features:
    • Works with any CSV — auto-detects dtypes and date-like columns
    • Structured logging to console + timestamped file
    • Before/after profiling report (rows, cols, missing, duplicates, memory)
    • Per-column numeric stats in the comparison table
    • Per-step cleaning log with counts
    • JSON report persisted alongside logs
    • CLI flags: --sep, --drop-pct, --no-outliers, --no-dedup, --encoding
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# ════════════════════════════════════════════════════════════════════
#  Logging Configuration
# ════════════════════════════════════════════════════════════════════
LOG_DIR = Path('logs')
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / f"csv_clean_{datetime.now():%Y%m%d_%H%M%S}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s │ %(levelname)-7s │ %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger('csv_cleaner')

# ════════════════════════════════════════════════════════════════════
#  Display Settings
# ════════════════════════════════════════════════════════════════════
pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.width', 130)
pd.set_option('display.max_columns', 25)
pd.set_option('display.expand_frame_repr', False)


# ════════════════════════════════════════════════════════════════════
#  Profiling Helpers
# ════════════════════════════════════════════════════════════════════
def profile_snapshot(df: pd.DataFrame, label: str) -> dict:
    """Capture a structured profile of the DataFrame."""
    snap = {
        'label': label,
        'timestamp': datetime.now().isoformat(),
        'rows': int(df.shape[0]),
        'cols': int(df.shape[1]),
        'column_names': list(df.columns),
        'memory_mb': round(df.memory_usage(deep=True).sum() / 1_048_576, 2),
        'missing_total': int(df.isnull().sum().sum()),
        'duplicate_rows': int(df.duplicated().sum()),
        'numeric_cols': {},
        'categorical_cols': {},
    }

    for col in df.select_dtypes(include='number').columns:
        s = df[col]
        snap['numeric_cols'][col] = {
            'min':      float(s.min()) if not s.isnull().all() else None,
            'max':      float(s.max()) if not s.isnull().all() else None,
            'mean':     round(float(s.mean()), 4) if not s.isnull().all() else None,
            'std':      round(float(s.std()), 4) if not s.isnull().all() else None,
            'missing':  int(s.isnull().sum()),
        }

    for col in df.select_dtypes(include='object').columns:
        s = df[col]
        top = s.value_counts().head(3)
        snap['categorical_cols'][col] = {
            'unique_count': int(s.nunique()),
            'missing':      int(s.isnull().sum()),
            'top_values':   {str(k): int(v) for k, v in top.items()},
        }

    log.info(f"Profile [{label}]: {snap['rows']:,} rows × {snap['cols']} cols | "
             f"{snap['missing_total']:,} missing | "
             f"{snap['duplicate_rows']:,} duplicates | "
             f"{snap['memory_mb']} MB")
    return snap


def print_comparison_table(before: dict, after: dict) -> None:
    """Print a formatted before/after comparison."""
    banner_w = 72

    log.info("╔" + "═" * banner_w + "╗")
    log.info("║" + "  BEFORE vs AFTER CLEANING — COMPARISON REPORT".center(banner_w) + "║")
    log.info("╠" + "═" * banner_w + "╣")

    # ── Overall metrics ──────────────────────────────────────────
    metrics = [
        ('Rows',              before['rows'],          after['rows']),
        ('Columns',           before['cols'],          after['cols']),
        ('Total missing',     before['missing_total'],  after['missing_total']),
        ('Duplicate rows',    before['duplicate_rows'], after['duplicate_rows']),
        ('Memory (MB)',       before['memory_mb'],       after['memory_mb']),
    ]

    for name, b, a in metrics:
        if isinstance(b, float):
            delta = round(a - b, 2)
            b_str, a_str = f"{b:.2f}", f"{a:.2f}"
        else:
            delta = a - b
            b_str, a_str = f"{b:,}", f"{a:,}"
        arrow = '→' if delta == 0 else ('↓' if delta < 0 else '↑')
        log.info(f"║  {name:<22} {b_str:>14}  {arrow}  {a_str:>14}  "
                 f"({delta:+,})".ljust(42) + "║")

    # ── Column changes ───────────────────────────────────────────
    cols_before = set(before['column_names'])
    cols_after = set(after['column_names'])
    added = cols_after - cols_before
    removed = cols_before - cols_after

    if added or removed:
        log.info("╠" + "═" * banner_w + "╣")
        log.info("║" + "  COLUMN CHANGES".center(banner_w) + "║")
        log.info("╠" + "═" * banner_w + "╣")
        if added:
            log.info("║  Columns ADDED:" + " " * (banner_w - 16) + "║")
            for c in sorted(added):
                log.info(f"║    + {c}".ljust(banner_w + 3) + "║")
        if removed:
            log.info("║  Columns REMOVED:" + " " * (banner_w - 18) + "║")
            for c in sorted(removed):
                log.info(f"║    - {c}".ljust(banner_w + 3) + "║")

    # ── Numeric column details ───────────────────────────────────
    num_before = before.get('numeric_cols', {})
    num_after = after.get('numeric_cols', {})
    shared_num = sorted(set(num_before.keys()) & set(num_after.keys()))

    if shared_num:
        log.info("╠" + "═" * banner_w + "╣")
        log.info("║" + "  NUMERIC COLUMN DETAILS (shared columns)".center(banner_w) + "║")
        log.info("╠" + "═" * banner_w + "╣")

        for col in shared_num:
            b = num_before[col]
            a = num_after[col]

            def _fmt(val):
                if val is None:
                    return 'N/A'
                return f"{val:,.2f}"

            log.info(
                f"║  {col:<22}"
                f" min: {_fmt(b['min']):>10} → {_fmt(a['min']):>10}   "
                f" max: {_fmt(b['max']):>10} → {_fmt(a['max']):>10}   "
                f" miss: {b['missing']:>6,} → {a['missing']:>6,}"
            )

    # ── Categorical column details ───────────────────────────────
    cat_before = before.get('categorical_cols', {})
    cat_after = after.get('categorical_cols', {})
    shared_cat = sorted(set(cat_before.keys()) & set(cat_after.keys()))

    if shared_cat:
        log.info("╠" + "═" * banner_w + "╣")
        log.info("║" + "  CATEGORICAL COLUMN DETAILS (shared columns)".center(banner_w) + "║")
        log.info("╠" + "═" * banner_w + "╣")

        for col in shared_cat[:10]:
            b = cat_before[col]
            a = cat_after[col]
            log.info(
                f"║  {col:<22}"
                f" unique: {b['unique_count']:>7,} → {a['unique_count']:>7,}   "
                f" miss: {b['missing']:>6,} → {a['missing']:>6,}"
            )

    log.info("╚" + "═" * banner_w + "╝")


# ════════════════════════════════════════════════════════════════════
#  Pipeline Steps
# ════════════════════════════════════════════════════════════════════
def step_load(path: str, sep: str, encoding: str) -> pd.DataFrame:
    """Load the CSV file."""
    log.info("[STEP] Loading CSV …")
    log.info(f"  File:     {path}")
    log.info(f"  Delimiter: '{sep}'")
    log.info(f"  Encoding:  {encoding}")

    df = pd.read_csv(
        path,
        sep=sep,
        low_memory=False,
        parse_dates=True,
        infer_datetime_format=True,
        encoding_errors='replace',
        encoding=encoding,
    )
    log.info(f"  Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def step_audit_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Audit and report missing values."""
    log.info("[STEP] Missing-value audit …")
    counts = df.isnull().sum()
    pct = (counts / len(df) * 100).round(2)

    report = pd.DataFrame({
        'missing': counts,
        'missing_pct': pct,
    }).sort_values('missing', ascending=False)

    report = report[report['missing'] > 0]

    if report.empty:
        log.info("  No missing values found ✓")
    else:
        for col, row in report.iterrows():
            level = logging.WARNING if row['missing_pct'] > 20 else logging.INFO
            log.log(level, f"  {col:<35} {row['missing']:>7,} missing "
                           f"({row['missing_pct']:.2f}%)")

    return report


def step_drop_sparse_cols(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns exceeding the missing-fraction threshold."""
    log.info(f"[STEP] Dropping columns > {threshold*100:.0f}% missing …")
    before_cols = df.shape[1]
    df = df.dropna(axis=1, thresh=int(len(df) * (1 - threshold)))
    dropped = before_cols - df.shape[1]

    dropped_names = list(set(range(before_cols)) - set(range(df.shape[1])))
    # Recompute properly
    # Actually, we need the column names
    return df, []  # placeholder, replaced below


def step_drop_sparse_cols_v2(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns exceeding the missing-fraction threshold. Returns (df, dropped_names)."""
    log.info(f"[STEP] Dropping columns > {threshold*100:.0f}% missing …")
    missing_frac = df.isnull().mean()
    to_drop = missing_frac[missing_frac > threshold].index.tolist()

    if to_drop:
        for col in to_drop:
            log.warning(f"  Dropping column '{col}' "
                        f"({missing_frac[col]*100:.1f}% missing)")
        df = df.drop(columns=to_drop)
    else:
        log.info("  No columns exceeded the missing threshold ✓")

    return df, to_drop


def step_drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that are entirely empty."""
    log.info("[STEP] Dropping fully-empty rows …")
    before = df.shape[0]
    df = df.dropna(how='all')
    dropped = before - df.shape[0]
    if dropped:
        log.info(f"  Removed {dropped:,} empty row(s)")
    else:
        log.info("  No empty rows found ✓")
    return df


def step_remove_duplicates(
    df: pd.DataFrame, subset: list[str] | None = None
) -> pd.DataFrame:
    """Remove duplicate rows."""
    log.info("[STEP] Duplicate removal …")
    dup_count = df.duplicated(subset=subset).sum()
    if dup_count:
        log.warning(f"  Found {dup_count:,} duplicate rows — removing …")
        df = df.drop_duplicates(subset=subset)
        log.info(f"  Remaining: {df.shape[0]:,} rows")
    else:
        log.info("  No duplicates found ✓")
    return df


def step_flag_outliers(
    df: pd.DataFrame, cols: list[str] | None = None
) -> pd.DataFrame:
    """Flag outliers in numeric columns using the IQR method (non-destructive)."""
    log.info("[STEP] Outlier flagging (IQR method, non-destructive) …")
    if cols is None:
        cols = df.select_dtypes(include='number').columns.tolist()

    if not cols:
        log.info("  No numeric columns to evaluate ✓")
        return df

    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        flag_col = f'is_outlier_{col}'
        df[flag_col] = ~df[col].between(lo, hi)
        n = df[flag_col].sum()
        pct = n / len(df) * 100
        level = logging.WARNING if pct > 5 else logging.INFO
        log.log(level, f"  {col:<30} {n:>7,} outliers ({pct:.2f}%)  "
                       f"bounds: [{lo:.2f}, {hi:.2f}]")

    return df


def step_dtype_report(df: pd.DataFrame) -> None:
    """Print the inferred dtypes for each column."""
    log.info("[STEP] Data type report …")
    for col in df.columns:
        n_unique = df[col].nunique()
        dtype = str(df[col].dtype)
        log.info(f"  {col:<35} {dtype:<15} ({n_unique:,} unique)")


def step_preview(df: pd.DataFrame, n: int = 8) -> None:
    """Print a preview of the first N rows."""
    log.info(f"[STEP] Preview — first {n} rows …")
    with pd.option_context('display.max_columns', None):
        log.info("\n" + df.head(n).to_string())


def step_summary_stats(df: pd.DataFrame) -> None:
    """Print summary statistics for numeric and categorical columns."""
    log.info("[STEP] Summary statistics …")

    num_cols = df.select_dtypes(include='number').columns
    if len(num_cols):
        log.info("  ── Numeric columns ──")
        stats = df[num_cols].describe().T
        for col in stats.index:
            row = stats.loc[col]
            log.info(f"    {col:<30} mean={row['mean']:>12,.2f}  "
                     f"min={row['min']:>12,.2f}  max={row['max']:>12,.2f}  "
                     f"std={row['std']:>10,.2f}")

    cat_cols = df.select_dtypes(include='object').columns
    if len(cat_cols):
        log.info("  ── Categorical columns (top 5 values each) ──")
        for col in cat_cols[:8]:
            vc = df[col].value_counts()
            log.info(f"    {col} ({df[col].nunique():,} unique)")
            for val, cnt in vc.head(5).items():
                log.info(f"      {str(val):<30} {cnt:>8,}")


# ════════════════════════════════════════════════════════════════════
#  Main Pipeline
# ════════════════════════════════════════════════════════════════════
def run_pipeline(
    filepath: str,
    sep: str = ',',
    drop_pct: float = 0.5,
    skip_outliers: bool = False,
    skip_dedup: bool = False,
    encoding: str = 'utf-8',
) -> None:
    started_at = datetime.now()

    log.info("═" * 72)
    log.info("  GENERAL-PURPOSE CSV CLEANING PIPELINE")
    log.info(f"  Started:    {started_at:%Y-%m-%d %H:%M:%S}")
    log.info(f"  Input file: {filepath}")
    log.info(f"  Options:    sep='{sep}'  drop_pct={drop_pct}  "
             f"encoding={encoding}  "
             f"outliers={'off' if skip_outliers else 'on'}  "
             f"dedup={'off' if skip_dedup else 'on'}")
    log.info("═" * 72)

    steps_log = []

    # ── Load ──────────────────────────────────────────────────────
    df = step_load(filepath, sep, encoding)
    step_dtype_report(df)
    step_preview(df)

    # Capture BEFORE snapshot
    log.info("")
    log.info("── Capturing BEFORE snapshot ──")
    before_profile = profile_snapshot(df, 'BEFORE')
    step_audit_missing(df)
    log.info("")

    # ── Clean: sparse columns ─────────────────────────────────────
    df, dropped_cols = step_drop_sparse_cols_v2(df, threshold=drop_pct)
    if dropped_cols:
        steps_log.append(f"Dropped {len(dropped_cols)} sparse column(s): "
                         f"{', '.join(dropped_cols)}")

    # ── Clean: empty rows ────────────────────────────────────────
    rows_before = df.shape[0]
    df = step_drop_empty_rows(df)
    empty_removed = rows_before - df.shape[0]
    if empty_removed:
        steps_log.append(f"Removed {empty_removed:,} fully-empty row(s)")

    # ── Clean: duplicates ────────────────────────────────────────
    if not skip_dedup:
        rows_before = df.shape[0]
        df = step_remove_duplicates(df)
        dups_removed = rows_before - df.shape[0]
        if dups_removed:
            steps_log.append(f"Removed {dups_removed:,} duplicate row(s)")

    # ── Flag: outliers ────────────────────────────────────────────
    if not skip_outliers:
        df = step_flag_outliers(df)
        outlier_cols = [c for c in df.columns if c.startswith('is_outlier_')]
        total_outliers = sum(df[c].sum() for c in outlier_cols)
        if total_outliers:
            steps_log.append(f"Flagged {total_outliers:,} outlier values across "
                             f"{len(outlier_cols)} numeric column(s) (non-destructive)")

    log.info("")

    # ── After snapshot & comparison ───────────────────────────────
    log.info("── Capturing AFTER snapshot ──")
    after_profile = profile_snapshot(df, 'AFTER')

    log.info("")
    print_comparison_table(before_profile, after_profile)

    # ── Summary stats ─────────────────────────────────────────────
    log.info("")
    step_summary_stats(df)

    # ── Persist outputs ───────────────────────────────────────────
    log.info("")
    log.info("── Persisting outputs ──")

    stem = Path(filepath).stem
    out_dir = Path(filepath).parent

    out_csv = out_dir / f"{stem}_cleaned.csv"
    df.to_csv(out_csv, index=False)
    log.info(f"  Cleaned data:  {out_csv}")

    report = {
        'pipeline': 'general_purpose_csv_cleaner',
        'started_at': started_at.isoformat(),
        'finished_at': datetime.now().isoformat(),
        'duration_sec': round((datetime.now() - started_at).total_seconds(), 2),
        'input_file': filepath,
        'output_file': str(out_csv),
        'options': {
            'sep': sep,
            'drop_pct': drop_pct,
            'skip_outliers': skip_outliers,
            'skip_dedup': skip_dedup,
            'encoding': encoding,
        },
        'steps_executed': steps_log,
        'before': before_profile,
        'after': after_profile,
    }
    report_path = LOG_DIR / f'{stem}_cleaning_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)
    log.info(f"  Report JSON:   {report_path}")
    log.info(f"  Log file:      {LOG_FILE}")

    elapsed = datetime.now() - started_at
    log.info("")
    log.info("═" * 72)
    log.info(f"  PIPELINE COMPLETE — {elapsed.total_seconds():.1f}s")
    log.info(f"  Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    log.info("═" * 72)


# ════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ════════════════════════════════════════════════════════════════════
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Import and clean any CSV file with pandas. '
                    'Includes logging, profiling, and before/after comparison.'
    )
    p.add_argument('filepath', nargs='?',
                  help='Path to the CSV file. If omitted, you will be prompted.')
    p.add_argument('--sep', default=',',
                  help='Column delimiter (default: ",").')
    p.add_argument('--drop-pct', type=float, default=0.5,
                  help='Drop columns with more than this fraction missing '
                       '(default: 0.5 = 50%%).')
    p.add_argument('--no-outliers', action='store_true',
                  help='Skip outlier flagging step.')
    p.add_argument('--no-dedup', action='store_true',
                  help='Skip duplicate removal step.')
    p.add_argument('--encoding', default='utf-8',
                  help='File encoding (default: utf-8).')
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    filepath = args.filepath
    if filepath is None:
        filepath = input("Enter the path to your CSV file: ").strip()
        if not filepath:
            log.error("No file path provided. Exiting.")
            sys.exit(1)

    if not Path(filepath).exists():
        log.error(f"File not found: {filepath}")
        sys.exit(1)

    run_pipeline(
        filepath=filepath,
        sep=args.sep,
        drop_pct=args.drop_pct,
        skip_outliers=args.no_outliers,
        skip_dedup=args.no_dedup,
        encoding=args.encoding,
    )


if __name__ == '__main__':
    main()