"""
Extended CSV Import & Cleaning Script (with Logging & Profiling)
Dataset: Global Historical Climatology Network (landtempssample.csv)

Features:
    • Structured logging to console + file
    • Before/after profiling report
    • Missing-value audit, duplicate removal, outlier flagging,
      range validation, and categorical spot-checks
    • Cleaning summary persisted to JSON
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

# ════════════════════════════════════════════════════════════════════
#  Logging Configuration
# ════════════════════════════════════════════════════════════════════
LOG_DIR = Path('logs')
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / f"landtemps_clean_{datetime.now():%Y%m%d_%H%M%S}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s │ %(levelname)-7s │ %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger('landtemps')

# ════════════════════════════════════════════════════════════════════
#  Display Settings
# ════════════════════════════════════════════════════════════════════
pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.width', 130)
pd.set_option('display.max_columns', 25)
pd.set_option('display.expand_frame_repr', False)

# ════════════════════════════════════════════════════════════════════
#  Helper: Profile a DataFrame snapshot
# ════════════════════════════════════════════════════════════════════
def profile_snapshot(df: pd.DataFrame, label: str) -> dict:
    """Capture a lightweight profile of the DataFrame for before/after comparison."""
    snap = {
        'label': label,
        'rows': int(df.shape[0]),
        'cols': int(df.shape[1]),
        'memory_mb': round(df.memory_usage(deep=True).sum() / 1_048_576, 2),
        'missing_total': int(df.isnull().sum().sum()),
        'duplicate_rows': int(df.duplicated().sum()),
        'numeric_cols': {
            c: {
                'min': float(df[c].min()) if not df[c].isnull().all() else None,
                'max': float(df[c].max()) if not df[c].isnull().all() else None,
                'mean': round(float(df[c].mean()), 2) if not df[c].isnull().all() else None,
                'std': round(float(df[c].std()), 2) if not df[c].isnull().all() else None,
                'missing': int(df[c].isnull().sum()),
            }
            for c in df.select_dtypes(include='number').columns
        },
    }
    log.info(f"Profile [{label}]: {snap['rows']:,} rows × {snap['cols']} cols "
             f"| {snap['missing_total']:,} missing | "
             f"{snap['duplicate_rows']:,} duplicates | {snap['memory_mb']} MB")
    return snap


def print_profile_comparison(before: dict, after: dict) -> None:
    """Pretty-print a before/after comparison table."""
    banner = "╔" + "═" * 66 + "╗"
    log.info(banner)
    log.info("║" + "  BEFORE vs AFTER CLEANING COMPARISON".center(66) + "║")
    log.info("╠" + "═" * 66 + "╣")

    metrics = [
        ('Rows',           before['rows'],               after['rows']),
        ('Columns',        before['cols'],               after['cols']),
        ('Total missing',  before['missing_total'],       after['missing_total']),
        ('Duplicates',     before['duplicate_rows'],      after['duplicate_rows']),
        ('Memory (MB)',    before['memory_mb'],            after['memory_mb']),
    ]

    for name, b, a in metrics:
        delta = a - b
        arrow = '→' if delta == 0 else ('↓' if delta < 0 else '↑')
        log.info(f"║  {name:<20} {b:>14,}  {arrow}  {a:>14,}  "
                 f"({delta:+,})".ljust(36) + "║")

    log.info("╠" + "═" * 66 + "╣")
    log.info("║" + "  NUMERIC COLUMN DETAILS".center(66) + "║")
    log.info("╠" + "═" * 66 + "╣")

    num_before = before.get('numeric_cols', {})
    num_after = after.get('numeric_cols', {})
    for col in num_after:
        b_data = num_before.get(col, {})
        a_data = num_after.get(col, {})

        def fmt(val, kind='min'):
            if val is None:
                return f"{'N/A':>10}"
            return f"{val:>10,.2f}"

        log.info(
            f"║  {col:<16}"
            f" min: {fmt(b_data.get('min'))} → {fmt(a_data.get('min'))}   "
            f" max: {fmt(b_data.get('max'))} → {fmt(a_data.get('max'))}   "
            f" miss: {b_data.get('missing',0):>6,} → {a_data.get('missing',0):>6,}"
        )

    log.info("╚" + "═" * 66 + "╝")


# ════════════════════════════════════════════════════════════════════
#  Main Pipeline
# ════════════════════════════════════════════════════════════════════
def main():
    started_at = datetime.now()
    log.info("═" * 70)
    log.info("  LAND TEMPERATURES — EXTENDED CLEANING PIPELINE")
    log.info(f"  Started: {started_at:%Y-%m-%d %H:%M:%S}")
    log.info("═" * 70)

    # ── Step 1: Load ──────────────────────────────────────────────
    log.info("[1/8] Loading CSV …")

    df = pd.read_csv(
        'data/landtempssample.csv',
        names=[
            'stationid', 'year', 'month', 'avgtemp', 'latitude',
            'longitude', 'elevation', 'station', 'countryid', 'country'
        ],
        skiprows=1,
        parse_dates=[['month', 'year']],
        low_memory=False,
    )
    df.rename(columns={'month_year': 'measuredate'}, inplace=True)
    log.info(f"      Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Capture BEFORE snapshot
    before_profile = profile_snapshot(df, 'BEFORE')

    # ── Step 2: Dtype inspection ─────────────────────────────────
    log.info("[2/8] Inspecting data types …")
    for col in df.columns:
        log.info(f"      {col:<20} {str(df[col].dtype)}")

    # ── Step 3: Missing-value audit ──────────────────────────────
    log.info("[3/8] Auditing missing values …")
    missing_summary = df.isnull().sum()
    missing_pct = (missing_summary / len(df) * 100).round(2)
    for col in df.columns:
        if missing_summary[col] > 0:
            log.warning(f"      {col:<20} {missing_summary[col]:>7,} missing "
                        f"({missing_pct[col]:.2f}%)")

    # ── Step 4: Drop rows missing the key column ─────────────────
    log.info("[4/8] Dropping rows with missing avgtemp …")
    rows_before = df.shape[0]
    df.dropna(subset=['avgtemp'], inplace=True)
    dropped = rows_before - df.shape[0]
    log.info(f"      Removed {dropped:,} rows ({dropped/rows_before*100:.1f}%)")

    # ── Step 5: Duplicate detection & removal ────────────────────
    log.info("[5/8] Checking for duplicates …")
    dup_count = df.duplicated().sum()
    if dup_count:
        log.warning(f"      Found {dup_count:,} duplicates — removing …")
        df.drop_duplicates(inplace=True)
    else:
        log.info("      No duplicates found.")

    # ── Step 6: Range validation (sanity checks) ──────────────────
    log.info("[6/8] Validating numeric ranges …")

    checks = {
        'avgtemp':   (-90, 60),
        'latitude':  (-90, 90),
        'longitude': (-180, 180),
        'elevation': (-500, 9000),
    }

    invalid_masks = {}
    for col, (lo, hi) in checks.items():
        invalid = ~df[col].between(lo, hi)
        n_invalid = invalid.sum()
        invalid_masks[col] = invalid
        if n_invalid:
            log.warning(f"      {col:<16} {n_invalid:>5} rows outside "
                        f"[{lo}, {hi}]")
        else:
            log.info(f"      {col:<16} all values within [{lo}, {hi}] ✓")

    combined_invalid = pd.DataFrame(invalid_masks).any(axis=1)
    n_combined = combined_invalid.sum()
    if n_combined:
        log.warning(f"      Removing {n_combined:,} rows failing "
                    f"range validation …")
        df = df[~combined_invalid].copy()
    else:
        log.info("      All rows passed range validation ✓")

    # ── Step 7: Outlier flagging (IQR) ────────────────────────────
    log.info("[7/8] Flagging outliers (IQR method) …")
    q1 = df['avgtemp'].quantile(0.25)
    q3 = df['avgtemp'].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df['is_outlier_avgtemp'] = ~df['avgtemp'].between(lower, upper)
    n_outliers = df['is_outlier_avgtemp'].sum()
    log.info(f"      avgtemp IQR bounds: {lower:.2f} – {upper:.2f} °C | "
             f"{n_outliers:,} outliers flagged (non-destructive)")

    # Also flag elevation outliers
    for col in ['elevation', 'latitude', 'longitude']:
        q1c = df[col].quantile(0.25)
        q3c = df[col].quantile(0.75)
        iqrc = q3c - q1c
        loc = q1c - 1.5 * iqrc
        hic = q3c + 1.5 * iqrc
        flag = f'is_outlier_{col}'
        df[flag] = ~df[col].between(loc, hic)
        n = df[flag].sum()
        log.info(f"      {col:<16} IQR bounds: {loc:.2f} – {hic:.2f} | "
                 f"{n:,} outliers flagged")

    # ── Step 8: Final summary & persist ───────────────────────────
    log.info("[8/8] Generating final report & persisting …")

    after_profile = profile_snapshot(df, 'AFTER')

    # Print before/after comparison
    print_profile_comparison(before_profile, after_profile)

    # Categorical spot-check
    log.info("")
    log.info("── Top 10 countries by record count ──")
    for country, count in df['country'].value_counts().head(10).items():
        log.info(f"  {country:<25} {count:>8,}")

    # Numeric summary stats
    log.info("")
    log.info("── Summary statistics (avgtemp) ──")
    stats = df['avgtemp'].describe()
    for stat_name in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']:
        log.info(f"  {stat_name:<8} {stats[stat_name]:>12,.2f}")

    # Save cleaned data
    out_csv = 'data/landtemps_cleaned.csv'
    df.to_csv(out_csv, index=False)
    log.info(f"Cleaned data saved to: {out_csv}")

    # Save cleaning report as JSON
    report = {
        'pipeline': 'landtemps_extended',
        'started_at': started_at.isoformat(),
        'finished_at': datetime.now().isoformat(),
        'duration_sec': round((datetime.now() - started_at).total_seconds(), 2),
        'input_file': 'data/landtempssample.csv',
        'output_file': out_csv,
        'before': before_profile,
        'after': after_profile,
        'steps_executed': [
            f'Dropped {dropped:,} rows missing avgtemp',
            f'Removed {dup_count:,} duplicates',
            f'Removed {n_combined:,} rows failing range validation',
            f'Flagged {n_outliers:,} avgtemp outliers (IQR)',
        ],
    }
    report_path = LOG_DIR / 'last_cleaning_report_landtemps.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)
    log.info(f"Cleaning report saved to: {report_path}")

    elapsed = datetime.now() - started_at
    log.info("")
    log.info("═" * 70)
    log.info(f"  PIPELINE COMPLETE — {elapsed.total_seconds():.1f}s")
    log.info(f"  Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    log.info(f"  Log file:    {LOG_FILE}")
    log.info("═" * 70)


if __name__ == '__main__':
    main()