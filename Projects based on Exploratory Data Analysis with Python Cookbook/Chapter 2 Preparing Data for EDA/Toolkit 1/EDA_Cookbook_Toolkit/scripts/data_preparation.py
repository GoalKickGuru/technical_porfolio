"""
Data Preparation Module for Exploratory Data Analysis
=====================================================
Reusable functions implementing and extending the material from
"Exploratory Data Analysis with Python Cookbook" Chapter 2:
Preparing Data for EDA.

Covers:
- Grouping data (groupby + aggregations)
- Appending / concatenating data (vertical & horizontal)
- Merging data (joins on keys)
- Sorting data
- Categorizing / binning data
- Removing duplicate data
- Dropping rows and columns
- Changing data formats / dtypes
- Replacing values
- Dealing with missing values

Enhancements beyond the book recipes:
- Consistent error handling and validation helpers
- Flexible multi-column / multi-agg groupby
- Left / right / outer / inner merges with indicator
- Multi-key sorting and custom sort orders
- Quantile-based and equal-width binning helpers
- Duplicate detection reports (not only drop)
- Safe dtype casting with fill strategies
- Multiple missing-value strategies (drop, fill mean/median/mode, indicator)
- Convenience pipeline function that returns a cleaned DataFrame + audit log
- Logging / summary reporting helpers

Dependencies: pandas, numpy
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

ArrayLike = Union[pd.Series, np.ndarray, List, tuple]
DataFrameLike = pd.DataFrame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_df(data: Any, name: str = "data") -> pd.DataFrame:
    """Convert common inputs to DataFrame or raise a clear error."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, (pd.Series, dict, list, np.ndarray)):
        return pd.DataFrame(data)
    raise TypeError(f"{name} must be a pandas DataFrame (or convertible), got {type(data)}")


def _validate_columns(df: pd.DataFrame, cols: Sequence[str], context: str = "") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found{(' in ' + context) if context else ''}: {missing}")


# ---------------------------------------------------------------------------
# 1. Grouping
# ---------------------------------------------------------------------------

def group_and_aggregate(
    df: pd.DataFrame,
    by: Union[str, List[str]],
    agg: Union[str, List[str], Dict[str, Union[str, List[str]]]],
    as_index: bool = False,
) -> pd.DataFrame:
    """
    Group data by one or more columns and compute aggregations.

    Parameters
    ----------
    df : DataFrame
    by : str or list of str
        Column(s) to group by (usually categorical).
    agg : str, list, or dict
        Aggregation(s). Examples:
        - "mean"
        - ["mean", "sum", "count"]
        - {"NumStorePurchases": "mean", "Income": ["mean", "median"]}
    as_index : bool
        Whether to keep group keys as index (default False for flat result).

    Returns
    -------
    DataFrame with grouped aggregations.
    """
    df = _ensure_df(df)
    if isinstance(by, str):
        by = [by]
    _validate_columns(df, by, "groupby keys")

    result = df.groupby(by, as_index=as_index).agg(agg)
    # Flatten multi-level columns if present
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = [
            "_".join(str(c) for c in col if c != "").strip("_")
            for col in result.columns.values
        ]
    return result.reset_index() if as_index else result


def group_mean(df: pd.DataFrame, by: str, value_col: str) -> pd.Series:
    """Simple mean aggregation (matches book recipe style)."""
    df = _ensure_df(df)
    _validate_columns(df, [by, value_col])
    return df.groupby(by)[value_col].mean()


# ---------------------------------------------------------------------------
# 2. Appending / Vertical Concat
# ---------------------------------------------------------------------------

def append_dataframes(
    dfs: Sequence[pd.DataFrame],
    ignore_index: bool = True,
    sort: bool = False,
) -> pd.DataFrame:
    """
    Append (stack) multiple DataFrames vertically (row-wise).

    Equivalent to pd.concat(..., axis=0). Useful when combining
    samples or batches that share the same columns.
    """
    if not dfs:
        raise ValueError("At least one DataFrame is required")
    cleaned = [_ensure_df(d) for d in dfs]
    return pd.concat(cleaned, axis=0, ignore_index=ignore_index, sort=sort)


# ---------------------------------------------------------------------------
# 3. Concatenating (Horizontal)
# ---------------------------------------------------------------------------

def concatenate_dataframes(
    dfs: Sequence[pd.DataFrame],
    axis: int = 1,
    join: str = "outer",
    ignore_index: bool = False,
) -> pd.DataFrame:
    """
    Concatenate DataFrames along rows (axis=0) or columns (axis=1).

    When axis=1 the DataFrames are assumed to be aligned on the index
    (typical for side-by-side feature sets that share the same rows).
    """
    cleaned = [_ensure_df(d) for d in dfs]
    return pd.concat(cleaned, axis=axis, join=join, ignore_index=ignore_index)


# ---------------------------------------------------------------------------
# 4. Merging / Joining
# ---------------------------------------------------------------------------

def merge_dataframes(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: Optional[Union[str, List[str]]] = None,
    left_on: Optional[Union[str, List[str]]] = None,
    right_on: Optional[Union[str, List[str]]] = None,
    how: str = "inner",
    suffixes: Tuple[str, str] = ("_x", "_y"),
    indicator: bool = False,
) -> pd.DataFrame:
    """
    Merge two DataFrames on common key(s) (SQL-style join).

    Parameters
    ----------
    how : {'inner', 'left', 'right', 'outer', 'cross'}
    indicator : bool
        If True, adds a column `_merge` showing source of each row.
    """
    left = _ensure_df(left, "left")
    right = _ensure_df(right, "right")
    return pd.merge(
        left,
        right,
        on=on,
        left_on=left_on,
        right_on=right_on,
        how=how,
        suffixes=suffixes,
        indicator=indicator,
    )


# ---------------------------------------------------------------------------
# 5. Sorting
# ---------------------------------------------------------------------------

def sort_dataframe(
    df: pd.DataFrame,
    by: Union[str, List[str]],
    ascending: Union[bool, List[bool]] = True,
    na_position: str = "last",
) -> pd.DataFrame:
    """
    Sort DataFrame by one or more columns.

    Supports mixed ascending/descending directions.
    """
    df = _ensure_df(df)
    if isinstance(by, str):
        by = [by]
    _validate_columns(df, by, "sort keys")
    return df.sort_values(by=by, ascending=ascending, na_position=na_position).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. Categorizing / Binning
# ---------------------------------------------------------------------------

def categorize_numeric(
    series: ArrayLike,
    bins: Union[int, Sequence[float]] = 3,
    labels: Optional[Sequence[str]] = None,
    right: bool = True,
    include_lowest: bool = True,
) -> pd.Series:
    """
    Bin a numeric Series into discrete categories using pd.cut.

    Parameters
    ----------
    bins : int or sequence of edges
        Number of equal-width bins, or explicit bin edges.
    labels : sequence of str, optional
        Labels for the resulting bins.
    """
    s = pd.Series(series) if not isinstance(series, pd.Series) else series
    return pd.cut(s, bins=bins, labels=labels, right=right, include_lowest=include_lowest)


def categorize_by_quantiles(
    series: ArrayLike,
    q: int = 4,
    labels: Optional[Sequence[str]] = None,
) -> pd.Series:
    """Bin into quantile-based categories (equal frequency) using pd.qcut."""
    s = pd.Series(series) if not isinstance(series, pd.Series) else series
    return pd.qcut(s, q=q, labels=labels, duplicates="drop")


# ---------------------------------------------------------------------------
# 7. Removing Duplicates
# ---------------------------------------------------------------------------

def drop_duplicates(
    df: pd.DataFrame,
    subset: Optional[Sequence[str]] = None,
    keep: str = "first",
) -> pd.DataFrame:
    """
    Remove duplicate rows.

    Parameters
    ----------
    subset : list of column names or None
        Consider only these columns for identifying duplicates.
    keep : {'first', 'last', False}
        Which duplicates to keep.
    """
    df = _ensure_df(df)
    if subset is not None:
        _validate_columns(df, subset, "duplicate subset")
    return df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)


def duplicate_report(df: pd.DataFrame, subset: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Return a small audit of duplicate counts."""
    df = _ensure_df(df)
    if subset is not None:
        _validate_columns(df, subset)
    total = len(df)
    n_dup = df.duplicated(subset=subset).sum()
    n_unique = df.drop_duplicates(subset=subset).shape[0]
    return {
        "total_rows": total,
        "duplicate_rows": int(n_dup),
        "unique_rows": n_unique,
        "duplicate_pct": round(100.0 * n_dup / total, 2) if total else 0.0,
    }


# ---------------------------------------------------------------------------
# 8. Dropping Rows / Columns
# ---------------------------------------------------------------------------

def drop_rows(
    df: pd.DataFrame,
    labels: Optional[Sequence[Any]] = None,
    index: Optional[Sequence[Any]] = None,
    axis: int = 0,
) -> pd.DataFrame:
    """Drop rows by label or position."""
    df = _ensure_df(df)
    if labels is not None:
        return df.drop(labels=labels, axis=axis)
    if index is not None:
        return df.drop(index=index)
    raise ValueError("Provide either labels or index")


def drop_columns(df: pd.DataFrame, columns: Union[str, Sequence[str]]) -> pd.DataFrame:
    """Drop one or more columns."""
    df = _ensure_df(df)
    if isinstance(columns, str):
        columns = [columns]
    _validate_columns(df, columns, "drop columns")
    return df.drop(columns=columns)


# ---------------------------------------------------------------------------
# 9. Changing Data Format / Dtypes
# ---------------------------------------------------------------------------

def change_dtype(
    df: pd.DataFrame,
    column: str,
    dtype: Any,
    errors: str = "raise",
) -> pd.DataFrame:
    """
    Cast a column to a new dtype.

    Common targets: 'int64', 'float64', 'str', 'category', 'datetime64[ns]'
    """
    df = _ensure_df(df)
    _validate_columns(df, [column])
    out = df.copy()
    out[column] = out[column].astype(dtype, errors=errors)
    return out


def to_numeric_safe(
    series: ArrayLike,
    errors: str = "coerce",
    downcast: Optional[str] = None,
) -> pd.Series:
    """Convert to numeric, coercing invalid values to NaN by default."""
    s = pd.Series(series) if not isinstance(series, pd.Series) else series
    return pd.to_numeric(s, errors=errors, downcast=downcast)


# ---------------------------------------------------------------------------
# 10. Replacing Values
# ---------------------------------------------------------------------------

def replace_values(
    df: pd.DataFrame,
    column: str,
    to_replace: Any,
    value: Any,
    regex: bool = False,
) -> pd.DataFrame:
    """Replace values in a column (supports list/dict mapping)."""
    df = _ensure_df(df)
    _validate_columns(df, [column])
    out = df.copy()
    out[column] = out[column].replace(to_replace=to_replace, value=value, regex=regex)
    return out


def map_values(
    df: pd.DataFrame,
    column: str,
    mapping: Dict[Any, Any],
    default: Any = None,
) -> pd.DataFrame:
    """Map values using a dictionary; optionally fill unmapped with default."""
    df = _ensure_df(df)
    _validate_columns(df, [column])
    out = df.copy()
    mapped = out[column].map(mapping)
    if default is not None:
        mapped = mapped.fillna(default)
    out[column] = mapped
    return out


# ---------------------------------------------------------------------------
# 11. Missing Values
# ---------------------------------------------------------------------------

def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy summary of missing values per column."""
    df = _ensure_df(df)
    total = len(df)
    null_count = df.isnull().sum()
    report = pd.DataFrame({
        "column": null_count.index,
        "missing_count": null_count.values,
        "missing_pct": (100.0 * null_count / total).round(2).values if total else 0,
        "dtype": df.dtypes.values,
    })
    return report.sort_values("missing_count", ascending=False).reset_index(drop=True)


def drop_missing(
    df: pd.DataFrame,
    how: str = "any",
    subset: Optional[Sequence[str]] = None,
    thresh: Optional[int] = None,
) -> pd.DataFrame:
    """Drop rows with missing values."""
    df = _ensure_df(df)
    return df.dropna(how=how, subset=subset, thresh=thresh).reset_index(drop=True)


def fill_missing(
    df: pd.DataFrame,
    strategy: str = "mean",
    columns: Optional[Sequence[str]] = None,
    fill_value: Any = None,
) -> pd.DataFrame:
    """
    Fill missing values.

    strategy : {'mean', 'median', 'mode', 'constant', 'ffill', 'bfill'}
    fill_value : used when strategy='constant'
    """
    df = _ensure_df(df)
    out = df.copy()
    cols = list(columns) if columns is not None else list(out.columns)

    for col in cols:
        if col not in out.columns:
            continue
        if strategy == "mean" and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].fillna(out[col].mean())
        elif strategy == "median" and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].fillna(out[col].median())
        elif strategy == "mode":
            mode_val = out[col].mode(dropna=True)
            if not mode_val.empty:
                out[col] = out[col].fillna(mode_val.iloc[0])
        elif strategy == "constant":
            out[col] = out[col].fillna(fill_value)
        elif strategy == "ffill":
            out[col] = out[col].ffill()
        elif strategy == "bfill":
            out[col] = out[col].bfill()
        else:
            # leave as-is for unsupported combinations
            pass
    return out


# ---------------------------------------------------------------------------
# Convenience pipeline + reporting
# ---------------------------------------------------------------------------

def prepare_pipeline(
    df: pd.DataFrame,
    drop_cols: Optional[Sequence[str]] = None,
    drop_duplicates_subset: Optional[Sequence[str]] = None,
    fill_strategy: Optional[str] = "median",
    fill_columns: Optional[Sequence[str]] = None,
    sort_by: Optional[Union[str, List[str]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Simple end-to-end preparation pipeline.

    Returns cleaned DataFrame and an audit dictionary.
    """
    df = _ensure_df(df)
    audit: Dict[str, Any] = {"original_shape": df.shape}

    if drop_cols:
        df = drop_columns(df, drop_cols)
        audit["dropped_columns"] = list(drop_cols)

    if drop_duplicates_subset is not None or True:  # always check
        rep = duplicate_report(df, subset=drop_duplicates_subset)
        audit["duplicates"] = rep
        df = drop_duplicates(df, subset=drop_duplicates_subset)

    if fill_strategy:
        before = df.isnull().sum().sum()
        df = fill_missing(df, strategy=fill_strategy, columns=fill_columns)
        after = df.isnull().sum().sum()
        audit["missing_filled"] = int(before - after)

    if sort_by:
        df = sort_dataframe(df, by=sort_by)

    audit["final_shape"] = df.shape
    return df, audit


def print_audit(audit: Dict[str, Any], title: str = "Data Preparation Audit") -> None:
    """Pretty-print an audit dictionary."""
    print("=" * 60)
    print(title)
    print("=" * 60)
    for k, v in audit.items():
        print(f"{k:25}: {v}")
    print("=" * 60)
