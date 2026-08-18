"""
Summary Statistics Module for Exploratory Data Analysis
=======================================================
Reusable functions implementing and extending the material from
"Exploratory Data Analysis with Python Cookbook" Chapter 1:
Generating Summary Statistics.

Covers: Mean, Median, Mode, Variance, Standard Deviation, Range,
Percentiles, Quartiles, and Interquartile Range (IQR).

Enhancements:
- Support for pandas Series / DataFrame columns or numpy arrays
- Optional group-by analysis
- Robust handling of missing values
- Comparison of numpy vs pandas implementations (ddof differences)
- Convenience function that returns a full summary dictionary
- Simple text reporting helper

Dependencies: numpy, pandas, scipy
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats


ArrayLike = Union[pd.Series, np.ndarray, List[float], List[int]]


def _to_series(data: ArrayLike, name: str = "values") -> pd.Series:
    """Convert input to a clean pandas Series (drops NaN by default for most stats)."""
    if isinstance(data, pd.Series):
        s = data.copy()
    else:
        s = pd.Series(data, name=name)
    return s


def compute_mean(data: ArrayLike, skipna: bool = True) -> float:
    """
    Compute the arithmetic mean.

    Uses both numpy and pandas under the hood for illustration.
    Mean is sensitive to outliers.

    Parameters
    ----------
    data : array-like
        Numeric data.
    skipna : bool
        Whether to ignore NaN values (pandas behaviour).

    Returns
    -------
    float
        Mean value.
    """
    s = _to_series(data)
    # Prefer pandas for consistent skipna handling
    return float(s.mean(skipna=skipna))


def compute_median(data: ArrayLike, skipna: bool = True) -> float:
    """
    Compute the median (50th percentile).

    More robust to outliers than the mean.
    """
    s = _to_series(data)
    return float(s.median(skipna=skipna))


def compute_mode(data: ArrayLike, dropna: bool = True) -> Dict[str, Any]:
    """
    Compute the mode (most frequent value) using scipy.stats.mode.

    Returns a dictionary with mode value(s) and count(s).
    Works for numeric and categorical data.
    """
    s = _to_series(data)
    if dropna:
        s = s.dropna()
    if s.empty:
        return {"mode": None, "count": 0, "raw": None}

    # scipy 1.11+ returns ModeResult; keepdims for compatibility
    try:
        result = stats.mode(s, keepdims=True)
    except TypeError:
        result = stats.mode(s)

    mode_val = result.mode[0] if hasattr(result.mode, "__len__") else result.mode
    count_val = result.count[0] if hasattr(result.count, "__len__") else result.count
    return {
        "mode": mode_val,
        "count": int(count_val),
        "raw": result,
    }


def compute_variance(
    data: ArrayLike,
    ddof: int = 0,
    skipna: bool = True,
) -> Dict[str, float]:
    """
    Compute variance.

    Parameters
    ----------
    ddof : int
        Delta degrees of freedom. numpy default=0 (population),
        pandas default=1 (sample). Both are returned for comparison.
    """
    s = _to_series(data)
    np_var = float(np.var(s.dropna() if skipna else s, ddof=ddof))
    pd_var = float(s.var(skipna=skipna, ddof=ddof))
    return {
        "numpy_var": np_var,
        "pandas_var": pd_var,
        "ddof": ddof,
    }


def compute_std(
    data: ArrayLike,
    ddof: int = 0,
    skipna: bool = True,
) -> Dict[str, float]:
    """
    Compute standard deviation (square root of variance).

    Same ddof notes as variance.
    """
    s = _to_series(data)
    np_std = float(np.std(s.dropna() if skipna else s, ddof=ddof))
    pd_std = float(s.std(skipna=skipna, ddof=ddof))
    return {
        "numpy_std": np_std,
        "pandas_std": pd_std,
        "ddof": ddof,
    }


def compute_range(data: ArrayLike, skipna: bool = True) -> Dict[str, float]:
    """
    Compute the range (max - min).
    """
    s = _to_series(data)
    if skipna:
        s = s.dropna()
    data_max = float(s.max())
    data_min = float(s.min())
    return {
        "max": data_max,
        "min": data_min,
        "range": data_max - data_min,
    }


def compute_percentile(
    data: ArrayLike,
    q: float = 60,
    skipna: bool = True,
) -> float:
    """
    Compute a percentile (0-100 scale).

    Example: q=60 returns the 60th percentile.
    """
    s = _to_series(data)
    if skipna:
        s = s.dropna()
    return float(np.percentile(s, q))


def compute_quartile(
    data: ArrayLike,
    q: float = 0.75,
    skipna: bool = True,
) -> float:
    """
    Compute a quartile using numpy.quantile (0-1 scale).

    Common values: 0.25 (Q1), 0.5 (Q2/median), 0.75 (Q3).
    """
    s = _to_series(data)
    if skipna:
        s = s.dropna()
    return float(np.quantile(s, q))


def compute_iqr(
    data: ArrayLike,
    interpolation: str = "midpoint",
    skipna: bool = True,
) -> Dict[str, float]:
    """
    Compute the Interquartile Range (IQR = Q3 - Q1).

    Uses scipy.stats.iqr. Also returns the two quartiles for transparency.
    """
    s = _to_series(data)
    if skipna:
        s = s.dropna()
    iqr_val = float(stats.iqr(s, interpolation=interpolation))
    q1 = float(np.quantile(s, 0.25))
    q3 = float(np.quantile(s, 0.75))
    return {
        "iqr": iqr_val,
        "q1": q1,
        "q3": q3,
        "interpolation": interpolation,
    }


def full_summary(
    data: ArrayLike,
    percentiles: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Generate a comprehensive summary statistics dictionary.

    This is the main convenience function for EDA pipelines.
    """
    if percentiles is None:
        percentiles = [25, 50, 75, 90, 95, 99]

    s = _to_series(data).dropna()
    if s.empty:
        return {"error": "No valid numeric data"}

    mode_info = compute_mode(s)
    var_info = compute_variance(s, ddof=0)
    std_info = compute_std(s, ddof=0)
    range_info = compute_range(s)
    iqr_info = compute_iqr(s)

    summary = {
        "count": int(s.count()),
        "mean": compute_mean(s),
        "median": compute_median(s),
        "mode": mode_info["mode"],
        "mode_count": mode_info["count"],
        "variance_population": var_info["numpy_var"],
        "std_population": std_info["numpy_std"],
        "variance_sample_ddof1": float(s.var(ddof=1)),
        "std_sample_ddof1": float(s.std(ddof=1)),
        "min": range_info["min"],
        "max": range_info["max"],
        "range": range_info["range"],
        "iqr": iqr_info["iqr"],
        "q1": iqr_info["q1"],
        "q3": iqr_info["q3"],
        "percentiles": {f"p{int(p)}": compute_percentile(s, p) for p in percentiles},
        "skewness": float(s.skew()),
        "kurtosis": float(s.kurtosis()),
    }
    return summary


def summary_by_group(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
) -> pd.DataFrame:
    """
    Compute key summary statistics grouped by a categorical column.
    Useful extension for multi-country / multi-category analysis.
    """
    grouped = df.groupby(group_col)[value_col]
    result = grouped.agg(
        count="count",
        mean="mean",
        median="median",
        std="std",
        min="min",
        max="max",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
    )
    result["iqr"] = result["q75"] - result["q25"]
    result["range"] = result["max"] - result["min"]
    return result.reset_index()


def print_summary_report(summary: Dict[str, Any], title: str = "Summary Statistics") -> None:
    """Pretty-print a summary dictionary to the console."""
    print("=" * 60)
    print(title)
    print("=" * 60)
    for k, v in summary.items():
        if k == "percentiles" and isinstance(v, dict):
            print(f"{k}:")
            for pk, pv in v.items():
                print(f"  {pk}: {pv:,.4f}")
        else:
            if isinstance(v, float):
                print(f"{k}: {v:,.4f}")
            else:
                print(f"{k}: {v}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Demo / self-test when run as script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal self-test with synthetic data
    rng = np.random.default_rng(42)
    demo = pd.Series(rng.lognormal(mean=5, sigma=1.5, size=1000))
    demo = pd.concat([demo, pd.Series([0] * 50)])  # inject zeros

    print("=== Demo full_summary ===")
    s = full_summary(demo)
    print_summary_report(s, "Demo Dataset")

    print("\n=== Individual functions ===")
    print("Mean:", compute_mean(demo))
    print("Median:", compute_median(demo))
    print("Mode:", compute_mode(demo))
    print("Variance (ddof=0):", compute_variance(demo, ddof=0))
    print("Std (ddof=0):", compute_std(demo, ddof=0))
    print("Range:", compute_range(demo))
    print("60th percentile:", compute_percentile(demo, 60))
    print("Q3:", compute_quartile(demo, 0.75))
    print("IQR:", compute_iqr(demo))
