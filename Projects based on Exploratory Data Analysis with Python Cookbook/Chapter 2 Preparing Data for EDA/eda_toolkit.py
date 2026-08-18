import json
import numpy as np
import pandas as pd
from scipy import stats


class ExtendedEDA:

  def __init__(self, df_or_path, target_cols=None):
    """Initialize dataset profiling workflow."""
    if isinstance(df_or_path, str):
      self.df = pd.read_csv(df_or_path)
    elif isinstance(df_or_path, pd.DataFrame):
      self.df = df_or_path.copy()
    else:
      raise ValueError("Input must be a file path or pandas DataFrame.")

    if target_cols is None:
      self.target_cols = self.df.select_dtypes(
          include=[np.number]
      ).columns.tolist()
    else:
      self.target_cols = target_cols

    self.summary_results = {}

  def generate_summary_statistics(self, column_name):
    """Calculates full summary statistic profile for a numeric column."""
    series = self.df[column_name].dropna()

    mean_val = float(np.mean(series))
    median_val = float(np.median(series))

    mode_res = stats.mode(series, keepdims=True)
    mode_val = float(mode_res.mode[0]) if len(mode_res.mode) > 0 else None

    var_val = float(np.var(series, ddof=0))
    std_val = float(np.std(series, ddof=0))

    max_val = float(np.max(series))
    min_val = float(np.min(series))
    range_val = max_val - min_val

    q1 = float(np.quantile(series, 0.25))
    q2 = float(np.quantile(series, 0.50))
    q3 = float(np.quantile(series, 0.75))
    iqr_val = float(stats.iqr(series))

    # Tukey's Fence Outlier Detection (1.5x IQR Rule)
    lower_fence = q1 - (1.5 * iqr_val)
    upper_fence = q3 + (1.5 * iqr_val)
    outliers_count = int(
        ((series < lower_fence) | (series > upper_fence)).sum()
    )

    stats_profile = {
        "mean": mean_val,
        "median": median_val,
        "mode": mode_val,
        "variance": var_val,
        "std_dev": std_val,
        "range": range_val,
        "quartiles": {"Q1": q1, "Q2": q2, "Q3": q3},
        "IQR": iqr_val,
        "outliers_count": outliers_count,
    }

    self.summary_results[column_name] = stats_profile
    return stats_profile

  def to_dataframe(self):
    """Converts metric results into a clean tabular DataFrame."""
    if not self.summary_results:
      for col in self.target_cols:
        self.generate_summary_statistics(col)

    records = []
    for col, res in self.summary_results.items():
      records.append({
          "Column": col,
          "Mean": res["mean"],
          "Median": res["median"],
          "Mode": res["mode"],
          "Variance": res["variance"],
          "Std Dev": res["std_dev"],
          "Range": res["range"],
          "Q1": res["quartiles"]["Q1"],
          "Q3": res["quartiles"]["Q3"],
          "IQR": res["IQR"],
          "Outliers Count": res["outliers_count"],
      })
    return pd.DataFrame(records)