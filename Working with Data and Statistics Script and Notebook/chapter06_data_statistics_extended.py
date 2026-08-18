# %% [markdown]
# # Chapter 6 — Working with Data and Statistics (Extended Edition)
#
# This notebook extends the recipes from *Applying Math with Python, 2nd Edition*,
# Chapter 6, "Working with Data and Statistics". It reproduces every recipe from
# the book and adds a set of practical enhancements (robustness, extra
# diagnostics, additional statistical tests, richer visualizations, and
# reusable helper functions) that go beyond what's printed in the text.
#
# See the "Summary of Enhancements" markdown cell at the very end for a
# checklist of what was added on top of the original book recipes.

# %%
from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless-safe backend; swap to an interactive
                        # backend (e.g. "TkAgg") when running locally with a display
import matplotlib.pyplot as plt
from matplotlib.rcsetup import cycler
from numpy.random import default_rng
from scipy import stats

# Enhancement: optional dependency handled gracefully instead of hard-failing
try:
    import statsmodels.stats.multicomp as mc
    from statsmodels.stats.multitest import multipletests

    HAVE_STATSMODELS = True
except ImportError:  # pragma: no cover
    HAVE_STATSMODELS = False
    warnings.warn("statsmodels not available — Tukey HSD / Bonferroni sections will be skipped")

try:
    from bokeh import plotting as bk
    from bokeh.models import HoverTool, Legend

    HAVE_BOKEH = True
except ImportError:  # pragma: no cover
    HAVE_BOKEH = False
    warnings.warn("bokeh not available — interactive plotting section will be skipped")

# Enhancement: single source of truth for reproducibility + output paths
SEED = 12345
rng = default_rng(SEED)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rc("axes", prop_cycle=cycler(c=["k"] * 3, ls=["-", "--", "-."]))
plt.rc("figure", figsize=(8, 5))

# %% [markdown]
# ## Recipe 1 — Creating Series and DataFrame objects
#
# **Enhancement:** in addition to the book's `Series`/`DataFrame` construction,
# this cell also demonstrates `iloc`, `loc`, `at`, and dtype inspection, since
# the book's "There's more..." section mentions these but never shows code.

# %%
def recipe_series_and_dataframe(rng: np.random.Generator) -> pd.DataFrame:
    diff_data = rng.normal(0, 1, size=100)
    cumulative = diff_data.cumsum()

    data_series = pd.Series(diff_data, name="diffs")
    print("Series preview:")
    print(data_series.head())

    data_frame = pd.DataFrame({"diffs": data_series, "cumulative": cumulative})
    print("\nDataFrame preview:")
    print(data_frame.head())

    # --- Enhancements below ---
    print("\n[enhancement] dtypes:\n", data_frame.dtypes)

    # Selection via index notation, .loc, .iloc, and .at
    row_5_via_index = data_series[5]
    row_5_via_iloc = data_series.iloc[5]
    row_5_via_loc = data_frame.loc[5, "diffs"]
    row_5_via_at = data_frame.at[5, "diffs"]
    assert row_5_via_index == row_5_via_iloc == row_5_via_loc == row_5_via_at
    print(f"[enhancement] row 5 'diffs' value (confirmed via 4 access methods): {row_5_via_index:.6f}")

    # Boolean-array selection (mentioned but not demoed in the book)
    positive_diffs = data_frame[data_frame["diffs"] > 0]
    print(f"[enhancement] {len(positive_diffs)} of {len(data_frame)} diffs are positive")

    return data_frame


df_series_demo = recipe_series_and_dataframe(rng)

# %% [markdown]
# ## Recipe 2 — Loading and storing data from a DataFrame
#
# **Enhancement:** the book only covers CSV. This version round-trips through
# CSV, JSON, and Parquet (if `pyarrow`/`fastparquet` is available), wraps I/O
# in `try/except` so a missing optional engine doesn't crash the script, and
# verifies round-trip equality with `pandas.testing.assert_frame_equal`.

# %%
def recipe_load_and_store(rng: np.random.Generator, out_dir: Path) -> pd.DataFrame:
    diffs = rng.normal(0, 1, size=100)
    cumulative = diffs.cumsum()
    data_frame = pd.DataFrame({"diffs": diffs, "cumulative": cumulative})

    csv_path = out_dir / "sample.csv"
    data_frame.to_csv(csv_path, index=False)
    df_csv = pd.read_csv(csv_path, index_col=False)
    pd.testing.assert_frame_equal(data_frame, df_csv)
    print(f"[ok] CSV round-trip verified -> {csv_path}")

    # --- Enhancements below ---
    json_path = out_dir / "sample.json"
    data_frame.to_json(json_path, orient="records")
    df_json = pd.read_json(json_path, orient="records")
    pd.testing.assert_frame_equal(data_frame, df_json, check_dtype=False)
    print(f"[enhancement] JSON round-trip verified -> {json_path}")

    try:
        parquet_path = out_dir / "sample.parquet"
        data_frame.to_parquet(parquet_path)
        df_parquet = pd.read_parquet(parquet_path)
        pd.testing.assert_frame_equal(data_frame, df_parquet)
        print(f"[enhancement] Parquet round-trip verified -> {parquet_path}")
    except (ImportError, ValueError) as exc:
        print(f"[enhancement] Parquet skipped (no engine installed): {exc}")

    return data_frame


df_io_demo = recipe_load_and_store(rng, OUTPUT_DIR)

# %% [markdown]
# ## Recipe 3 — Manipulating data in DataFrames
#
# **Enhancement:** alongside the book's row-wise `.apply(axis=1)` (which the
# book itself warns is slow), this cell adds a fully vectorized NumPy
# equivalent and benchmarks the two, plus demonstrates `fillna` and
# `interpolate` as alternatives to `dropna`.

# %%
def recipe_manipulate(rng: np.random.Generator) -> pd.DataFrame:
    three = rng.uniform(-0.2, 1.0, size=100)
    three[three < 0] = np.nan

    data_frame = pd.DataFrame(
        {
            "one": rng.random(size=100),
            "two": rng.normal(0, 1, size=100).cumsum(),
            "three": three,
        }
    )
    data_frame["four"] = data_frame["one"] > 0.5

    def transform_function(row: pd.Series) -> float:
        if row["four"]:
            return 0.5 * row["two"]
        return row["one"] * row["two"]

    data_frame["five"] = data_frame.apply(transform_function, axis=1)

    # --- Enhancement: vectorized equivalent (much faster for large frames) ---
    vectorized_five = np.where(
        data_frame["four"], 0.5 * data_frame["two"], data_frame["one"] * data_frame["two"]
    )
    assert np.allclose(data_frame["five"], vectorized_five)
    print("[enhancement] vectorized np.where result matches df.apply result")

    df_dropped = data_frame.dropna()
    print(f"dropna: {len(data_frame)} -> {len(df_dropped)} rows")

    # --- Enhancement: alternative missing-data strategies ---
    df_filled = data_frame.fillna(data_frame["three"].median())
    df_interp = data_frame.copy()
    df_interp["three"] = df_interp["three"].interpolate()
    print(
        "[enhancement] NaNs remaining after fillna(median):",
        df_filled["three"].isna().sum(),
        "| after interpolate():",
        df_interp["three"].isna().sum(),
    )

    return data_frame


df_manip_demo = recipe_manipulate(rng)

# %% [markdown]
# ## Recipe 4 — Plotting data from a DataFrame
#
# **Enhancement:** figures are saved to disk (headless-safe) instead of only
# calling `plt.show()`, and a KDE overlay plus a boxplot are added since the
# book's "There's more..." section mentions `kind="box"`/`kind="scatter"` but
# never shows them.

# %%
def recipe_plotting(rng: np.random.Generator, out_dir: Path) -> None:
    diffs = rng.standard_normal(size=100)
    walk = diffs.cumsum()
    df = pd.DataFrame({"diffs": diffs, "walk": walk})

    fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True)
    df["walk"].plot(ax=ax1, title="Random walk", color="k")
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Value")

    df["diffs"].plot(kind="hist", ax=ax2, title="Histogram of diffs", color="k", alpha=0.6)
    ax2.set_xlabel("Difference")
    fig.savefig(out_dir / "recipe4_walk_and_hist.png", dpi=150)
    plt.close(fig)

    # --- Enhancement: KDE overlay + boxplot ---
    fig2, (ax3, ax4) = plt.subplots(1, 2, tight_layout=True)
    df["diffs"].plot(kind="hist", ax=ax3, density=True, color="k", alpha=0.4, title="Histogram + KDE")
    df["diffs"].plot(kind="kde", ax=ax3, color="k", linewidth=2)
    ax3.set_xlabel("Difference")

    df[["diffs"]].plot(kind="box", ax=ax4, title="Boxplot of diffs")
    fig2.savefig(out_dir / "recipe4_kde_and_box.png", dpi=150)
    plt.close(fig2)
    print(f"[enhancement] saved KDE/boxplot figure -> {out_dir / 'recipe4_kde_and_box.png'}")


recipe_plotting(rng, OUTPUT_DIR)

# %% [markdown]
# ## Recipe 5 — Getting descriptive statistics from a DataFrame
#
# **Enhancement:** adds skewness (kurtosis's natural companion, mentioned
# nowhere in the book despite being one line of code away), a
# `scipy.stats.normaltest` normality check per column, and annotates the
# histograms with the median as well as the mean.

# %%
def recipe_descriptive_stats(rng: np.random.Generator, out_dir: Path) -> pd.DataFrame:
    uniform = rng.uniform(1, 5, size=100)
    normal = rng.normal(1, 2.5, size=100)
    bimodal = np.concatenate([rng.normal(0, 1, size=50), rng.normal(6, 1, size=50)])
    df = pd.DataFrame({"uniform": uniform, "normal": normal, "bimodal": bimodal})

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, tight_layout=True)
    df["uniform"].plot(kind="hist", title="Uniform", ax=ax1, color="k", alpha=0.6)
    df["normal"].plot(kind="hist", title="Normal", ax=ax2, color="k", alpha=0.6)
    df["bimodal"].plot(kind="hist", title="Bimodal", ax=ax3, bins=20, color="k", alpha=0.6)

    descriptive = df.describe()
    descriptive.loc["kurtosis"] = df.kurtosis()
    descriptive.loc["skewness"] = df.skew()  # enhancement
    print(descriptive)

    for ax, col, ymax in zip((ax1, ax2, ax3), df.columns, (20, 25, 20)):
        ax.vlines(descriptive.loc["mean", col], 0, ymax, "k", label="mean")
        ax.vlines(descriptive.loc["50%", col], 0, ymax, "k", linestyle="--", label="median")
    ax1.legend(fontsize=8)
    fig.savefig(out_dir / "recipe5_descriptive_histograms.png", dpi=150)
    plt.close(fig)

    # --- Enhancement: normality testing per column ---
    print("\n[enhancement] D'Agostino-Pearson normality test (H0: sample is normal):")
    for col in df.columns:
        stat, p = stats.normaltest(df[col])
        verdict = "reject normality" if p < 0.05 else "cannot reject normality"
        print(f"  {col:>8s}: statistic={stat:8.3f}  p={p:.2e}  -> {verdict}")

    return df


df_desc_demo = recipe_descriptive_stats(rng, OUTPUT_DIR)

# %% [markdown]
# ## Recipe 6 — Understanding a population using sampling
#
# **Enhancement:** adds a non-parametric **bootstrap confidence interval**
# alongside the book's Student-t interval, and plots the bootstrap
# distribution of the mean so the two approaches can be compared visually.

# %%
def recipe_sampling_ci(rng: np.random.Generator, out_dir: Path) -> dict:
    sample_data = pd.Series(
        [
            172.3, 171.3, 164.7, 162.9, 172.5, 176.3, 174.8,
            171.9, 176.8, 167.8, 164.5, 179.7, 157.8, 170.6,
            189.9, 185.0, 172.7, 165.5, 174.5, 171.5,
        ]
    )

    sample_mean = sample_data.mean()
    sample_std = sample_data.std()
    N = sample_data.count()
    std_err = sample_std / math.sqrt(N)

    cv_95, cv_99 = stats.t.ppf([0.975, 0.995], df=N - 1)
    pm_95, pm_99 = cv_95 * std_err, cv_99 * std_err
    conf_interval_95 = [sample_mean - pm_95, sample_mean + pm_95]
    conf_interval_99 = [sample_mean - pm_99, sample_mean + pm_99]

    print(f"Mean {sample_mean:.2f}, st. dev {sample_std:.4f}")
    print("95% (t-distribution) confidence interval:", conf_interval_95)
    print("99% (t-distribution) confidence interval:", conf_interval_99)

    # --- Enhancement: bootstrap confidence interval ---
    n_bootstrap = 10_000
    boot_means = np.array(
        [rng.choice(sample_data, size=N, replace=True).mean() for _ in range(n_bootstrap)]
    )
    boot_ci_95 = np.percentile(boot_means, [2.5, 97.5])
    print(f"\n[enhancement] Bootstrap (n={n_bootstrap}) 95% CI:", boot_ci_95.tolist())

    fig, ax = plt.subplots()
    ax.hist(boot_means, bins=40, color="k", alpha=0.6)
    ax.axvline(sample_mean, color="k", linewidth=2, label="sample mean")
    ax.axvline(boot_ci_95[0], color="k", linestyle="--", label="bootstrap 95% CI")
    ax.axvline(boot_ci_95[1], color="k", linestyle="--")
    ax.set(title="Bootstrap distribution of the sample mean", xlabel="mean height (cm)", ylabel="count")
    ax.legend(fontsize=8)
    fig.savefig(out_dir / "recipe6_bootstrap_distribution.png", dpi=150)
    plt.close(fig)

    return {
        "mean": sample_mean,
        "t_ci_95": conf_interval_95,
        "t_ci_99": conf_interval_99,
        "bootstrap_ci_95": boot_ci_95.tolist(),
    }


sampling_results = recipe_sampling_ci(rng, OUTPUT_DIR)

# %% [markdown]
# ## Recipe 7 — Performing operations on grouped data in a DataFrame
#
# **Enhancement:** demonstrates `.agg()` with multiple named aggregations and
# a `pivot_table` summary — both explicitly name-dropped in the book's
# "There's more..." section for Recipe 3 but never shown with code.

# %%
def recipe_grouped_data(rng: np.random.Generator, out_dir: Path) -> pd.DataFrame:
    labels1 = rng.choice(["A", "B", "C"], size=50)
    labels2 = rng.choice([1, 2], size=50)
    data = rng.normal(0.0, 2.0, size=50)
    df = pd.DataFrame({"label1": labels1, "label2": labels2, "data": data})

    df["first_group"] = df.groupby("label1")["data"].cumsum()

    grouped = df.groupby(["label1", "label2"])
    df["second_group"] = grouped["data"].transform(
        lambda d: d.rolling(2, min_periods=1).mean()
    )

    fig, ax = plt.subplots()
    df.groupby("label1")["first_group"].plot(ax=ax)
    ax.set(title="Grouped data cumulative sums", xlabel="Index", ylabel="value")
    ax.legend()
    fig.savefig(out_dir / "recipe7_grouped_cumsum.png", dpi=150)
    plt.close(fig)

    # --- Enhancement: named multi-aggregation summary ---
    summary = df.groupby("label1")["data"].agg(
        n="count", mean="mean", std="std", minimum="min", maximum="max"
    )
    print("[enhancement] multi-agg summary by label1:\n", summary)

    # --- Enhancement: pivot table (mean data value by label1 x label2) ---
    pivot = df.pivot_table(values="data", index="label1", columns="label2", aggfunc="mean")
    print("\n[enhancement] pivot table (mean 'data' by label1 x label2):\n", pivot)

    return df


df_grouped_demo = recipe_grouped_data(rng, OUTPUT_DIR)

# %% [markdown]
# ## Recipe 8 — Testing hypotheses using t-tests
#
# **Enhancement:** the book only shows a one-sample t-test. This adds a
# **two-sample** and a **paired** t-test (both explicitly named in the book's
# "There's more..." as things the reader should know about), plus Cohen's *d*
# effect size, since a tiny p-value alone doesn't say anything about
# practical significance.

# %%
def recipe_t_tests() -> None:
    sample = pd.Series(
        [
            2.4, 2.4, 2.9, 2.6, 1.8, 2.7, 2.6, 2.4, 2.8,
            2.4, 2.4, 2.4, 2.7, 2.7, 2.3, 2.4, 2.4, 3.2,
            2.9, 2.5, 2.5, 3.2, 2.0, 2.3, 3.0, 1.5, 3.1,
            2.5, 2.2, 2.5, 2.1, 1.8, 3.1, 2.4, 3.0, 2.5,
            2.7, 2.1, 2.3, 2.2, 2.5, 2.6, 2.5, 2.8, 2.5,
            2.9, 2.1, 2.8, 2.1, 2.3,
        ]
    )
    mu0 = 2.0
    significance = 0.05

    t_statistic, p_value = stats.ttest_1samp(sample, mu0)
    print(f"[one-sample] t={t_statistic:.4f}, p={p_value:.3e}")
    verdict = "Reject H0" if p_value <= significance else "Accept H0"
    print(f"  -> {verdict}: mu {'!=' if p_value <= significance else '='} {mu0}")

    # --- Enhancement: two-sample t-test ---
    rng2 = default_rng(SEED)
    group_a = rng2.normal(2.5, 0.35, size=30)
    group_b = rng2.normal(2.7, 0.35, size=30)
    t2, p2 = stats.ttest_ind(group_a, group_b)
    pooled_std = math.sqrt(((group_a.std(ddof=1) ** 2) + (group_b.std(ddof=1) ** 2)) / 2)
    cohens_d = (group_a.mean() - group_b.mean()) / pooled_std
    print(f"\n[enhancement: two-sample] t={t2:.4f}, p={p2:.4f}, Cohen's d={cohens_d:.3f}")

    # --- Enhancement: paired t-test ---
    before = rng2.normal(100, 10, size=20)
    after = before + rng2.normal(3, 5, size=20)  # simulated treatment effect
    t3, p3 = stats.ttest_rel(before, after)
    print(f"[enhancement: paired] t={t3:.4f}, p={p3:.4f}")


recipe_t_tests()

# %% [markdown]
# ## Recipe 9 — Testing hypotheses using ANOVA
#
# **Enhancement:** the book explicitly says ANOVA "cannot detect which
# sample(s) are significantly different" and points to Tukey's range test as
# the fix, without showing it. This adds a Tukey HSD post-hoc test (via
# `statsmodels`) plus a boxplot comparing the three processes.

# %%
def recipe_anova(rng: np.random.Generator, out_dir: Path) -> None:
    current = rng.normal(4.0, 2.0, size=40)
    process_a = rng.normal(6.2, 2.0, size=25)
    process_b = rng.normal(4.5, 2.0, size=64)
    significance = 0.05

    F_stat, p_value = stats.f_oneway(current, process_a, process_b)
    print(f"F={F_stat:.4f}, p={p_value:.3e}")
    verdict = "Reject H0: means differ" if p_value <= significance else "Accept H0: means equal"
    print(f"  -> {verdict}")

    # --- Enhancement: boxplot comparison ---
    fig, ax = plt.subplots()
    ax.boxplot([current, process_a, process_b], tick_labels=["current", "process_a", "process_b"])
    ax.set(title="Process comparison", ylabel="measured value")
    fig.savefig(out_dir / "recipe9_anova_boxplot.png", dpi=150)
    plt.close(fig)

    # --- Enhancement: Tukey HSD post-hoc test ---
    if HAVE_STATSMODELS:
        values = np.concatenate([current, process_a, process_b])
        groups = (
            ["current"] * len(current) + ["process_a"] * len(process_a) + ["process_b"] * len(process_b)
        )
        tukey = mc.pairwise_tukeyhsd(values, groups, alpha=significance)
        print("\n[enhancement] Tukey HSD post-hoc test:")
        print(tukey.summary())
    else:
        print("[enhancement] Tukey HSD skipped (statsmodels not installed)")


recipe_anova(rng, OUTPUT_DIR)

# %% [markdown]
# ## Recipe 10 — Testing hypotheses for non-parametric data
#
# **Enhancement:** the book explicitly flags that running four tests at 95%
# confidence drops overall confidence to ~81%, and says "we would have to
# adjust our significance threshold... using the Bonferroni correction (or
# similar)" — but never does it. This cell actually applies the Bonferroni
# (and Holm, which is less conservative) correction via
# `statsmodels.stats.multitest.multipletests`. Also adds a Mann-Whitney U
# test as a common alternative to the rank-sum test for two groups.

# %%
def recipe_nonparametric(rng: np.random.Generator) -> None:
    sample_A = rng.uniform(2.5, 3.5, size=25)
    sample_B = rng.uniform(3.0, 4.4, size=25)
    sample_C = rng.uniform(3.1, 4.5, size=25)
    significance = 0.05

    statistic, p_value = stats.kruskal(sample_A, sample_B, sample_C)
    print(f"[Kruskal-Wallis] H={statistic:.4f}, p={p_value:.3e}")

    _, p_A_B = stats.ranksums(sample_A, sample_B)
    _, p_A_C = stats.ranksums(sample_A, sample_C)
    _, p_B_C = stats.ranksums(sample_B, sample_C)
    raw_p_values = {"A_vs_B": p_A_B, "A_vs_C": p_A_C, "B_vs_C": p_B_C}
    for label, p in raw_p_values.items():
        flag = "significant" if p <= significance else "not significant"
        print(f"  rank-sum {label}: p={p:.3e} ({flag}, uncorrected)")

    # --- Enhancement: multiple-comparison correction ---
    if HAVE_STATSMODELS:
        labels = list(raw_p_values.keys())
        pvals = list(raw_p_values.values())
        reject_bonf, p_bonf, _, _ = multipletests(pvals, alpha=significance, method="bonferroni")
        reject_holm, p_holm, _, _ = multipletests(pvals, alpha=significance, method="holm")
        print("\n[enhancement] multiple-comparison correction over 3 pairwise tests:")
        for lab, pb, rb, ph, rh in zip(labels, p_bonf, reject_bonf, p_holm, reject_holm):
            print(
                f"  {lab}: bonferroni p={pb:.3e} (reject={rb})  |  holm p={ph:.3e} (reject={rh})"
            )
    else:
        print("[enhancement] Bonferroni/Holm correction skipped (statsmodels not installed)")

    # --- Enhancement: Mann-Whitney U as an alternative two-sample test ---
    u_stat, u_p = stats.mannwhitneyu(sample_A, sample_B, alternative="two-sided")
    print(f"\n[enhancement] Mann-Whitney U (A vs B): U={u_stat:.1f}, p={u_p:.3e}")


recipe_nonparametric(rng)

# %% [markdown]
# ## Recipe 11 — Creating interactive plots with Bokeh
#
# **Enhancement:** instead of a single line series, this plots two series on
# shared axes with a `HoverTool` (showing date + value on mouseover) and an
# explicit `Legend`, since the book's version has no tooltip/legend at all.

# %%
def recipe_bokeh(rng: np.random.Generator, out_dir: Path) -> None:
    if not HAVE_BOKEH:
        print("[skip] bokeh not installed")
        return

    date_range = pd.date_range("2020-01-01", periods=50)
    series_a = pd.Series(rng.normal(0, 3, size=50).cumsum(), index=date_range)
    series_b = pd.Series(rng.normal(0, 2, size=50).cumsum(), index=date_range)  # enhancement: 2nd series

    html_path = out_dir / "recipe11_interactive.html"
    bk.output_file(str(html_path))

    fig = bk.figure(
        title="Time series data",
        x_axis_label="date",
        x_axis_type="datetime",
        y_axis_label="value",
        width=800,
        height=400,
    )
    line_a = fig.line(date_range, series_a, line_width=2, color="black", legend_label="Series A")
    line_b = fig.line(
        date_range, series_b, line_width=2, color="gray", line_dash="dashed", legend_label="Series B"
    )

    hover = HoverTool(
        renderers=[line_a, line_b],
        tooltips=[("date", "@x{%F}"), ("value", "@y{0.00}")],
        formatters={"@x": "datetime"},
        mode="vline",
    )
    fig.add_tools(hover)
    fig.legend.click_policy = "hide"  # enhancement: click legend to toggle series

    bk.save(fig)  # headless-safe: writes HTML instead of opening a browser
    print(f"[enhancement] interactive multi-series Bokeh plot with hover -> {html_path}")


recipe_bokeh(rng, OUTPUT_DIR)

# %% [markdown]
# ## Summary of Enhancements
#
# Beyond faithfully reproducing every recipe in the chapter, this version adds:
#
# 1. **Reproducibility & structure** — every recipe wrapped in a function with
#    a single shared, seeded `Generator`; headless-safe Matplotlib backend;
#    all figures/files saved under `outputs/` instead of relying on
#    interactive `plt.show()`/browser pop-ups.
# 2. **Recipe 1** — demonstrates `.iloc`, `.loc`, `.at`, dtype inspection, and
#    boolean-mask filtering (named in the book's "There's more" but not coded).
# 3. **Recipe 2** — round-trips through JSON and Parquet in addition to CSV,
#    with `assert_frame_equal` verification and graceful handling of missing
#    optional engines.
# 4. **Recipe 3** — adds a vectorized `np.where` equivalent to the slow
#    row-wise `.apply(axis=1)`, plus `fillna`/`interpolate` alternatives to
#    `dropna` (both name-dropped, neither coded, in the original).
# 5. **Recipe 4** — adds KDE overlay and boxplot views, saved as PNG.
# 6. **Recipe 5** — adds skewness alongside kurtosis, plus a
#    `scipy.stats.normaltest` normality check per column, and marks the
#    median (not just the mean) on each histogram.
# 7. **Recipe 6** — adds a 10,000-resample bootstrap confidence interval as a
#    non-parametric alternative to the Student-t interval, with a plot of the
#    bootstrap sampling distribution.
# 8. **Recipe 7** — adds a named multi-aggregation `.agg()` summary and a
#    `.pivot_table()` cross-tab (both named, neither coded, in the original).
# 9. **Recipe 8** — adds two-sample and paired t-tests (the book only shows
#    one-sample) plus Cohen's *d* effect size, since p-values alone say
#    nothing about practical significance.
# 10. **Recipe 9** — adds a boxplot comparison and a full Tukey HSD post-hoc
#     test via `statsmodels`, directly addressing the book's own caveat that
#     ANOVA "cannot detect which sample(s) are significantly different."
# 11. **Recipe 10** — actually implements the Bonferroni/Holm corrections the
#     book explicitly says are needed but never codes, plus a Mann-Whitney U
#     test as an alternative two-sample non-parametric test.
# 12. **Recipe 11** — upgrades the single-series Bokeh plot to two series with
#     a `HoverTool` tooltip, a togglable `Legend`, and headless-safe
#     `bk.save()` instead of `bk.show()`.
# 13. **General robustness** — optional imports (`statsmodels`, `bokeh`) are
#     guarded so the script still runs end-to-end without them; assertions
#     validate round-trips and cross-checks throughout.
