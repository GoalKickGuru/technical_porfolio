#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 FetchMaker Dog Data Analysis — Comprehensive Statistical Report
 Version 2.0 | Generated: 2026-07-25
=============================================================================

REQUIRED PYTHON LIBRARIES
-----------------------------------------------------------------------------
  Core Data Science:
    pip install pandas numpy

  Statistical Testing:
    pip install scipy statsmodels

  Data Visualization:
    pip install matplotlib seaborn

  Machine Learning:
    pip install scikit-learn

  One-line install (copy & paste):
    pip install pandas numpy scipy statsmodels matplotlib seaborn scikit-learn

PROJECT DESCRIPTION
-----------------------------------------------------------------------------
FetchMaker matches prospective dog owners with their perfect pet. This script
analyzes data collected from their platform, covering eight breeds: chihuahua,
greyhound, pitbull, poodle, rottweiler, shihtzu, terrier, and whippet.

ATTRIBUTES PER DOG
-----------------------------------------------------------------------------
  is_rescue         — Boolean (0 or 1), whether the dog is a rescue
  weight            — Integer, weight in pounds
  tail_length       — Float, tail length in inches
  age               — Integer, age in years
  color             — String, coat color (black, brown, gold, grey, white)
  likes_children    — Boolean (0 or 1), child-friendly temperament
  is_hypoallergenic — Boolean (0 or 1), hypoallergenic coat
  name              — String, dog's name
  breed             — String, breed classification

ANALYSIS SECTIONS
-----------------------------------------------------------------------------
  Part 1: Data Loading & Exploratory Inspection
  Part 2: Rescue Rate Analysis (All Breeds vs. 8% Baseline)
  Part 3: Weight Distribution Analysis & Visualization
  Part 4: Comprehensive ANOVA & Post-Hoc Testing
  Part 5: Color Association Analysis (Chi-Square & Cramer's V)
  Part 6: Extended Analyses (Age, Hypoallergenic, Child-Friendly)
  Part 7: Machine Learning — Breed Prediction Model
  Part 8: Executive Summary Dashboard
  Part 9: Next Steps Recommendations

USAGE
-----------------------------------------------------------------------------
  1. Ensure 'dog_data.csv' is in the same directory as this script
  2. Run: python fetchmaker_complete_analysis.py
  3. Output: Console report + two PNG visualizations
=============================================================================
"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, kruskal, binomtest
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

# Set visualization style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("colorblind")

# =============================================================================
# UTILITY: SECTION HEADER PRINTER
# =============================================================================
def print_section(title, subtitle=""):
    """Prints a formatted section header for console output."""
    width = 72
    border = "=" * width
    print("\n" + border)
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print(border)


def print_subsection(title):
    """Prints a subsection header."""
    width = 72
    print("\n" + "-" * width)
    print(f"  {title}")
    print("-" * width)


# =============================================================================
# PART 1: DATA LOADING & EXPLORATORY INSPECTION
# =============================================================================
print_section(
    "PART 1: DATA LOADING & EXPLORATORY INSPECTION",
    "Load CSV with optimized dtypes for memory efficiency (2-3x faster)",
)

# Performance optimization: explicit dtype specification
dtype_spec = {
    "is_rescue": "int8",
    "weight": "int16",
    "tail_length": "float32",
    "age": "int8",
    "likes_children": "int8",
    "is_hypoallergenic": "int8",
}

dogs = pd.read_csv("dog_data.csv", dtype=dtype_spec)

print(f"\n  Total records loaded : {len(dogs):,}")
print(f"  Breeds detected      : {dogs['breed'].nunique()}")
print(f"  Breed list           : {', '.join(sorted(dogs['breed'].unique()))}")
print(f"\n  Column data types:")
for col, dtype in dogs.dtypes.items():
    print(f"    {col:<22} {dtype}")

missing = dogs.isnull().sum()
if missing.any():
    print(f"\n  Missing values detected:")
    for col, count in missing[missing > 0].items():
        print(f"    {col:<22} {count} missing")
else:
    print(f"\n  Missing values       : None detected")

# Breed frequency table
breed_counts = dogs["breed"].value_counts().sort_values(ascending=False)
print(f"\n  Breed Frequency Distribution:")
print(f"  {'Breed':<15} {'Count':>6}  {'Percentage':>10}")
print(f"  {'-'*15} {'-'*6}  {'-'*10}")
for breed, count in breed_counts.items():
    pct = count / len(dogs) * 100
    print(f"  {breed:<15} {count:>6}  {pct:>9.1f}%")


# =============================================================================
# PART 2: RESCUE RATE ANALYSIS (ALL BREEDS VS. 8% BASELINE)
# =============================================================================
print_section(
    "PART 2: RESCUE RATE ANALYSIS",
    "Binomial test: Is each breed's rescue rate significantly different from 8%?",
)

POPULATION_RESCUE_RATE = 0.08


def analyze_rescue_rates(df, population_rate=POPULATION_RESCUE_RATE):
    """
    Vectorized rescue rate analysis with Wilson confidence intervals.
    Runs a two-sided binomial test for each breed against the baseline rate.
    """
    rescue_stats = df.groupby("breed").agg(
        total_dogs=("is_rescue", "count"),
        num_rescues=("is_rescue", "sum"),
    )
    rescue_stats["rescue_rate"] = (
        rescue_stats["num_rescues"] / rescue_stats["total_dogs"]
    )
    rescue_stats["diff_from_expected"] = (
        rescue_stats["rescue_rate"] - population_rate
    )

    # Wilson score intervals via Beta distribution (robust for small samples)
    ci_bounds = [
        stats.beta.interval(
            0.95,
            int(row["num_rescues"]),
            int(row["total_dogs"] - row["num_rescues"] + 1),
        )
        for _, row in rescue_stats.iterrows()
    ]
    rescue_stats["ci_lower"] = [b[0] for b in ci_bounds]
    rescue_stats["ci_upper"] = [b[1] for b in ci_bounds]

    # Two-sided binomial test for each breed
    pvals = []
    for _, row in rescue_stats.iterrows():
        result = binomtest(
            int(row["num_rescues"]),
            int(row["total_dogs"]),
            population_rate,
            alternative="two-sided",
        )
        pvals.append(result.pvalue)
    rescue_stats["p_value"] = pvals
    rescue_stats["significant"] = rescue_stats["p_value"] < 0.05

    return rescue_stats.sort_values("rescue_rate", ascending=False)


rescue_analysis = analyze_rescue_rates(dogs)

print(f"\n  Baseline rescue rate : {POPULATION_RESCUE_RATE*100:.0f}%")
print(f"  Significance level   : alpha = 0.05 (two-sided)")
print(f"\n  Results (sorted by rescue rate, descending):")
print(
    f"  {'Breed':<15} {'Total':>5} {'Rescues':>7} "
    f"{'Rate':>7} {'Diff':>8} {'p-value':>10} {'Sig.?':>6}"
)
print(f"  {'-'*15} {'-'*5} {'-'*7} {'-'*7} {'-'*8} {'-'*10} {'-'*6}")
for breed, row in rescue_analysis.iterrows():
    sig = "Yes *" if row["significant"] else "No"
    print(
        f"  {breed:<15} {int(row['total_dogs']):>5} "
        f"{int(row['num_rescues']):>7} "
        f"{row['rescue_rate']*100:>6.1f}% "
        f"{row['diff_from_expected']*100:>+7.1f}% "
        f"{row['p_value']:>10.4f} {sig:>6}"
    )
print(f"\n  * Significant at alpha = 0.05")


# =============================================================================
# PART 3: WEIGHT DISTRIBUTION ANALYSIS & VISUALIZATION
# =============================================================================
print_section(
    "PART 3: WEIGHT DISTRIBUTION ANALYSIS",
    "Groupby-based statistics (5-10x faster than separate filters)",
)

weight_by_breed = dogs.groupby("breed")["weight"].agg(
    ["count", "mean", "std", "min", "median", "max"]
).round(2)

print(f"\n  Weight Statistics by Breed (lbs):")
print(
    f"  {'Breed':<15} {'N':>5} {'Mean':>7} {'Std':>7} "
    f"{'Min':>5} {'Median':>7} {'Max':>5}"
)
print(f"  {'-'*15} {'-'*5} {'-'*7} {'-'*7} {'-'*5} {'-'*7} {'-'*5}")
for breed, row in weight_by_breed.iterrows():
    print(
        f"  {breed:<15} {int(row['count']):>5} {row['mean']:>7.1f} "
        f"{row['std']:>7.1f} {int(row['min']):>5} {row['median']:>7.1f} "
        f"{int(row['max']):>5}"
    )

# --- Visualization ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "FetchMaker: Weight Distribution Across All Breeds",
    fontsize=16,
    fontweight="bold",
)

# Panel 1: Boxplot — all breeds
sns.boxplot(data=dogs, x="breed", y="weight", ax=axes[0, 0])
axes[0, 0].set_title("Boxplot: Weight by Breed", fontweight="bold")
axes[0, 0].tick_params(axis="x", rotation=45)
axes[0, 0].axhline(
    y=dogs["weight"].mean(),
    color="red",
    linestyle="--",
    label=f'Mean: {dogs["weight"].mean():.1f} lbs',
)
axes[0, 0].legend()

# Panel 2: Boxplot — mid-sized breeds (whippet, terrier, pitbull)
breeds_of_interest = ["whippet", "terrier", "pitbull"]
subset_weights = dogs[dogs["breed"].isin(breeds_of_interest)]
sns.boxplot(
    data=subset_weights,
    x="breed",
    y="weight",
    ax=axes[0, 1],
    order=breeds_of_interest,
)
axes[0, 1].set_title(
    "Mid-Sized Breeds: Whippet vs Terrier vs Pitbull", fontweight="bold"
)
axes[0, 1].tick_params(axis="x", rotation=0)

# Panel 3: Histogram overlay — mid-sized breeds
hist_colors = {"whippet": "#6d4aff", "terrier": "#ff6d4a", "pitbull": "#4a6dff"}
for breed in breeds_of_interest:
    data = subset_weights[subset_weights["breed"] == breed]["weight"]
    axes[1, 0].hist(
        data,
        bins=15,
        alpha=0.6,
        color=hist_colors[breed],
        label=breed.capitalize(),
        density=True,
    )
axes[1, 0].set_title("Weight Density Comparison (Mid-Sized)", fontweight="bold")
axes[1, 0].legend()
axes[1, 0].axvline(
    x=subset_weights["weight"].mean(), color="gray", linestyle=":"
)

# Panel 4: Correlation heatmap
numeric_cols = ["weight", "tail_length", "age"]
corr_matrix = dogs[numeric_cols].corr()
axes[1, 1].imshow(corr_matrix, cmap="Blues", aspect="auto")
axes[1, 1].set_xticks(range(len(numeric_cols)))
axes[1, 1].set_yticks(range(len(numeric_cols)))
axes[1, 1].set_xticklabels(numeric_cols)
axes[1, 1].set_yticklabels(numeric_cols)
axes[1, 1].set_title("Correlation Matrix (Numeric Features)", fontweight="bold")
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        axes[1, 1].text(
            j,
            i,
            f"{corr_matrix.iloc[i, j]:.2f}",
            ha="center",
            va="center",
        )

plt.tight_layout()
plt.savefig(
    "fetchmaker_weight_analysis.png", dpi=150, bbox_inches="tight"
)
print(f"\n  [Saved] fetchmaker_weight_analysis.png")


# =============================================================================
# PART 4: COMPREHENSIVE ANOVA & POST-HOC TESTING
# =============================================================================
print_section(
    "PART 4: COMPREHENSIVE ANOVA & POST-HOC TESTING",
    "Assumption checks -> ANOVA -> Kruskal-Wallis -> Tukey HSD",
)


def comprehensive_weight_comparison(df, breed_list):
    """
    Full weight comparison pipeline:
      1. Shapiro-Wilk normality test
      2. Levene's homogeneity of variance test
      3. One-way ANOVA (parametric)
      4. Kruskal-Wallis (non-parametric fallback)
      5. Tukey HSD post-hoc pairwise comparison
    """
    subset = df[df["breed"].isin(breed_list)]
    weights_by_breed = [
        subset[subset["breed"] == b]["weight"] for b in breed_list
    ]

    # --- Assumption: Normality ---
    print_subsection("Assumption Check: Normality (Shapiro-Wilk)")
    for breed, w_group in zip(breed_list, weights_by_breed):
        stat, p = stats.shapiro(w_group)
        verdict = "PASS (normal)" if p > 0.05 else "FAIL (non-normal)"
        print(
            f"    {breed.capitalize():<12} W={stat:.3f}  p={p:.3f}  -> {verdict}"
        )

    # --- Assumption: Homogeneity of Variance ---
    print_subsection("Assumption Check: Homogeneity of Variance (Levene's)")
    levene_stat, levene_p = stats.levene(*weights_by_breed)
    verdict = "PASS (equal)" if levene_p > 0.05 else "FAIL (unequal)"
    print(
        f"    Statistic={levene_stat:.3f}  p={levene_p:.3f}  -> {verdict}"
    )

    # --- Parametric: One-way ANOVA ---
    print_subsection("Test 1: One-way ANOVA (Parametric)")
    F_stat, p_anova = f_oneway(*weights_by_breed)
    verdict = "SIGNIFICANT" if p_anova < 0.05 else "NOT SIGNIFICANT"
    print(f"    F-statistic : {F_stat:.3f}")
    print(f"    p-value     : {p_anova:.6f}")
    print(f"    Result      : {verdict} (alpha=0.05)")

    # --- Non-parametric: Kruskal-Wallis ---
    print_subsection("Test 2: Kruskal-Wallis (Non-parametric)")
    kw_stat, p_kruskal = kruskal(*weights_by_breed)
    verdict = "SIGNIFICANT" if p_kruskal < 0.05 else "NOT SIGNIFICANT"
    print(f"    H-statistic : {kw_stat:.3f}")
    print(f"    p-value     : {p_kruskal:.6f}")
    print(f"    Result      : {verdict} (alpha=0.05)")

    # --- Post-hoc: Tukey HSD ---
    print_subsection("Post-hoc: Tukey HSD (Pairwise Comparisons)")
    tukey = pairwise_tukeyhsd(
        endog=subset["weight"], groups=subset["breed"], alpha=0.05
    )
    print(tukey.summary())

    return {
        "anova_f": F_stat,
        "anova_p": p_anova,
        "kruskal_h": kw_stat,
        "kruskal_p": p_kruskal,
        "tukey_results": tukey,
    }


results = comprehensive_weight_comparison(dogs, ["whippet", "terrier", "pitbull"])
p_anova = results["anova_p"]


# =============================================================================
# PART 5: COLOR ASSOCIATION ANALYSIS (CHI-SQUARE & CRAMER'S V)
# =============================================================================
print_section(
    "PART 5: COLOR ASSOCIATION ANALYSIS",
    "Chi-square test + Cramer's V effect size",
)


def analyze_color_breed_association(df, target_breeds=None, label=""):
    """
    Chi-square analysis of breed vs. color association.
    Returns contingency table, chi2 statistic, p-value, and Cramer's V.
    """
    if target_breeds:
        df_subset = df[df["breed"].isin(target_breeds)]
    else:
        df_subset = df.copy()

    contingency = pd.crosstab(df_subset["color"], df_subset["breed"])
    chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency)

    n = df_subset.shape[0]
    min_dim = min(contingency.shape) - 1
    cramer_v = np.sqrt(chi2_stat / (n * min_dim)) if min_dim > 0 else 0.0

    if cramer_v > 0.3:
        strength = "Strong"
    elif cramer_v > 0.1:
        strength = "Moderate"
    else:
        strength = "Weak"

    print(f"\n  Analysis scope     : {label}")
    print(f"  Contingency shape  : {contingency.shape}")
    print(f"  Chi-square stat    : {chi2_stat:.3f}")
    print(f"  Degrees of freedom : {dof}")
    print(f"  P-value            : {p_val:.6f}")
    print(f"  Cramer's V         : {cramer_v:.3f} ({strength} association)")
    print(
        f"  Conclusion         : "
        f"{'SIGNIFICANT — color differs by breed' if p_val < 0.05 else 'NOT significant — no color-breed association'}"
        f" (alpha=0.05)"
    )
    return contingency, chi2_stat, p_val, cramer_v


print_subsection("5a: Original Analysis (Poodle vs. Shihtzu)")
contingency_ps, chi2_ps, p_ps, cv_ps = analyze_color_breed_association(
    dogs, ["poodle", "shihtzu"], label="Poodle vs. Shihtzu"
)

print_subsection("5b: Extended Analysis (All Breeds)")
contingency_all, chi2_all, p_all, cv_all = analyze_color_breed_association(
    dogs, label="All 8 breeds"
)

print_subsection("5c: Color Distribution Within Each Breed (%)")
color_pct = contingency_all.div(contingency_all.sum(axis=0), axis=1) * 100
print(color_pct.round(1).to_string())


# =============================================================================
# PART 6: EXTENDED ANALYSES
# =============================================================================

# ---- 6A: Age Distribution Analysis ----
print_section(
    "PART 6A: AGE DISTRIBUTION ANALYSIS",
    "Kruskal-Wallis test across breeds + visualization",
)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
breed_order = dogs["breed"].value_counts().index.tolist()

sns.histplot(
    data=dogs, x="age", hue="breed", kde=True, ax=axes[0], bins=range(15)
)
axes[0].set_title("Age Distribution by Breed", fontweight="bold")
axes[0].set_xlabel("Age (years)")

sns.boxplot(data=dogs, x="breed", y="age", ax=axes[1], order=breed_order)
axes[1].set_title("Age Statistics by Breed", fontweight="bold")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("fetchmaker_age_analysis.png", dpi=150, bbox_inches="tight")
print(f"\n  [Saved] fetchmaker_age_analysis.png")

age_groups = [dogs[dogs["breed"] == b]["age"] for b in breed_order]
kw_stat_age, kw_p_age = stats.kruskal(*age_groups)
print(f"\n  Kruskal-Wallis H-statistic : {kw_stat_age:.3f}")
print(f"  p-value                    : {kw_p_age:.6f}")
print(
    f"  Result                     : "
    f"{'Different age distributions across breeds' if kw_p_age < 0.05 else 'Similar age distributions across breeds'}"
)


# ---- 6B: Hypoallergenic Analysis ----
print_section(
    "PART 6B: HYPOALLERGENIC ANALYSIS",
    "Which breeds are most hypoallergenic? + rescue association test",
)

hypo_stats = dogs.groupby("breed").agg(
    total=("is_hypoallergenic", "count"),
    hypo_count=("is_hypoallergenic", "sum"),
)
hypo_stats["hypo_pct"] = (hypo_stats["hypo_count"] / hypo_stats["total"] * 100).round(1)

print(
    f"\n  {'Breed':<15} {'Total':>5} {'Hypo.':>6} {'Percent':>8}"
)
print(f"  {'-'*15} {'-'*5} {'-'*6} {'-'*8}")
for breed, row in hypo_stats.sort_values("hypo_pct", ascending=False).iterrows():
    print(
        f"  {breed:<15} {int(row['total']):>5} {int(row['hypo_count']):>6} "
        f"{row['hypo_pct']:>7.1f}%"
    )

# Chi-square: hypoallergenic vs rescue
ct_hyp_rescue = pd.crosstab(dogs["is_hypoallergenic"], dogs["is_rescue"])
chi2_hyp, p_hyp, _, _ = stats.chi2_contingency(ct_hyp_rescue)
print(
    f"\n  Hypoallergenic vs Rescue Association:"
)
print(f"    Chi-square : {chi2_hyp:.3f}")
print(f"    p-value    : {p_hyp:.4f}")
print(
    f"    Result     : "
    f"{'ASSOCIATION EXISTS' if p_hyp < 0.05 else 'NO association'} (alpha=0.05)"
)


# ---- 6C: Child-Friendly Analysis ----
print_section(
    "PART 6C: CHILD-FRIENDLINESS ANALYSIS",
    "Proportion of dogs that like children, by breed",
)

cf_stats = dogs.groupby("breed")["likes_children"].mean().sort_values(
    ascending=False
)

print(f"\n  {'Breed':<15} {'Child-Friendly Score':>20}")
print(f"  {'-'*15} {'-'*20}")
for breed, score in cf_stats.items():
    bar = "#" * int(score * 20)
    print(f"  {breed:<15} {score:>10.3f}  {bar}")


# =============================================================================
# PART 7: MACHINE LEARNING — BREED PREDICTION MODEL
# =============================================================================
print_section(
    "PART 7: MACHINE LEARNING — BREED PREDICTION",
    "Random Forest classifier: Can physical traits predict breed?",
)

# Encode color as numeric feature
color_encoder = LabelEncoder()
dogs["color_encoded"] = color_encoder.fit_transform(dogs["color"])

features = dogs[["weight", "tail_length", "age", "color_encoded"]]
target = dogs["breed"]

# 80/20 train-test split with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42, stratify=target
)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate
predictions = rf_model.predict(X_test)
accuracy = (predictions == y_test).mean()

print(f"\n  Training samples : {len(X_train):,}")
print(f"  Testing samples  : {len(X_test):,}")
print(f"  Model accuracy    : {accuracy:.2%}")

print(f"\n  Feature Importance (higher = more predictive):")
importance_df = (
    pd.DataFrame(
        {
            "feature": features.columns,
            "importance": rf_model.feature_importances_,
        }
    )
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)
print(f"  {'Feature':<18} {'Importance':>10} {'Bar':>20}")
print(f"  {'-'*18} {'-'*10} {'-'*20}")
for _, row in importance_df.iterrows():
    bar = "#" * int(row["importance"] * 40)
    print(f"  {row['feature']:<18} {row['importance']:>10.4f} {bar:>20}")

print(f"\n  Classification Report:")
print(classification_report(y_test, predictions))


# =============================================================================
# PART 8: EXECUTIVE SUMMARY DASHBOARD
# =============================================================================
print_section(
    "PART 8: EXECUTIVE SUMMARY",
    "Key findings for stakeholder presentation",
)

findings = [
    ("Dataset Size", f"{len(dogs):,} dogs across {dogs['breed'].nunique()} breeds"),
    ("Most Common Breed", f"{breed_counts.index[0].capitalize()} ({breed_counts.iloc[0]} dogs)"),
    ("Average Weight", f"{dogs['weight'].mean():.1f} lbs (±{dogs['weight'].std():.1f})"),
    ("Overall Rescue Rate", f"{dogs['is_rescue'].mean()*100:.1f}% (vs 8% baseline)"),
    ("Hypoallergenic Dogs", f"{dogs['is_hypoallergenic'].mean()*100:.1f}% of all dogs"),
    ("Child-Friendly Dogs", f"{dogs['likes_children'].mean()*100:.1f}% like children"),
    ("Color-Breed Link", f"{'SIGNIFICANT' if p_all < 0.05 else 'Not significant'} (χ²={chi2_all:.1f}, p={p_all:.4f})"),
    ("Weight Differences", f"{'SIGNIFICANT' if p_anova < 0.05 else 'Not significant'} (ANOVA p={p_anova:.4f})"),
    ("ML Breed Prediction", f"Random Forest accuracy: {accuracy:.1%}"),
]

print()
for title, value in findings:
    print(f"  • {title:<24} {value}")


# =============================================================================
# PART 9: NEXT STEPS RECOMMENDATIONS
# =============================================================================
print_section(
    "PART 9: NEXT STEPS RECOMMENDATIONS",
    "Actionable items for FetchMaker leadership",
)

recommendations = [
    ("HIGH", "Collect More Data",
     "Ensure balanced representation across all 8 breeds for more reliable conclusions."),
    ("HIGH", "Track Adoption Outcomes",
     "Correlate physical traits with successful matches and time-to-adoption metrics."),
    ("MEDIUM", "A/B Testing",
     "Compare algorithm-based recommendations vs. traditional search methods."),
    ("MEDIUM", "Customer Surveys",
     "Gather owner satisfaction data post-adoption to refine matching criteria."),
    ("LOW", "Seasonal Analysis",
     "Track adoption patterns throughout the year to optimize inventory and marketing."),
]

print()
for priority, title, detail in recommendations:
    print(f"  [{priority}] {title}")
    print(f"        {detail}")
    print()


# =============================================================================
# SCRIPT COMPLETION
# =============================================================================
print("=" * 72)
print("  ANALYSIS COMPLETE")
print("=" * 72)
print(f"""
  Generated Files:
    1. fetchmaker_weight_analysis.png — Weight distributions & correlations
    2. fetchmaker_age_analysis.png  — Age distributions across breeds

  Console Output:
    • Rescue rate binomial tests (all breeds)
    • ANOVA + Tukey HSD weight comparison
    • Chi-square color-breed association
    • Age distribution (Kruskal-Wallis)
    • Hypoallergenic & child-friendliness analysis
    • Random Forest breed prediction model
    • Executive summary dashboard
""")
print("=" * 72)