"""
===============================================================================
EXPLORATORY DATA ANALYSIS - SUMMARY STATISTICS TEMPLATE WITH VISUALIZATION
===============================================================================

Author: [Your Name]
Date: Generated Template
Description: General-purpose template for computing summary statistics on
             any tabular dataset with automatic visualization generation.

This template implements all 9 core statistical measures from the EDA Cookbook
PLUS 8 visualization types for comprehensive data exploration:
  1. Mean (average)
  2. Median (middle value)
  3. Mode (most frequent value)
  4. Variance (spread/variability)
  5. Standard Deviation (square root of variance)
  6. Range (max - min)
  7. Percentiles (divides data into 100 portions)
  8. Quartiles (divides data into 4 portions)
  9. Interquartile Range (IQR - middle 50% spread)

VISUALIZATIONS GENERATED:
  1. Histogram with density overlay
  2. Box plot (outlier detection)
  3. Q-Q plot (normality assessment)
  4. Violin plot (distribution shape)
  5. Empirical CDF (cumulative distribution)
  6. Combined distribution panel (multi-chart view)

Usage:
    python eda_template_full.py --input data.csv --output report.txt

Dependencies:
    - pandas >= 1.0.0
    - numpy >= 1.18.0
    - scipy >= 1.4.0
    - matplotlib >= 3.3.0
    - seaborn >= 0.11.0 (recommended for enhanced aesthetics)
===============================================================================
"""

# -----------------------------------------------------------------------------
# IMPORT STATEMENTS
# -----------------------------------------------------------------------------
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from datetime import datetime
import sys
import os
from pathlib import Path

# Configure matplotlib for better rendering
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.format': 'png',
    'savefig.bbox': 'tight',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
})

# Seaborn style for publication-quality plots
sns.set_style("whitegrid")
sns.set_palette("husl")

# Report configuration
REPORT_TITLE = "Exploratory Data Analysis - Summary Statistics & Visualization Report"
SECTION_WIDTH = 80
DECIMAL_PRECISION = 4
CHART_RESOLUTION = 150  # DPI for saved charts
MAX_COLUMNS = 5  # Limit columns for visualization performance

def load_and_validate_data(filepath: str, target_column: str = None) -> tuple:
    """Load CSV data and validate structure."""
    try:
        df = pd.read_csv(filepath)
        print(f"[INFO] Successfully loaded {filepath}")
        print(f"[INFO] Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) == 0:
            raise ValueError("No numeric columns found in the dataset!")
        
        if target_column:
            if target_column in numeric_columns:
                target_columns = [target_column]
                print(f"[INFO] Analyzing specific column: {target_column}")
            else:
                raise ValueError(f"Column '{target_column}' not found or not numeric!")
        else:
            target_columns = numeric_columns[:MAX_COLUMNS]  # Limit for viz performance
            if len(numeric_columns) > MAX_COLUMNS:
                print(f"[WARNING] Only analyzing first {MAX_COLUMNS} of {len(numeric_columns)} numeric columns")
            print(f"[INFO] Analyzing {len(target_columns)} numeric columns: {target_columns}")
        
        return df, target_columns
    except FileNotFoundError:
        print(f"[ERROR] File not found: {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        sys.exit(1)

def compute_summary_statistics(df: pd.DataFrame, column: str) -> dict:
    """Compute all 9 summary statistics for a given numeric column."""
    data = df[column].dropna().values
    
    if len(data) == 0:
        return {"error": f"No valid numeric data in column '{column}'"}
    
    # Central Tendency
    mean_value = np.mean(data)
    median_value = np.median(data)
    mode_result = stats.mode(data)
    mode_value = mode_result.mode[0]
    mode_count = mode_result.count[0]
    
    # Dispersion
    variance_value = np.var(data)
    std_dev_value = np.std(data)
    max_value = np.max(data)
    min_value = np.min(data)
    range_value = max_value - min_value
    
    # Quantiles
    q1 = np.quantile(data, 0.25)
    q2 = np.quantile(data, 0.50)
    q3 = np.quantile(data, 0.75)
    p25 = np.percentile(data, 25)
    p50 = np.percentile(data, 50)
    p75 = np.percentile(data, 75)
    p90 = np.percentile(data, 90)
    p95 = np.percentile(data, 95)
    
    # IQR and Outliers
    iqr_value = stats.iqr(data)
    lower_fence = q1 - 1.5 * iqr_value
    upper_fence = q3 + 1.5 * iqr_value
    outliers_low = np.sum(data < lower_fence)
    outliers_high = np.sum(data > upper_fence)
    total_outliers = outliers_low + outliers_high
    
    # Additional diagnostics
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    shapiro_stat, shapiro_p = stats.shapiro(data) if len(data) <= 5000 else (None, None)
    
    results = {
        'column_name': column,
        'sample_size': len(data),
        'missing_values': len(df[column]) - len(data),
        'mean': mean_value,
        'median': median_value,
        'mode_value': mode_value,
        'mode_count': mode_count,
        'variance': variance_value,
        'std_deviation': std_dev_value,
        'range': range_value,
        'min_value': min_value,
        'max_value': max_value,
        'percentile_25': p25,
        'percentile_50': p50,
        'percentile_75': p75,
        'percentile_90': p90,
        'percentile_95': p95,
        'quartile_1': q1,
        'quartile_2': q2,
        'quartile_3': q3,
        'iqr': iqr_value,
        'lower_fence': lower_fence,
        'upper_fence': upper_fence,
        'outliers_below': outliers_low,
        'outliers_above': outliers_high,
        'total_outliers': total_outliers,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'shapiro_stat': shapiro_stat,
        'shapiro_p': shapiro_p,
    }
    
    return results

def create_visualizations(df: pd.DataFrame, column: str, output_dir: str, 
                          stats_dict: dict) -> dict:
    """
    Create comprehensive visualization suite for a single column.
    
    Generates 6 chart types stored in organized subdirectory.
    
    Parameters:
        df: Input DataFrame
        column: Column name to visualize
        output_dir: Directory for saving charts
        stats_dict: Pre-computed statistics dictionary
    
    Returns:
        dict: Paths to generated chart files
    """
    
    chart_paths = {}
    data = df[column].dropna()
    
    # Create output directory
    viz_subdir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_subdir, exist_ok=True)
    
    # =========================================================================
    # CHART 1: HISTOGRAM WITH DENSITY OVERLAY
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram
    n_bins = int(np.sqrt(len(data)))  # Sturges' rule approximation
    counts, bin_edges, patches = ax.hist(data, bins=n_bins, alpha=0.6, 
                                         color='skyblue', edgecolor='black',
                                         label='Frequency', density=False)
    
    # Density curve (KDE)
    kde_x = np.linspace(data.min(), data.max(), 100)
    kde_y = stats.gaussian_kde(data)(kde_x)
    ax2 = ax.twinx()
    ax2.plot(kde_x, kde_y, 'r-', linewidth=2, label='Density (KDE)', alpha=0.8)
    ax2.fill_between(kde_x, kde_y, alpha=0.2, color='red')
    
    # Markers for key statistics
    ax.axvline(stats_dict['mean'], color='green', linestyle='--', 
               linewidth=2, label=f'Mean: {stats_dict["mean"]:.2f}')
    ax.axvline(stats_dict['median'], color='orange', linestyle='-', 
               linewidth=2, label=f'Median: {stats_dict["median"]:.2f}')
    ax.axvline(stats_dict['mode_value'], color='purple', linestyle=':', 
               linewidth=2, label=f'Mode: {stats_dict["mode_value"]:.2f}')
    
    # Outlier fences
    ax.axvline(stats_dict['lower_fence'], color='gray', linestyle=':', 
               linewidth=1, alpha=0.5, label='Lower Fence')
    ax.axvline(stats_dict['upper_fence'], color='gray', linestyle=':', 
               linewidth=1, alpha=0.5, label='Upper Fence')
    
    ax.set_title(f'{column}\nHistogram with Density Overlay', fontsize=14, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    
    hist_path = os.path.join(viz_subdir, f'{column}_histogram.png')
    plt.savefig(hist_path, dpi=CHART_RESOLUTION, bbox_inches='tight')
    chart_paths['histogram'] = hist_path
    plt.close()
    
    # =========================================================================
    # CHART 2: BOX PLOT (OUTLIER DETECTION)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Vertical box plot with jittered scatter points
    bp = ax.boxplot(data, vert=True, patch_artist=True,
                    showfliers=True, flierprops=dict(marker='o', 
                                                      markerfacecolor='red', 
                                                      markersize=6, 
                                                      alpha=0.6))
    
    # Color the box
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    # Add jittered strip plot for individual values
    y_positions = np.random.normal(0.75, 0.04, size=len(data))
    ax.scatter(y_positions, data, alpha=0.3, s=30, color='darkblue', zorder=3)
    
    # Reference lines
    ax.axhline(stats_dict['mean'], color='green', linestyle='--', 
               linewidth=2, label=f'Mean: {stats_dict["mean"]:.2f}')
    ax.axhline(stats_dict['median'], color='orange', linestyle='-', 
               linewidth=2, label=f'Median: {stats_dict["median"]:.2f}')
    
    ax.set_title(f'{column}\nBox Plot with Outlier Detection', fontsize=14, fontweight='bold')
    ax.set_ylabel('Value')
    ax.set_xticks([0.75])
    ax.set_xticklabels([column])
    ax.legend(loc='best', fontsize=9)
    
    box_path = os.path.join(viz_subdir, f'{column}_boxplot.png')
    plt.savefig(box_path, dpi=CHART_RESOLUTION, bbox_inches='tight')
    chart_paths['boxplot'] = box_path
    plt.close()
    
    # =========================================================================
    # CHART 3: Q-Q PLOT (NORMALITY ASSESSMENT)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create theoretical quantiles
    if len(data) > 0:
        qq = stats.probplot(data, dist="norm", plot=ax)
        ax.grid(True, alpha=0.3)
        
        # Add 45-degree reference line
        min_q, max_q = min(qq[0][0], qq[0][1]), max(qq[0][0], qq[0][1])
        ax.plot([min_q, max_q], [min_q, max_q], 'r--', linewidth=2, 
                label='Perfect Normality')
    
    ax.set_title(f'{column}\nQ-Q Plot (Normality Assessment)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Sample Quantiles')
    
    if stats_dict['shapiro_p']:
        ax.text(0.05, 0.95, f"Shapiro-Wilk p-value: {stats_dict['shapiro_p']:.4f}\n"
                            f"(p < 0.05 rejects normality)", 
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    qq_path = os.path.join(viz_subdir, f'{column}_qqplot.png')
    plt.savefig(qq_path, dpi=CHART_RESOLUTION, bbox_inches='tight')
    chart_paths['qqplot'] = qq_path
    plt.close()
    
    # =========================================================================
    # CHART 4: VIOLIN PLOT (DISTRIBUTION SHAPE)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create violin plot
    parts = ax.violinplot(data, vert=True, showmeans=True, showmedians=True, 
                          showextrema=True)
    
    # Style the violin bodies
    for pc in parts['bodies']:
        pc.set_facecolor('#4169E1')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    
    # Style the statistical indicators
    parts['cmeans'].set_facecolor('green')
    parts['cmeans'].set_edgecolor('green')
    parts['cmedians'].set_facecolor('orange')
    parts['cmedians'].set_edgecolor('orange')
    parts['cmaxes'].set_edgecolor('gray')
    parts['cmins'].set_edgecolor('gray')
    parts['cbars'].set_edgecolor('gray')
    
    ax.set_title(f'{column}\nViolin Plot (Distribution Shape)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Value')
    ax.set_xticks([1])
    ax.set_xticklabels([column])
    
    violin_path = os.path.join(viz_subdir, f'{column}_violinplot.png')
    plt.savefig(violin_path, dpi=CHART_RESOLUTION, bbox_inches='tight')
    chart_paths['violinplot'] = violin_path
    plt.close()
    
    # =========================================================================
    # CHART 5: EMPIRICAL CDF (CUMULATIVE DISTRIBUTION FUNCTION)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort data for CDF
    sorted_data = np.sort(data)
    cdf_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
    # Plot CDF
    ax.step(sorted_data, cdf_values, where='post', linewidth=2, 
            color='darkblue', label='Empirical CDF')
    
    # Mark percentiles
    percentiles_to_mark = [25, 50, 75, 90, 95]
    for p in percentiles_to_mark:
        val = np.percentile(sorted_data, p)
        idx = np.searchsorted(sorted_data, val)
        idx = min(idx, len(sorted_data) - 1)
        ax.scatter(val, cdf_values[idx], color='red', s=80, zorder=5)
        ax.annotate(f'P{p}: {val:.1f}', (val, cdf_values[idx]), 
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    
    # Grid and labels
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Value')
    ax.set_ylabel('Cumulative Probability')
    ax.set_xlim(left=data.min()*0.95, right=data.max()*1.05)
    ax.set_ylim(bottom=0, top=1.05)
    ax.set_title(f'{column}\nEmpirical Cumulative Distribution Function', fontsize=14, fontweight='bold')
    
    cdf_path = os.path.join(viz_subdir, f'{column}_cdf.png')
    plt.savefig(cdf_path, dpi=CHART_RESOLUTION, bbox_inches='tight')
    chart_paths['cdf'] = cdf_path
    plt.close()
    
    # =========================================================================
    # CHART 6: COMBINED MULTI-PANEL VIEW
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'{column}\nComprehensive Distribution Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Panel A: Histogram
    ax = axes[0, 0]
    n_bins = int(np.sqrt(len(data)))
    ax.hist(data, bins=n_bins, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(stats_dict['mean'], color='red', linestyle='--', linewidth=2, label='Mean')
    ax.axvline(stats_dict['median'], color='green', linestyle='-', linewidth=2, label='Median')
    ax.set_title('(A) Histogram', fontsize=12, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=8)
    
    # Panel B: Box Plot
    ax = axes[0, 1]
    bp = ax.boxplot(data, vert=True, patch_artist=True, showfliers=True,
                    flierprops=dict(marker='o', markerfacecolor='red', 
                                    markersize=4, alpha=0.5))
    for patch in bp['boxes']:
        patch.set_facecolor('lightgreen')
    ax.set_title('(B) Box Plot', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value')
    ax.set_xticks([1])
    ax.set_xticklabels([''])
    
    # Panel C: KDE Density
    ax = axes[1, 0]
    sns.kdeplot(data, shade=True, color='purple', ax=ax)
    ax.axvline(stats_dict['mean'], color='red', linestyle='--', linewidth=2)
    ax.axvline(stats_dict['median'], color='green', linestyle='-', linewidth=2)
    ax.set_title('(C) Density Curve (KDE)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    
    # Panel D: Empirical CDF
    ax = axes[1, 1]
    sorted_data = np.sort(data)
    cdf_vals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax.step(sorted_data, cdf_vals, where='post', linewidth=2, color='darkblue')
    ax.axhline(0.5, color='green', linestyle='--', alpha=0.5, label='Median (P50)')
    ax.set_title('(D) Empirical CDF', fontsize=12, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Cumulative Probability')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    
    combined_path = os.path.join(viz_subdir, f'{column}_combined.png')
    plt.savefig(combined_path, dpi=CHART_RESOLUTION, bbox_inches='tight')
    chart_paths['combined'] = combined_path
    plt.close()
    
    print(f"  ✓ Generated 6 charts for column: {column}")
    return chart_paths

def generate_report_with_charts(results_list: list, output_path: str, 
                                viz_dir: str) -> None:
    """Generate formal report with embedded chart references."""
    
    report_lines = []
    separator = "=" * SECTION_WIDTH
    
    # Header
    report_lines.append(separator)
    report_lines.append(REPORT_TITLE.upper())
    report_lines.append(separator)
    report_lines.append("")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total Columns Analyzed: {len(results_list)}")
    report_lines.append(f"Visualizations Directory: {viz_dir}")
    report_lines.append("")
    
    # Executive Summary
    report_lines.append(separator)
    report_lines.append("EXECUTIVE SUMMARY")
    report_lines.append(separator)
    report_lines.append("")
    
    for results in results_list:
        col = results.get('column_name', 'Unknown')
        n = results.get('sample_size', 0)
        mean = results.get('mean', 'N/A')
        std = results.get('std_deviation', 'N/A')
        skew = results.get('skewness', 'N/A')
        outliers = results.get('total_outliers', 0)
        
        report_lines.append(f"Column: {col}")
        report_lines.append(f"  - Sample Size: {n:,} observations")
        report_lines.append(f"  - Mean: {format_number(mean, DECIMAL_PRECISION)}")
        report_lines.append(f"  - Std Dev: {format_number(std, DECIMAL_PRECISION)}")
        report_lines.append(f"  - Skewness: {format_number(skew, 2)}")
        report_lines.append(f"  - Outliers: {outliers}")
        
        # Distribution assessment
        if isinstance(skew, (int, float)):
            if abs(skew) < 0.5:
                dist_type = "approximately symmetric"
            elif skew > 0.5:
                dist_type = "right-skewed (positive)"
            else:
                dist_type = "left-skewed (negative)"
            report_lines.append(f"  - Distribution: {dist_type}")
        
        report_lines.append("")
    
    # Methodology
    report_lines.append(separator)
    report_lines.append("METHODOLOGY")
    report_lines.append(separator)
    report_lines.append("")
    report_lines.append("STATISTICAL MEASURES:")
    report_lines.append("  1. MEAN     : Arithmetic average (sum/count)")
    report_lines.append("  2. MEDIAN   : Middle value (robust to outliers)")
    report_lines.append("  3. MODE     : Most frequent value")
    report_lines.append("  4. VARIANCE : Average squared deviation from mean")
    report_lines.append("  5. STD DEV  : Square root of variance (original units)")
    report_lines.append("  6. RANGE    : Difference between max and min")
    report_lines.append("  7. PERCENTILES: 25th, 50th, 75th, 90th, 95th")
    report_lines.append("  8. QUARTILES: Q1, Q2 (median), Q3")
    report_lines.append("  9. IQR      : Interquartile range (Q3 - Q1)")
    report_lines.append("")
    report_lines.append("VISUALIZATIONS (6 per column):")
    report_lines.append("  A. Histogram with Density Overlay")
    report_lines.append("  B. Box Plot (Outlier Detection)")
    report_lines.append("  C. Q-Q Plot (Normality Assessment)")
    report_lines.append("  D. Violin Plot (Distribution Shape)")
    report_lines.append("  E. Empirical CDF (Cumulative Distribution)")
    report_lines.append("  F. Combined Multi-Panel View")
    report_lines.append("")
    
    # Detailed Results
    report_lines.append(separator)
    report_lines.append("DETAILED RESULTS BY COLUMN")
    report_lines.append(separator)
    
    for results in results_list:
        report_lines.append("")
        col = results.get('column_name', 'Unknown')
        
        # Box drawing for column header
        report_lines.append(f"╔{'═' * (SECTION_WIDTH-2)}╗")
        report_lines.append(f"║ COLUMN: {col:<{SECTION_WIDTH-15}}║")
        report_lines.append(f"╚{'═' * (SECTION_WIDTH-2)}╝")
        report_lines.append("")
        
        # Statistics
        report_lines.append("  CENTRAL TENDENCY")
        report_lines.append(f"    • Mean:       {format_number(results['mean'], DECIMAL_PRECISION)}")
        report_lines.append(f"    • Median:     {format_number(results['median'], DECIMAL_PRECISION)}")
        report_lines.append(f"    • Mode:       {format_number(results['mode_value'], DECIMAL_PRECISION)} (frequency: {results['mode_count']})")
        report_lines.append(f"    • Skewness:   {format_number(results['skewness'], 2)}")
        report_lines.append("")
        
        report_lines.append("  DISPERSION MEASURES")
        report_lines.append(f"    • Variance:        {format_number(results['variance'], DECIMAL_PRECISION)}")
        report_lines.append(f"    • Std Deviation:   {format_number(results['std_deviation'], DECIMAL_PRECISION)}")
        report_lines.append(f"    • Range:           {format_number(results['range'], DECIMAL_PRECISION)}")
        report_lines.append(f"    • Minimum:         {format_number(results['min_value'], DECIMAL_PRECISION)}")
        report_lines.append(f"    • Maximum:         {format_number(results['max_value'], DECIMAL_PRECISION)}")
        report_lines.append(f"    • Kurtosis:        {format_number(results['kurtosis'], 2)}")
        report_lines.append("")
        
        report_lines.append("  QUANTILE-BASED MEASURES")
        report_lines.append(f"    • P25 (Q1): {format_number(results['percentile_25'], DECIMAL_PRECISION)}")
        report_lines.append(f"    • P50 (Q2): {format_number(results['percentile_50'], DECIMAL_PRECISION)}")
        report_lines.append(f"    • P75 (Q3): {format_number(results['percentile_75'], DECIMAL_PRECISION)}")
        report_lines.append(f"    • P90:      {format_number(results['percentile_90'], DECIMAL_PRECISION)}")
        report_lines.append(f"    • P95:      {format_number(results['percentile_95'], DECIMAL_PRECISION)}")
        report_lines.append(f"    • IQR:      {format_number(results['iqr'], DECIMAL_PRECISION)}")
        report_lines.append("")
        
        report_lines.append("  OUTLIER DETECTION (1.5×IQR Method)")
        report_lines.append(f"    • Lower Fence:   {format_number(results['lower_fence'], DECIMAL_PRECISION)}")
        report_lines.append(f"    • Upper Fence:   {format_number(results['upper_fence'], DECIMAL_PRECISION)}")
        report_lines.append(f"    • Outliers Below: {results['outliers_below']}")
        report_lines.append(f"    • Outliers Above: {results['outliers_above']}")
        report_lines.append(f"    • Total Outliers: {results['total_outliers']} ({calculate_outlier_pct(results):.2f}% of data)")
        report_lines.append("")
        
        # Chart references
        report_lines.append("  VISUALIZATIONS GENERATED:")
        viz_files = [
            ('Histogram with Density', f'{col}_histogram.png'),
            ('Box Plot (Outliers)', f'{col}_boxplot.png'),
            ('Q-Q Plot (Normality)', f'{col}_qqplot.png'),
            ('Violin Plot', f'{col}_violinplot.png'),
            ('CDF', f'{col}_cdf.png'),
            ('Combined Multi-Panel', f'{col}_combined.png'),
        ]
        for desc, filename in viz_files:
            full_path = os.path.join(viz_dir, filename)
            report_lines.append(f"    • {desc}: {full_path}")
        report_lines.append("")
    
    # Recommendations
    report_lines.append(separator)
    report_lines.append("AUTOMATED INSIGHTS & RECOMMENDATIONS")
    report_lines.append(separator)
    report_lines.append("")
    
    for results in results_list:
        col = results.get('column_name', 'Unknown')
        outliers = results.get('total_outliers', 0)
        sample = results.get('sample_size', 0)
        skew = results.get('skewness', 0)
        kurt = results.get('kurtosis', 0)
        
        report_lines.append(f"Column: {col}")
        
        # Outlier analysis
        if outliers > 0:
            pct = (outliers / sample) * 100 if sample > 0 else 0
            if pct > 5:
                report_lines.append(f"  ⚠ HIGH OUTLIER PRESENCE ({pct:.1f}%): Investigate data quality")
            else:
                report_lines.append(f"  ℹ Some outliers detected ({outliers}): May be legitimate extreme values")
        
        # Skewness interpretation
        if isinstance(skew, (int, float)) and abs(skew) > 0.5:
            direction = "right-skewed" if skew > 0 else "left-skewed"
            report_lines.append(f"  ⚠ {direction.capitalize()} distribution (skewness={skew:.2f}): Consider log/box-cox transformation for parametric tests")
        
        # Kurtosis interpretation
        if isinstance(kurt, (int, float)) and abs(kurt) > 1:
            heavy = "heavy-tailed" if kurt > 0 else "light-tailed"
            report_lines.append(f"  ℹ {heavy.capitalize()} distribution (kurtosis={kurt:.2f}): More/less extreme values than normal")
        
        # Shapiro test (normality)
        if results.get('shapiro_p'):
            p_val = results['shapiro_p']
            if p_val < 0.05:
                report_lines.append(f"  ⚠ Non-normal distribution (Shapiro p={p_val:.4f}): Use non-parametric methods")
            else:
                report_lines.append(f"  ✓ Approximately normal (Shapiro p={p_val:.4f}): Parametric tests applicable")
        
        report_lines.append("")
    
    # Footer
    report_lines.append(separator)
    report_lines.append("END OF REPORT")
    report_lines.append(f"Review visualizations in: {viz_dir}")
    report_lines.append(separator)
    
    # Write report
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        print(f"\n[SUCCESS] Report successfully written to: {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to write report: {e}")
        sys.exit(1)

def format_number(value, precision: int = 4) -> str:
    """Format numeric values with consistent decimal precision."""
    if isinstance(value, (int, float)):
        if np.isnan(value) or np.isinf(value):
            return "N/A"
        return f"{value:,.{precision}f}"
    elif isinstance(value, str):
        return value
    else:
        return str(value)

def calculate_outlier_pct(results: dict) -> float:
    """Calculate percentage of outliers in a column."""
    total = results.get('sample_size', 0)
    outliers = results.get('total_outliers', 0)
    if total > 0:
        return (outliers / total) * 100
    return 0.0

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Summary Statistics & Visualization Template for EDA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input data.csv
  %(prog)s --input data.csv --column new_cases
  %(prog)s --input data.csv --output analysis_report.txt

Output:
  - Text report: output path specified (--output)
  - Charts: automatically saved in <output_parent>/visualizations/
        """
    )
    
    parser.add_argument('-i', '--input', required=True, help='Path to input CSV file')
    parser.add_argument('-c', '--column', default=None, help='Specific column to analyze (optional)')
    parser.add_argument('-o', '--output', default='eda_report.txt', help='Output report filename')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization generation (faster)')
    
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("\n" + "=" * SECTION_WIDTH)
    print(f"  EXPLORATORY DATA ANALYSIS - TEMPLATE WITH VISUALIZATION")
    print("=" * SECTION_WIDTH)
    print(f"Input File:    {args.input}")
    print(f"Target Column: {args.column or '(all numeric columns, max 5)'}")
    print(f"Output Report: {args.output}")
    print(f"Visualizations: {'Enabled' if not args.no_viz else 'Disabled (--no-viz)'}")
    print("=" * SECTION_WIDTH + "\n")
    
    # Create output directory structure
    output_path = Path(args.output).resolve()
    output_dir = output_path.parent
    viz_dir = os.path.join(output_dir, 'visualizations')
    
    if not args.no_viz:
        os.makedirs(viz_dir, exist_ok=True)
        print(f"[INFO] Visualizations will be saved to: {viz_dir}")
    
    # Load data
    df, target_columns = load_and_validate_data(args.input, args.column)
    
    # Compute statistics for each column
    all_results = []
    for column in target_columns:
        print(f"\n[ANALYSIS] Processing column: {column}")
        results = compute_summary_statistics(df, column)
        all_results.append(results)
        
        # Console preview
        if 'error' not in results:
            print(f"  ✓ Mean:     {format_number(results['mean'], DECIMAL_PRECISION)}")
            print(f"  ✓ Median:   {format_number(results['median'], DECIMAL_PRECISION)}")
            print(f"  ✓ Std Dev:  {format_number(results['std_deviation'], DECIMAL_PRECISION)}")
            print(f"  ✓ IQR:      {format_number(results['iqr'], DECIMAL_PRECISION)}")
            print(f"  ✓ Skewness: {format_number(results['skewness'], 2)}")
            print(f"  ✓ Outliers: {results['total_outliers']}")
            
            # Generate visualizations
            if not args.no_viz:
                create_visualizations(df, column, viz_dir, results)
    
    # Generate report
    generate_report_with_charts(all_results, str(output_path), viz_dir)
    
    # Completion
    print("\n" + "-" * SECTION_WIDTH)
    print("Analysis complete!")
    print(f"  • Report:     {output_path}")
    if not args.no_viz:
        print(f"  • Charts:     {viz_dir}/")
    print("-" * SECTION_WIDTH + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())