"""
===============================================================================
EXPLORATORY DATA ANALYSIS - SUMMARY STATISTICS TEMPLATE
===============================================================================

Author: [Your Name]
Date: Generated Template
Description: General-purpose template for computing summary statistics on
             any tabular dataset following EDA best practices.

This template implements all 9 core statistical measures from the EDA Cookbook:
  1. Mean (average)
  2. Median (middle value)
  3. Mode (most frequent value)
  4. Variance (spread/variability)
  5. Standard Deviation (square root of variance)
  6. Range (max - min)
  7. Percentiles (divides data into 100 portions)
  8. Quartiles (divides data into 4 portions)
  9. Interquartile Range (IQR - middle 50% spread)

Usage:
    python summary_stats_template.py --input data.csv --output report.txt

Dependencies:
    - pandas >= 1.0.0
    - numpy >= 1.18.0
    - scipy >= 1.4.0
===============================================================================
"""

# -----------------------------------------------------------------------------
# IMPORT STATEMENTS
# -----------------------------------------------------------------------------
# Import required libraries for data manipulation and statistical analysis
import pandas as pd          # Data frame operations and CSV handling
import numpy as np           # Numerical computations and array operations
from scipy import stats      # Advanced statistical functions (mode, IQR)
import argparse              # Command-line argument parsing
from datetime import datetime  # Timestamp for report generation
import sys                   # System exit codes for error handling


# -----------------------------------------------------------------------------
# CONFIGURATION CONSTANTS
# -----------------------------------------------------------------------------
# Define constants for consistent formatting throughout the script
REPORT_TITLE = "Exploratory Data Analysis - Summary Statistics Report"
SECTION_WIDTH = 80  # Width for section dividers
DECIMAL_PRECISION = 4  # Decimal places for floating point numbers


def load_and_validate_data(filepath: str, target_column: str = None) -> tuple:
    """
    Load CSV data and validate structure.
    
    Parameters:
        filepath (str): Path to the input CSV file
        target_column (str, optional): Specific column to analyze; if None,
                                       analyzes all numeric columns
    
    Returns:
        tuple: (DataFrame object, list of target columns to analyze)
    
    Raises:
        FileNotFoundError: If the specified file does not exist
        ValueError: If no numeric columns found in the dataset
    """
    
    try:
        # Attempt to read the CSV file into a pandas DataFrame
        df = pd.read_csv(filepath)
        
        # Log basic dataset information
        print(f"[INFO] Successfully loaded {filepath}")
        print(f"[INFO] Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Identify numeric columns for statistical analysis
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Validate that numeric columns exist in the dataset
        if len(numeric_columns) == 0:
            raise ValueError("No numeric columns found in the dataset!")
        
        # Determine which columns to analyze
        if target_column:
            # Analyze only specified column if provided
            if target_column in numeric_columns:
                target_columns = [target_column]
                print(f"[INFO] Analyzing specific column: {target_column}")
            else:
                raise ValueError(f"Column '{target_column}' not found or not numeric!")
        else:
            # Analyze all numeric columns by default
            target_columns = numeric_columns
            print(f"[INFO] Analyzing all {len(target_columns)} numeric columns: {target_columns}")
        
        return df, target_columns
        
    except FileNotFoundError:
        # Handle missing file gracefully with informative message
        print(f"[ERROR] File not found: {filepath}")
        print("[HINT] Please verify the file path is correct.")
        sys.exit(1)
    except Exception as e:
        # Catch-all for other unexpected errors during loading
        print(f"[ERROR] Failed to load data: {e}")
        sys.exit(1)


def compute_summary_statistics(df: pd.DataFrame, column: str) -> dict:
    """
    Compute all 9 summary statistics for a given numeric column.
    
    This function encapsulates the core analytical workflow described in the
    EDA Cookbook, implementing each statistical measure with proper error
    handling for edge cases (empty data, non-numeric values, etc.).
    
    Parameters:
        df (pd.DataFrame): Input pandas DataFrame containing the data
        column (str): Name of the column to analyze
    
    Returns:
        dict: Dictionary containing all computed statistics with metadata
    
    Note:
        Missing values (NaN) are automatically excluded from calculations
        by numpy/pandas functions unless explicitly configured otherwise.
    """
    
    # Extract the target column as a numpy array for efficient computation
    # dropna() ensures NaN values don't interfere with calculations
    data = df[column].dropna().values
    
    # Validate that we have actual data to work with
    if len(data) == 0:
        return {"error": f"No valid numeric data in column '{column}'"}
    
    # -------------------------------------------------------------------------
    # STATISTIC 1: MEAN (Arithmetic Average)
    # -------------------------------------------------------------------------
    # The mean is sensitive to outliers but provides a central tendency measure
    # Formula: sum(data) / count(data)
    try:
        mean_value = np.mean(data)
    except Exception as e:
        mean_value = f"Error: {e}"
    
    # -------------------------------------------------------------------------
    # STATISTIC 2: MEDIAN (Middle Value)
    # -------------------------------------------------------------------------
    # The median is robust to outliers, representing the 50th percentile
    # For odd n: middle value; For even n: average of two middle values
    try:
        median_value = np.median(data)
    except Exception as e:
        median_value = f"Error: {e}"
    
    # -------------------------------------------------------------------------
    # STATISTIC 3: MODE (Most Frequent Value)
    # -------------------------------------------------------------------------
    # Mode can apply to numeric AND categorical data, identifies peak frequency
    # Using scipy.stats.mode for proper mode detection with count
    try:
        mode_result = stats.mode(data)
        # scipy.mode returns ModeResult object; extract mode value and count
        mode_value = mode_result.mode[0]
        mode_count = mode_result.count[0]
    except Exception as e:
        mode_value = f"Error: {e}"
        mode_count = None
    
    # -------------------------------------------------------------------------
    # STATISTIC 4: VARIANCE (Average Squared Deviation from Mean)
    # -------------------------------------------------------------------------
    # Variance measures overall spread; squared units make interpretation harder
    # Sample variance (ddof=1) vs Population variance (ddof=0) - using default ddof=0
    try:
        variance_value = np.var(data)
    except Exception as e:
        variance_value = f"Error: {e}"
    
    # -------------------------------------------------------------------------
    # STATISTIC 5: STANDARD DEVIATION (Square Root of Variance)
    # -------------------------------------------------------------------------
    # Std dev restores original units, making it more interpretable than variance
    # Same sensitivity to outliers as variance, but easier to communicate
    try:
        std_dev_value = np.std(data)
    except Exception as e:
        std_dev_value = f"Error: {e}"
    
    # -------------------------------------------------------------------------
    # STATISTIC 6: RANGE (Difference Between Max and Min)
    # -------------------------------------------------------------------------
    # Range shows total spread but is highly sensitive to extreme outliers
    # Useful as a quick first-pass measure of variability
    try:
        max_value = np.max(data)
        min_value = np.min(data)
        range_value = max_value - min_value
    except Exception as e:
        max_value = min_value = range_value = f"Error: {e}"
    
    # -------------------------------------------------------------------------
    # STATISTIC 7: PERCENTILES (Position-Based Thresholds)
    # -------------------------------------------------------------------------
    # Percentiles divide data into 100 equal portions; 50th percentile = median
    # Computing key percentiles: 25th, 50th, 75th (matching quartile definitions)
    try:
        p25 = np.percentile(data, 25)   # First quartile boundary
        p50 = np.percentile(data, 50)   # Equal to median
        p75 = np.percentile(data, 75)   # Third quartile boundary
        p90 = np.percentile(data, 90)   # Top 10% threshold
        p95 = np.percentile(data, 95)   # Top 5% threshold
    except Exception as e:
        p25 = p50 = p75 = p90 = p95 = f"Error: {e}"
    
    # -------------------------------------------------------------------------
    # STATISTIC 8: QUARTILES (Four Equal Portions)
    # -------------------------------------------------------------------------
    # Q1 = 25th percentile, Q2 = median (50th), Q3 = 75th percentile
    # Explicitly calculated for clarity and cross-validation with percentiles
    try:
        q1 = np.quantile(data, 0.25)
        q2 = np.quantile(data, 0.50)
        q3 = np.quantile(data, 0.75)
    except Exception as e:
        q1 = q2 = q3 = f"Error: {e}"
    
    # -------------------------------------------------------------------------
    # STATISTIC 9: INTERQUARTILE RANGE (IQR - Middle 50% Spread)
    # -------------------------------------------------------------------------
    # IQR = Q3 - Q1, robust to outliers since it ignores extremes
    # Critical for outlier detection: points beyond Q1 - 1.5×IQR or Q3 + 1.5×IQR
    try:
        iqr_value = stats.iqr(data)
        # Calculate outlier boundaries for diagnostic purposes
        lower_fence = q1 - 1.5 * iqr_value
        upper_fence = q3 + 1.5 * iqr_value
        # Count actual outliers in the dataset
        outliers_low = np.sum(data < lower_fence)
        outliers_high = np.sum(data > upper_fence)
        total_outliers = outliers_low + outliers_high
    except Exception as e:
        iqr_value = lower_fence = upper_fence = None
        outliers_low = outliers_high = total_outliers = 0
    
    # -------------------------------------------------------------------------
    # COMPILE ALL RESULTS INTO DICTIONARY
    # -------------------------------------------------------------------------
    results = {
        'column_name': column,
        'sample_size': len(data),
        'missing_values': len(df[column]) - len(data),
        
        # Central Tendency Measures
        'mean': mean_value,
        'median': median_value,
        'mode_value': mode_value,
        'mode_count': mode_count,
        
        # Dispersion Measures
        'variance': variance_value,
        'std_deviation': std_dev_value,
        'range': range_value,
        'min_value': min_value,
        'max_value': max_value,
        
        # Quantile-Based Measures
        'percentile_25': p25,
        'percentile_50': p50,
        'percentile_75': p75,
        'percentile_90': p90,
        'percentile_95': p95,
        'quartile_1': q1,
        'quartile_2': q2,
        'quartile_3': q3,
        'iqr': iqr_value,
        
        # Outlier Diagnostics
        'lower_fence': lower_fence,
        'upper_fence': upper_fence,
        'outliers_below': outliers_low,
        'outliers_above': outliers_high,
        'total_outliers': total_outliers,
    }
    
    return results


def generate_report(results_list: list, output_path: str) -> None:
    """
    Generate a formal text-based report from analysis results.
    
    Creates a professional, formatted report suitable for documentation,
    sharing with stakeholders, or inclusion in technical deliverables.
    
    Parameters:
        results_list (list): List of dictionaries from compute_summary_statistics()
        output_path (str): File path for the output report (.txt or .md)
    
    Note:
        Report includes sections for methodology, results, and recommendations
        following standard technical documentation conventions.
    """
    
    # Build report content as a list of strings for efficient concatenation
    report_lines = []
    
    # -------------------------------------------------------------------------
    # REPORT HEADER SECTION
    # -------------------------------------------------------------------------
    separator = "=" * SECTION_WIDTH
    report_lines.append(separator)
    report_lines.append(REPORT_TITLE.upper())
    report_lines.append(separator)
    report_lines.append("")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total Columns Analyzed: {len(results_list)}")
    report_lines.append("")
    
    # -------------------------------------------------------------------------
    # EXECUTIVE SUMMARY SECTION
    # -------------------------------------------------------------------------
    report_lines.append(separator)
    report_lines.append("EXECUTIVE SUMMARY")
    report_lines.append(separator)
    report_lines.append("")
    
    # Compile quick overview of key findings
    for results in results_list:
        col = results.get('column_name', 'Unknown')
        n = results.get('sample_size', 0)
        mean = results.get('mean', 'N/A')
        std = results.get('std_deviation', 'N/A')
        
        report_lines.append(f"Column: {col}")
        report_lines.append(f"  - Sample Size: {n:,} observations")
        report_lines.append(f"  - Mean: {format_number(mean, DECIMAL_PRECISION)}")
        report_lines.append(f"  - Std Dev: {format_number(std, DECIMAL_PRECISION)}")
        report_lines.append("")
    
    # -------------------------------------------------------------------------
    # METHODOLOGY SECTION
    # -------------------------------------------------------------------------
    report_lines.append(separator)
    report_lines.append("METHODOLOGY")
    report_lines.append(separator)
    report_lines.append("")
    report_lines.append("This analysis computes nine core summary statistics for each")
    report_lines.append("numeric column in the dataset:")
    report_lines.append("")
    report_lines.append("  1. MEAN    : Arithmetic average (sum/count)")
    report_lines.append("  2. MEDIAN  : Middle value (robust to outliers)")
    report_lines.append("  3. MODE    : Most frequent value")
    report_lines.append("  4. VARIANCE: Average squared deviation from mean")
    report_lines.append("  5. STD DEV : Square root of variance (original units)")
    report_lines.append("  6. RANGE   : Difference between max and min")
    report_lines.append("  7. PERCENTILES: 25th, 50th, 75th, 90th, 95th")
    report_lines.append("  8. QUARTILES: Q1, Q2 (median), Q3")
    report_lines.append("  9. IQR     : Interquartile range (Q3 - Q1)")
    report_lines.append("")
    report_lines.append("Libraries Used: pandas, numpy, scipy")
    report_lines.append("")
    
    # -------------------------------------------------------------------------
    # DETAILED RESULTS SECTION
    # -------------------------------------------------------------------------
    report_lines.append(separator)
    report_lines.append("DETAILED RESULTS BY COLUMN")
    report_lines.append(separator)
    
    for results in results_list:
        report_lines.append("")
        col = results.get('column_name', 'Unknown')
        report_lines.append(f"┌{'─' * (SECTION_WIDTH-2)}┐")
        report_lines.append(f"│ COLUMN: {col:<{SECTION_WIDTH-15}}│")
        report_lines.append(f"└{'─' * (SECTION_WIDTH-2)}┘")
        report_lines.append("")
        
        # Central Tendency Subsection
        report_lines.append("  CENTRAL TENDENCY")
        report_lines.append(f"    • Mean:       {format_number(results['mean'], DECIMAL_PRECISION)}")
        report_lines.append(f"    • Median:     {format_number(results['median'], DECIMAL_PRECISION)}")
        report_lines.append(f"    • Mode:       {format_number(results['mode_value'], DECIMAL_PRECISION)} (count: {results['mode_count']})")
        report_lines.append("")
        
        # Dispersion Subsection
        report_lines.append("  DISPERSION MEASURES")
        report_lines.append(f"    • Variance:        {format_number(results['variance'], DECIMAL_PRECISION)}")
        report_lines.append(f"    • Std Deviation:   {format_number(results['std_deviation'], DECIMAL_PRECISION)}")
        report_lines.append(f"    • Range:           {format_number(results['range'], DECIMAL_PRECISION)}")
        report_lines.append(f"    • Minimum:         {format_number(results['min_value'], DECIMAL_PRECISION)}")
        report_lines.append(f"    • Maximum:         {format_number(results['max_value'], DECIMAL_PRECISION)}")
        report_lines.append("")
        
        # Quantile Subsection
        report_lines.append("  QUANTILE-BASED MEASURES")
        report_lines.append(f"    • P25 (Q1): {format_number(results['percentile_25'], DECIMAL_PRECISION)}")
        report_lines.append(f"    • P50 (Q2): {format_number(results['percentile_50'], DECIMAL_PRECISION)}")
        report_lines.append(f"    • P75 (Q3): {format_number(results['percentile_75'], DECIMAL_PRECISION)}")
        report_lines.append(f"    • P90:      {format_number(results['percentile_90'], DECIMAL_PRECISION)}")
        report_lines.append(f"    • P95:      {format_number(results['percentile_95'], DECIMAL_PRECISION)}")
        report_lines.append(f"    • IQR:      {format_number(results['iqr'], DECIMAL_PRECISION)}")
        report_lines.append("")
        
        # Outlier Detection Subsection
        report_lines.append("  OUTLIER DETECTION (1.5×IQR Method)")
        report_lines.append(f"    • Lower Fence:   {format_number(results['lower_fence'], DECIMAL_PRECISION)}")
        report_lines.append(f"    • Upper Fence:   {format_number(results['upper_fence'], DECIMAL_PRECISION)}")
        report_lines.append(f"    • Outliers Below: {results['outliers_below']}")
        report_lines.append(f"    • Outliers Above: {results['outliers_above']}")
        report_lines.append(f"    • Total Outliers: {results['total_outliers']} ({calculate_outlier_pct(results), 2}% of data)")
        report_lines.append("")
    
    # -------------------------------------------------------------------------
    # RECOMMENDATIONS SECTION
    # -------------------------------------------------------------------------
    report_lines.append(separator)
    report_lines.append("RECOMMENDATIONS")
    report_lines.append(separator)
    report_lines.append("")
    
    # Generate data-driven recommendations based on analysis
    recommendations = []
    for results in results_list:
        col = results.get('column_name', 'Unknown')
        outliers = results.get('total_outliers', 0)
        sample = results.get('sample_size', 0)
        
        if outliers > 0:
            pct = (outliers / sample) * 100 if sample > 0 else 0
            if pct > 5:
                recommendations.append(f"• {col}: High outlier presence ({pct:.1f}%) - investigate data quality")
            else:
                recommendations.append(f"• {col}: Some outliers detected ({outliers}) - may be legitimate extreme values")
        
        # Check skewness via mean vs median comparison
        mean_val = results.get('mean', 0)
        median_val = results.get('median', 0)
        if isinstance(mean_val, (int, float)) and isinstance(median_val, (int, float)):
            if mean_val > median_val * 1.2:
                recommendations.append(f"• {col}: Right-skewed (mean >> median) - consider transformations")
            elif mean_val < median_val * 0.8:
                recommendations.append(f"• {col}: Left-skewed (mean << median) - consider transformations")
    
    if recommendations:
        for rec in recommendations:
            report_lines.append(rec)
    else:
        report_lines.append("• No major issues detected - data appears reasonably distributed")
    
    report_lines.append("")
    report_lines.append("• Consider visualizing distributions (histograms, box plots)")
    report_lines.append("• Validate outliers against business context before removal")
    report_lines.append("• Document any data cleaning decisions made post-analysis")
    report_lines.append("")
    
    # -------------------------------------------------------------------------
    # FOOTER SECTION
    # -------------------------------------------------------------------------
    report_lines.append(separator)
    report_lines.append("END OF REPORT")
    report_lines.append(separator)
    
    # Write the complete report to the output file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        print(f"\n[SUCCESS] Report successfully written to: {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to write report: {e}")
        sys.exit(1)


def format_number(value, precision: int = 4) -> str:
    """
    Format numeric values for display with consistent decimal precision.
    
    Handles various input types including floats, strings, and error messages.
    
    Parameters:
        value: Numeric value or string to format
        precision (int): Number of decimal places (default: 4)
    
    Returns:
        str: Formatted string representation
    """
    if isinstance(value, (int, float)):
        return f"{value:,.{precision}f}"
    elif isinstance(value, str):
        return value  # Return error messages and non-numeric strings unchanged
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
    """Parse command-line arguments for flexible script execution."""
    parser = argparse.ArgumentParser(
        description="Summary Statistics Template for Exploratory Data Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input data.csv
  %(prog)s --input data.csv --column new_cases
  %(prog)s --input data.csv --output analysis_report.txt

Notes:
  - If no column specified, all numeric columns are analyzed
  - Output format auto-detected by extension (.txt or .md)
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Path to input CSV file'
    )
    
    parser.add_argument(
        '-c', '--column',
        default=None,
        help='Specific column to analyze (optional)'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='eda_report.txt',
        help='Output report filename (default: eda_report.txt)'
    )
    
    return parser.parse_args()


def main():
    """
    Main execution function orchestrating the complete analysis pipeline.
    
    Workflow:
        1. Parse command-line arguments
        2. Load and validate input data
        3. Compute summary statistics for each target column
        4. Generate and save formal report
    
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    
    # Step 1: Parse user-provided arguments
    args = parse_arguments()
    
    print("\n" + "=" * SECTION_WIDTH)
    print(f"  EXPLORATORY DATA ANALYSIS - SUMMARY STATISTICS TEMPLATE")
    print("=" * SECTION_WIDTH)
    print(f"Input File:    {args.input}")
    print(f"Target Column: {args.column or '(all numeric columns)'}")
    print(f"Output Report: {args.output}")
    print("=" * SECTION_WIDTH + "\n")
    
    # Step 2: Load and validate the input dataset
    df, target_columns = load_and_validate_data(args.input, args.column)
    
    # Step 3: Compute statistics for each target column
    all_results = []
    for column in target_columns:
        print(f"\n[ANALYSIS] Processing column: {column}")
        results = compute_summary_statistics(df, column)
        all_results.append(results)
        
        # Print quick preview of results to console
        if 'error' not in results:
            print(f"  ✓ Mean:     {format_number(results['mean'], DECIMAL_PRECISION)}")
            print(f"  ✓ Median:   {format_number(results['median'], DECIMAL_PRECISION)}")
            print(f"  ✓ Std Dev:  {format_number(results['std_deviation'], DECIMAL_PRECISION)}")
            print(f"  ✓ IQR:      {format_number(results['iqr'], DECIMAL_PRECISION)}")
            print(f"  ✓ Outliers: {results['total_outliers']}")
    
    # Step 4: Generate the formal report
    generate_report(all_results, args.output)
    
    # Final completion message
    print("\n" + "-" * SECTION_WIDTH)
    print("Analysis complete! Review the output report for full details.")
    print("-" * SECTION_WIDTH + "\n")
    
    return 0


# Entry point check ensures main() runs only when script executed directly
if __name__ == "__main__":
    sys.exit(main())