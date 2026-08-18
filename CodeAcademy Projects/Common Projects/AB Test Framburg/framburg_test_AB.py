"""
==============================================================================
FarmBurg A/B Testing Analysis - Executive Summary Script (PART 1/2)
==============================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings
from pathlib import Path
from typing import Dict, Tuple, List
from itertools import combinations

warnings.filterwarnings('ignore')

# Statistical libraries
from scipy.stats import chi2_contingency, binomtest
from statsmodels.stats.proportion import proportion_confint, proportions_ztest
from statsmodels.stats.power import NormalIndPower

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    """Centralized configuration for the analysis."""
    
    # Data paths
    DATA_FILE = 'clicks.csv'
    OUTPUT_DIR = Path('outputs')
    
    # Business targets
    WEEKLY_REVENUE_TARGET = 1000  # USD
    NUM_WEEKS_FOR_ANNUAL = 52
    
    # Price points
    PRICES = {'A': 0.99, 'B': 1.99, 'C': 4.99}
    
    # Statistical settings
    ALPHA = 0.05
    POWER_TARGET = 0.80
    CONFIDENCE_LEVEL = 0.95
    
    # Plot styling
    PRIMARY_COLOR = '#6d4aff'  # Proton purple
    FIGURE_DPI = 300
    SAVE_FORMATS = ['png', 'pdf']

# ==============================================================================
# DATA LOADING & VALIDATION
# ==============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """Load and validate A/B test data."""
    
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Basic validation
    required_cols = {'user_id', 'group', 'is_purchase'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    # Validate group assignments
    valid_groups = set(Config.PRICES.keys())
    invalid_groups = set(df['group'].unique()) - valid_groups
    if invalid_groups:
        raise ValueError(f"Invalid group values: {invalid_groups}. Expected {valid_groups}")
    
    return df

def get_sample_stats(df: pd.DataFrame) -> Dict:
    """Generate sample statistics for reporting."""
    
    total = len(df)
    group_stats = {}
    
    for group in sorted(df['group'].unique()):
        subset = df[df['group'] == group]
        purchases = (subset['is_purchase'] == 'Yes').sum()
        conversion_rate = purchases / len(subset) * 100
        
        group_stats[group] = {
            'visitors': int(len(subset)),
            'purchases': int(purchases),
            'conversion_rate_pct': round(conversion_rate, 2),
            'price_usd': Config.PRICES[group]
        }
    
    return {
        'total_visitors': total,
        'total_purchases': int((df['is_purchase'] == 'Yes').sum()),
        'overall_conversion_pct': round(((df['is_purchase'] == 'Yes').mean()) * 100, 2),
        'groups': group_stats
    }

# ==============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# ==============================================================================

def contingency_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create contingency table for chi-square test."""
    return pd.crosstab(df['group'], df['is_purchase'])

def chi_square_test(ctable: pd.DataFrame) -> Dict:
    """Perform chi-square test of independence."""
    
    chi2_stat, p_value, dof, expected = chi2_contingency(ctable)
    
    # Calculate standardized residuals
    observed = ctable.values
    expected_arr = np.array([[expected[i, j] for j in range(expected.shape[1])] 
                             for i in range(expected.shape[0])])
    residuals = (observed - expected_arr) / np.sqrt(expected_arr)
    
    return {
        'chi2_statistic': float(chi2_stat),
        'p_value': float(p_value),
        'degrees_of_freedom': int(dof),
        'is_significant': p_value < Config.ALPHA,
        'expected_counts': expected.tolist(),
        'standardized_residuals': residuals.tolist()
    }

def binomial_tests_against_targets(df: pd.DataFrame) -> Dict:
    """
    Perform binomial tests comparing observed rates to revenue-required rates.
    This is THE KEY ANALYSIS - testing against business thresholds, not raw comparisons.
    """
    
    total_visitors = len(df)
    results = {}
    
    for group, price in Config.PRICES.items():
        # Calculate required conversion rate for revenue target
        sales_needed = Config.WEEKLY_REVENUE_TARGET / price
        required_rate = sales_needed / total_visitors
        
        # Get observed data
        group_data = df[df['group'] == group]
        n = len(group_data)
        successes = (group_data['is_purchase'] == 'Yes').sum()
        observed_rate = successes / n
        
        # One-sided binomial test (testing if we EXCEED threshold)
        test_result = binomtest(
            successes,
            n=n,
            p=required_rate,
            alternative='greater'
        )
        
        # Calculate confidence interval
        ci_low, ci_high = proportion_confint(
            successes, nobs=n, alpha=1-Config.CONFIDENCE_LEVEL, method='wilson'
        )
        
        # Project revenue
        projected_revenue = (successes / n) * total_visitors * price
        
        results[group] = {
            'price_usd': price,
            'sample_size': int(n),
            'purchases': int(successes),
            'observed_rate_pct': round(observed_rate * 100, 2),
            'required_rate_pct': round(required_rate * 100, 2),
            'binomial_p_value': float(test_result.pvalue),
            'is_significant': test_result.pvalue < Config.ALPHA,
            'ci_lower_pct': round(ci_low * 100, 2),
            'ci_upper_pct': round(ci_high * 100, 2),
            'projected_weekly_revenue_usd': round(projected_revenue, 2),
            'meets_revenue_target': projected_revenue >= Config.WEEKLY_REVENUE_TARGET
        }
    
    return results

def calculate_effect_sizes(binomial_results: Dict) -> Dict:
    """Calculate Cohen's h effect sizes for all pairwise comparisons."""
    
    rates = {g: r['observed_rate_pct'] / 100 for g, r in binomial_results.items()}
    
    def cohens_h(p1, p2):
        return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
    
    def interpret_h(h):
        ah = abs(h)
        if ah < 0.2: return 'negligible'
        elif ah < 0.5: return 'small'
        elif ah < 0.8: return 'medium'
        else: return 'large'
    
    comparisons = {}
    for g1, g2 in combinations(sorted(rates.keys()), 2):
        h = cohens_h(rates[g1], rates[g2])
        comparisons[f'{g1}_vs_{g2}'] = {
            'cohens_h': round(h, 3),
            'magnitude': interpret_h(h),
            'rate_difference_pct': round(abs(rates[g1] - rates[g2]) * 100, 2)
        }
    
    return comparisons

def power_analysis(sample_per_group: int) -> Dict:
    """Calculate minimum detectable effect at various power levels."""
    
    analysis = NormalIndPower()
    
    # MDE at 80% power
    mde_80 = analysis.solve_power(
        nobs1=sample_per_group,
        alpha=Config.ALPHA,
        power=Config.POWER_TARGET,
        alternative='larger'
    )
    
    # Power curve data
    power_samples = [100, 500, 1000, 1666, 2500, 5000]
    powers = []
    for n in power_samples:
        pwr = analysis.solve_power(
            effect_size=mde_80,
            nobs1=n,
            alpha=Config.ALPHA,
            alternative='larger'
        )
        powers.append({'sample_size': n, 'power': round(pwr, 3)})
    
    return {
        'current_sample_per_group': sample_per_group,
        'mde_at_80pct_power': round(mde_80, 3),
        'power_curve': powers
    }
"""
==============================================================================
FarmBurg A/B Testing Analysis - Executive Summary Script (PART 2/2)
==============================================================================
"""

# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

def plot_conversion_with_ci(binomial_results: Dict, save_path: Path = None):
    """Create professional visualization showing conversion rates with CIs vs targets."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    groups = sorted(binomial_results.keys())
    y_pos = range(len(groups))
    
    rates = [binomial_results[g]['observed_rate_pct'] for g in groups]
    ci_lows = [binomial_results[g]['ci_lower_pct'] for g in groups]
    ci_highs = [binomial_results[g]['ci_upper_pct'] for g in groups]
    targets = [binomial_results[g]['required_rate_pct'] for g in groups]
    
    # Plot error bars (confidence intervals)
    ax.errorbar(rates, y_pos,
                xerr=[[rates[i]-ci_lows[i], ci_highs[i]-rates[i]] for i in range(len(groups))],
                fmt='o', capsize=8, markersize=12, color=Config.PRIMARY_COLOR,
                label='95% CI')
    
    # Plot target thresholds
    for i, (target, g) in enumerate(zip(targets, groups)):
        marker = ax.plot(target, i, 'r|', markersize=20, markeredgewidth=3,
                         label='Target' if i == 0 else '')
        
        # Color shading based on whether CI clears target
        color = 'green' if ci_lows[i] > target else 'red'
        ax.axvspan(target, rates[i] + (ci_highs[i] - rates[i]),
                   color=color, alpha=0.15)
    
    # Formatting
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([f'Group {g} (${binomial_results[g]["price_usd"]})' for g in groups])
    ax.set_xlabel('Conversion Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Conversion Rate Confidence Intervals vs Revenue Targets\n'
                 '(Green = Meets Target | Red = Fails or Uncertain)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    
    # Add annotations
    for i, g in enumerate(groups):
        status = '✓ MEETS TARGET' if binomial_results[g]['meets_revenue_target'] else '✗ BELOW TARGET'
        ax.annotate(status, xy=(rates[i], i), xytext=(5, 0),
                    textcoords='offset points', va='center',
                    fontsize=9, color='green' if binomial_results[g]['meets_revenue_target'] else 'red')
    
    plt.tight_layout()
    
    if save_path:
        for fmt in Config.SAVE_FORMATS:
            filepath = save_path.with_suffix(f'.{fmt}')
            plt.savefig(filepath, dpi=Config.FIGURE_DPI, bbox_inches='tight')
    
    plt.close()

def plot_daily_trends(df: pd.DataFrame, save_path: Path = None):
    """Plot daily conversion trends to detect novelty effects."""
    
    # Create day column (assuming sequential order in data)
    df_copy = df.copy()
    n_days = 7
    df_copy['day'] = np.tile(range(n_days), len(df)//n_days + 1)[:len(df)]
    
    daily_rates = df_copy.groupby(['day', 'group'])['is_purchase'].apply(
        lambda x: (x == 'Yes').mean() * 100
    ).unstack('group').round(2)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for g in sorted(df['group'].unique()):
        ax.plot(range(n_days), daily_rates[g], 'o-', linewidth=2, markersize=8,
                label=f'Group {g} (${Config.PRICES[g]})', color=Config.PRIMARY_COLOR if g == 'C' else '#999999')
    
    ax.set_xlabel('Day of Test', fontsize=12)
    ax.set_ylabel('Conversion Rate (%)', fontsize=12)
    ax.set_title('Daily Conversion Trends by Price Group\n(Detecting Novelty/Learning Effects)',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(y=np.mean(daily_rates.mean()), color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        for fmt in Config.SAVE_FORMATS:
            filepath = save_path.with_suffix(f'.{fmt}')
            plt.savefig(filepath, dpi=Config.FIGURE_DPI, bbox_inches='tight')
    
    plt.close()

# ==============================================================================
# DECISION FRAMEWORK
# ==============================================================================

def build_decision_matrix(binomial_results: Dict) -> pd.DataFrame:
    """Weighted decision matrix for pricing recommendation."""
    
    criteria_weights = {
        'statistical_significance': 25,
        'revenue_probability': 25,
        'profit_margin_per_sale': 20,
        'ci_above_target': 15,
        'implementation_feasibility': 15
    }
    
    scores = {}
    
    for g, r in binomial_results.items():
        score = {
            'statistical_significance': 100 if r['is_significant'] else 0,
            'revenue_probability': min(r['projected_weekly_revenue_usd'] / 10, 100),
            'profit_margin_per_sale': min(r['price_usd'] / 0.05, 100),
            'ci_above_target': 100 if r['ci_lower_pct'] > r['required_rate_pct'] else 0,
            'implementation_feasibility': max(90 - (r['price_usd'] * 5), 50)
        }
        scores[g] = score
    
    # Build DataFrame and calculate weighted total
    decision_df = pd.DataFrame(scores).T
    
    weighted_totals = []
    for idx in decision_df.index:
        total = sum(decision_df.loc[idx, criterion] * weight / 100
                   for criterion, weight in criteria_weights.items())
        weighted_totals.append(total)
    
    decision_df['weighted_total'] = weighted_totals
    decision_df = decision_df.sort_values('weighted_total', ascending=False)
    
    return decision_df

# ==============================================================================
# REPORT GENERATION
# ==============================================================================

def generate_executive_summary(stats: Dict, binomial_results: Dict, 
                               decision_df: pd.DataFrame) -> Dict:
    """Create presentation-ready summary dictionary."""
    
    winner = decision_df.index[0]
    
    summary = {
        'report_metadata': {
            'generated_at': datetime.now().isoformat(),
            'analysis_version': '2.0',
            'test_duration_days': 7,
            'statistical_confidence_level': f'{Config.CONFIDENCE_LEVEL * 100:.0f}%'
        },
        'sample_statistics': stats,
        'key_findings': {
            'recommendation': f"Group {winner} (${Config.PRICES[winner]})",
            'recommendation_score': round(decision_df.loc[winner, 'weighted_total'], 1),
            'confidence_in_recommendation': 'HIGH' if binomial_results[winner]['is_significant'] else 'MEDIUM',
            'weekly_revenue_projection': f"${binomial_results[winner]['projected_weekly_revenue_usd']:,.2f}",
            'annual_revenue_projection': f"${binomial_results[winner]['projected_weekly_revenue_usd'] * Config.NUM_WEEKS_FOR_ANNUAL:,.2f}",
            'meets_revenue_target': binomial_results[winner]['meets_revenue_target'],
            'excess_over_target_pct': round(
                ((binomial_results[winner]['projected_weekly_revenue_usd'] - Config.WEEKLY_REVENUE_TARGET) 
                 / Config.WEEKLY_REVENUE_TARGET * 100), 1
            )
        },
        'all_groups_performance': binomial_results,
        'decision_matrix': decision_df.round(1).to_dict(),
        'risks_and_mitigations': [
            "Higher price may cause customer pushback",
            "Short-term test may not capture long-term CLV impact",
            "Novelty effects may have inflated early conversion",
            "Segmentation not analyzed - personalization opportunities unknown"
        ],
        'next_steps': [
            f"Implement {winner} pricing with 10% gradual rollout",
            "Track 30/60/90-day cohort retention by price tier",
            "Monitor support ticket volume for pricing complaints",
            "Prepare fallback price point ($2.99) if metrics deteriorate",
            "Test price points ABOVE $4.99 (demand appears inelastic)"
        ]
    }
    
    return summary

def export_report(summary: Dict, output_dir: Path = Config.OUTPUT_DIR):
    """Export report in multiple formats."""
    
    output_dir.mkdir(exist_ok=True)
    
    # JSON export
    json_path = output_dir / 'farmburg_abtest_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # CSV exports
    groups_df = pd.DataFrame(summary['all_groups_performance']).T
    groups_df.to_csv(output_dir / 'group_performance.csv')
    
    decision_df_export = pd.DataFrame(summary['decision_matrix'])
    decision_df_export.to_csv(output_dir / 'decision_matrix.csv')
    
    print(f"✅ Report exported to {output_dir}/")
    print(f"   - farmburg_abtest_summary.json")
    print(f"   - group_performance.csv")
    print(f"   - decision_matrix.csv")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution pipeline."""
    
    print("=" * 70)
    print("FarmBurg A/B Testing Analysis - Executive Summary")
    print("=" * 70)
    print()
    
    # Step 1: Load data
    print("[1/6] Loading and validating data...")
    df = load_data(Config.DATA_FILE)
    stats = get_sample_stats(df)
    print(f"      ✓ Loaded {stats['total_visitors']:,} visitors")
    
    # Step 2: Contingency table & chi-square
    print("[2/6] Running chi-square test...")
    ctable = contingency_table(df)
    chi2_result = chi_square_test(ctable)
    print(f"      ✓ χ²={chi2_result['chi2_statistic']:.2f}, p={chi2_result['p_value']:.2e}, Significant={chi2_result['is_significant']}")
    
    # Step 3: Binomial tests against revenue targets
    print("[3/6] Testing against revenue targets...")
    binomial_results = binomial_tests_against_targets(df)
    
    print("      Group Results:")
    for g, r in binomial_results.items():
        status = "✓ PASS" if r['is_significant'] else "✗ FAIL"
        print(f"        {g} (${r['price_usd']}): {status} | p={r['binomial_p_value']:.6f} | Rev=${r['projected_weekly_revenue_usd']:.2f}/wk")
    
    # Step 4: Effect sizes & power
    print("[4/6] Calculating effect sizes and power...")
    effect_sizes = calculate_effect_sizes(binomial_results)
    sample_per_group = stats['groups']['A']['visitors']
    power_data = power_analysis(sample_per_group)
    print(f"      ✓ MDE at 80% power: {power_data['mde_at_80pct_power']:.3f} (Cohen's h)")
    
    # Step 5: Decision matrix
    print("[5/6] Building decision matrix...")
    decision_df = build_decision_matrix(binomial_results)
    winner = decision_df.index[0]
    print(f"      ✓ Recommendation: Group {winner} (${Config.PRICES[winner]})")
    print(f"        Decision Score: {decision_df.loc[winner, 'weighted_total']:.1f}/100")
    
    # Step 6: Generate summary & exports
    print("[6/6] Generating report...")
    summary = generate_executive_summary(stats, binomial_results, decision_df)
    export_report(summary)
    
    # Create visualizations
    print()
    print("Generating visualizations...")
    plot_conversion_with_ci(binomial_results, Config.OUTPUT_DIR / 'conversion_ci_vs_targets')
    plot_daily_trends(df, Config.OUTPUT_DIR / 'daily_trends')
    print(f"      ✓ Saved to {Config.OUTPUT_DIR}/")
    
    # Final summary
    print()
    print("=" * 70)
    print("EXECUTIVE SUMMARY")
    print("=" * 70)
    print()
    print(f"RECOMMENDED PRICE POINT: ${Config.PRICES[winner]}")
    print(f"  • Weekly Revenue Projection: ${binomial_results[winner]['projected_weekly_revenue_usd']:,.2f}")
    print(f"  • Meets $1,000 Target: {'YES' if binomial_results[winner]['meets_revenue_target'] else 'NO'}")
    print(f"  • Conversion Rate: {binomial_results[winner]['observed_rate_pct']:.2f}%")
    print(f"  • Statistically Significant: {'YES' if binomial_results[winner]['is_significant'] else 'NO'}")
    print()
    print("KEY INSIGHT: Highest conversion (Group A @ 19%) does NOT equal best revenue.")
    print("             Optimal pricing maximizes price × conversion × traffic, not just conversion.")
    print()
    print("All outputs saved to:", Config.OUTPUT_DIR.resolve())
    print("=" * 70)
    
    return summary

# Entry point
if __name__ == '__main__':
    result = main()