# ============================================================
# ALTERNATE: Simplified, production-ready version
# Using a class-based approach with plotly for interactive charts
# ============================================================

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression


class StartupAnalysis:
    """Post-pandemic startup data analysis pipeline."""

    def __init__(self, fin_path, exp_path, emp_path):
        self.financial_data = pd.read_csv(fin_path)
        self.expense_overview = pd.read_csv(exp_path)
        self.employees = pd.read_csv(emp_path)
        self.results = {}

    def analyze_finances(self):
        """Compute profit trends and project future months."""
        df = self.financial_data.copy()
        df['Profit'] = df['Revenue'] - df['Expenses']
        df['Margin_%'] = (df['Profit'] / df['Revenue']) * 100

        # Linear regression forecast
        X = df[['Month']].values
        rev_pred = LinearRegression().fit(X, df['Revenue']).predict([[7],[8],[9]])
        exp_pred = LinearRegression().fit(X, df['Expenses']).predict([[7],[8],[9]])

        # Find break-even month
        break_even = None
        for m, r, e in zip([7,8,9], rev_pred, exp_pred):
            if r <= e:
                break_even = m
                break

        self.results['finance'] = {
            'profit_margins': dict(zip(df['Month'], df['Margin_%'].round(1))),
            'projected_loss_month': break_even,
            'revenue_decline_pct': round((1 - df['Revenue'].iloc[-1] /
                                          df['Revenue'].iloc[0]) * 100, 1),
            'total_profit_6mo': int(df['Profit'].sum()),
        }
        return df

    def analyze_expenses(self, threshold=0.05):
        """Collapse small expense categories into 'Other'."""
        df = self.expense_overview.copy()
        large = df[df['Proportion'] >= threshold].copy()
        small_sum = df[df['Proportion'] < threshold]['Proportion'].sum()

        collapsed = pd.concat([
            large,
            pd.DataFrame([{'Expense': 'Other', 'Proportion': small_sum}])
        ], ignore_index=True)

        largest = df.loc[df['Proportion'].idxmax(), 'Expense']
        self.results['expenses'] = {
            'largest_category': largest,
            'largest_pct': float(df['Proportion'].max() * 100),
            'collapsed_categories': collapsed.to_dict('records'),
        }
        return collapsed

    def analyze_employees(self, n_cut=100):
        """Sort by productivity, identify employees to cut."""
        df = self.employees.copy()
        sorted_df = df.sort_values('Productivity').reset_index(drop=True)
        cut = sorted_df.head(n_cut)
        kept = sorted_df.iloc[n_cut:]

        corr = df[['Salary', 'Productivity', 'Commute Time']].corr()

        self.results['employees'] = {
            'total_employees': len(df),
            'employees_cut': n_cut,
            'salary_savings': int(cut['Salary'].sum()),
            'avg_productivity_cut': round(float(cut['Productivity'].mean()), 2),
            'avg_productivity_kept': round(float(kept['Productivity'].mean()), 2),
            'corr_salary_productivity': round(float(corr.loc['Salary','Productivity']), 3),
            'recommendation': 'Standardization + log(salary)',
        }
        return cut, kept, corr

    def analyze_commutes(self):
        """Analyze commute time distribution and log transformation."""
        ct = self.employees['Commute Time']
        ct_log = np.log(ct)
        remote_eligible = (ct > 30).sum()

        self.results['commute'] = {
            'mean': round(float(ct.mean()), 2),
            'median': round(float(ct.median()), 2),
            'std': round(float(ct.std()), 2),
            'min': round(float(ct.min()), 2),
            'max': round(float(ct.max()), 2),
            'skew_original': round(float(ct.skew()), 2),
            'skew_log': round(float(ct_log.skew()), 2),
            'remote_eligible_gt_30min': int(remote_eligible),
            'remote_eligible_pct': round(remote_eligible / len(ct) * 100, 1),
        }

    def run_all(self):
        """Execute full analysis pipeline."""
        self.analyze_finances()
        self.analyze_expenses()
        self.analyze_employees()
        self.analyze_commutes()
        return self.results


# Run the analysis
analysis = StartupAnalysis('financial_data.csv', 'expenses.csv', 'employees.csv')
results = analysis.run_all()

import json
print(json.dumps(results, indent=2))