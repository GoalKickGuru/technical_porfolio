"""
Feature Engineering Techniques for Credit Risk
Domain ratios, scaling, binning, encoding, interactions, missing indicators.
"""
import numpy as np
import pandas as pd

def make_synthetic_credit(n=40, seed=42):
    rng = np.random.default_rng(seed)
    income = rng.normal(65000, 22000, n).clip(18000, 160000)
    debt = income * rng.uniform(0.15, 1.4, n)
    age = rng.integers(22, 68, n)
    employment = rng.choice(['employed', 'self-employed', 'unemployed'], n, p=[0.65, 0.25, 0.10])
    credit_history = rng.uniform(0.5, 25, n).round(1)
    credit_limit = rng.uniform(2000, 35000, n)
    revolving = credit_limit * rng.uniform(0.05, 1.3, n)
    dti = debt / income
    util = revolving / credit_limit
    logit = -3.5 + 2.8 * dti + 1.6 * util - 0.03 * age
    default = (rng.random(n) < 1/(1+np.exp(-logit))).astype(int)
    df = pd.DataFrame({
        'income': income.round(0), 'debt': debt.round(0), 'age': age,
        'employment': employment, 'credit_history_years': credit_history,
        'revolving_balance': revolving.round(0), 'credit_limit': credit_limit.round(0),
        'default': default
    })
    return df

def engineer_features(df):
    df = df.copy()
    df['dti'] = df['debt'] / df['income']
    df['utilization'] = df['revolving_balance'] / df['credit_limit'].fillna(df['credit_limit'].median())
    df['log_income'] = np.log1p(df['income'])
    df['dti_std'] = (df['dti'] - df['dti'].mean()) / df['dti'].std(ddof=0)
    df['income_bin'] = pd.qcut(df['income'], q=4, labels=['Q1','Q2','Q3','Q4'], duplicates='drop')
    df['high_util'] = (df['utilization'] > 0.7).astype(int)
    emp_means = df.groupby('employment')['default'].mean()
    df['emp_target_enc'] = df['employment'].map(emp_means)
    df['income_x_dti'] = df['income'] * df['dti']
    return df

if __name__ == "__main__":
    df = make_synthetic_credit()
    print("Raw shape:", df.shape, "Default rate: {:.1%}".format(df.default.mean()))
    eng = engineer_features(df)
    print("\nEngineered columns sample:")
    print(eng[['dti','utilization','log_income','dti_std','high_util','emp_target_enc']].head().round(3))
    print("\nDefault rate by income bin:")
    print(eng.groupby('income_bin', observed=True)['default'].mean().round(3))
