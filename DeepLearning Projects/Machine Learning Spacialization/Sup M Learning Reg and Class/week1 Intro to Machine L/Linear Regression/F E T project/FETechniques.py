# ============================================================
# Feature Engineering Techniques – Complete Solution
# Credit Risk / Banking context
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ------------------------------------------------------------
# 1. Create synthetic credit dataset
# ------------------------------------------------------------
n = 40
income = rng.normal(65000, 22000, n).clip(18000, 160000)
debt = income * rng.uniform(0.15, 1.4, n)
age = rng.integers(22, 68, n)
employment = rng.choice(['employed', 'self-employed', 'unemployed'], n, p=[0.65, 0.25, 0.10])
credit_history = rng.uniform(0.5, 25, n).round(1)
credit_limit = rng.uniform(2000, 35000, n)
revolving = credit_limit * rng.uniform(0.05, 1.3, n)

# Generate default labels driven by DTI & utilisation
dti_raw = debt / income
util_raw = revolving / credit_limit
logit = -3.5 + 2.8 * dti_raw + 1.6 * util_raw - 0.03 * age
prob = 1 / (1 + np.exp(-logit))
default = (rng.random(n) < prob).astype(int)

df = pd.DataFrame({
    'income': income.round(0),
    'debt': debt.round(0),
    'age': age,
    'employment': employment,
    'credit_history_years': credit_history,
    'revolving_balance': revolving.round(0),
    'credit_limit': credit_limit.round(0),
    'default': default
})

# Inject a few missing values
df.loc[rng.choice(n, 3, replace=False), 'credit_limit'] = np.nan
df.loc[rng.choice(n, 2, replace=False), 'employment'] = np.nan

print("Raw data shape:", df.shape)
print("Default rate: {:.1%}".format(df['default'].mean()))
print(df.head())

# ------------------------------------------------------------
# 2. Domain-derived features
# ------------------------------------------------------------
df['dti'] = df['debt'] / df['income']
df['utilization'] = df['revolving_balance'] / df['credit_limit']
df['log_income'] = np.log1p(df['income'])
df['credit_age'] = df['credit_history_years']

print("\nDomain features:")
print(df[['income', 'debt', 'dti', 'utilization', 'log_income']].describe().round(3))

# ------------------------------------------------------------
# 3. Numerical transforms – Scaling & Binning
# ------------------------------------------------------------
def standard_scale(s):
    return (s - s.mean()) / s.std(ddof=0)

def minmax_scale(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-9)

df['dti_std'] = standard_scale(df['dti'].fillna(df['dti'].median()))
df['dti_mm']  = minmax_scale(df['dti'].fillna(df['dti'].median()))
df['age_std'] = standard_scale(df['age'])

# Quantile bins and custom bins
df['income_bin'] = pd.qcut(df['income'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
df['dti_bin'] = pd.cut(df['dti'],
                       bins=[0, 0.3, 0.5, 0.8, np.inf],
                       labels=['low', 'med', 'high', 'very_high'])

print("\nDefault rate by DTI bin:")
print(df.groupby('dti_bin', observed=True)['default'].agg(['count', 'mean']).round(3))

# ------------------------------------------------------------
# 4. Categorical encoding
# ------------------------------------------------------------
# One-hot
emp_dummies = pd.get_dummies(df['employment'], prefix='emp', dummy_na=True)
df = pd.concat([df, emp_dummies], axis=1)

# Simple target / mean encoding (educational – watch leakage in real CV)
emp_means = df.groupby('employment')['default'].mean()
df['emp_target_enc'] = df['employment'].map(emp_means)

# Ordinal
ord_map = {'unemployed': 0, 'self-employed': 1, 'employed': 2}
df['emp_ordinal'] = df['employment'].map(ord_map)

print("\nTarget encoding map:")
print(emp_means.round(3))

# ------------------------------------------------------------
# 5. Interaction & polynomial features
# ------------------------------------------------------------
df['income_x_dti'] = df['income'] * df['dti']
df['age_x_hist']   = df['age'] * df['credit_history_years']
df['util_sq']      = df['utilization'].fillna(0) ** 2

# ------------------------------------------------------------
# 6. Missing-value indicators + imputation
# ------------------------------------------------------------
df['credit_limit_missing'] = df['credit_limit'].isna().astype(int)
df['employment_missing']   = df['employment'].isna().astype(int)

df['credit_limit_imp'] = df['credit_limit'].fillna(df['credit_limit'].median())
df['employment_imp']   = df['employment'].fillna(df['employment'].mode()[0])

# Recompute utilisation with imputed limit
df['utilization'] = df['revolving_balance'] / df['credit_limit_imp']

print("\nMissing indicators:", df[['credit_limit_missing', 'employment_missing']].sum().to_dict())

# ------------------------------------------------------------
# 7. Assemble final feature matrix
# ------------------------------------------------------------
feature_cols = [
    'log_income', 'dti_std', 'age_std', 'credit_history_years',
    'utilization', 'util_sq', 'income_x_dti', 'age_x_hist',
    'emp_target_enc', 'emp_ordinal',
    'credit_limit_missing', 'employment_missing'
]
# add one-hot columns
feature_cols += [c for c in df.columns if c.startswith('emp_') and c not in feature_cols]
feature_cols = [c for c in feature_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

X = df[feature_cols].fillna(0)
y = df['default']

print("\nFeature matrix shape:", X.shape)
print("Top correlations with default:")
print(X.corrwith(y).abs().sort_values(ascending=False).head(8).round(3))

# ------------------------------------------------------------
# 8. More practice results
# ------------------------------------------------------------
print("\nDefault rate by income quartile:")
print(df.groupby('income_bin', observed=True)['default'].mean().round(3))

df['high_util'] = (df['utilization'] > 0.7).astype(int)
print("\nHigh-utilisation flag vs default:")
print(df.groupby('high_util')['default'].agg(['count', 'mean']).round(3))

# ------------------------------------------------------------
# 9. Simulation section (change these parameters)
# ------------------------------------------------------------
n_bins        = 4
scale_method  = 'standard'   # 'standard' or 'minmax'
include_inter = True

sim = df[['income', 'debt', 'age', 'default']].copy()
sim['dti'] = sim['debt'] / sim['income']

if scale_method == 'standard':
    sim['dti_scaled'] = (sim['dti'] - sim['dti'].mean()) / sim['dti'].std(ddof=0)
else:
    sim['dti_scaled'] = (sim['dti'] - sim['dti'].min()) / (sim['dti'].max() - sim['dti'].min() + 1e-9)

sim['income_bin'] = pd.qcut(sim['income'], q=n_bins, duplicates='drop')
if include_inter:
    sim['income_x_dti'] = sim['income'] * sim['dti']

print(f"\nSimulation → bins={n_bins}, scale={scale_method}, interactions={include_inter}")
print(sim.groupby('income_bin', observed=True)['default'].agg(['count', 'mean']).round(3))