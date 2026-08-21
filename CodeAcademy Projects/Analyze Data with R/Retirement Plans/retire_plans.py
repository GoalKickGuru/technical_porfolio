import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD DATASETS OR GENERATE IN-MEMORY
# ==========================================
csv_path = 'retirement_plans_project/retirement_contributions.csv'

if os.path.exists(csv_path):
    df_contrib = pd.read_csv(csv_path)
else:
    # Generate dataset directly if CSV is missing
    np.random.seed(42)
    years = 30
    years_arr = np.arange(1, years + 1)
    salaries = 95000 * ((1 + 0.03) ** (years_arr - 1))
    
    df_contrib = pd.DataFrame({
        'Year': years_arr,
        'Age': 30 + years_arr - 1,
        'Annual_Salary': np.round(salaries, 2),
        'Employee_401k_Contrib': np.round(salaries * 0.10, 2),
        'Employer_401k_Match': np.round(salaries * 0.04, 2),
        'Roth_IRA_Contrib': np.round(np.minimum(7000 * ((1 + 0.025)**(years_arr-1)), salaries * 0.08), 2),
        'Taxable_Brokerage_Contrib': np.round(salaries * 0.05, 2),
        'Est_Market_Return': 0.07,
        'Est_Inflation': 0.025
    })

# ==========================================
# 2. COMPOUND GROWTH CALCULATIONS
# ==========================================
r = 0.07
N = len(df_contrib)
compound_factors = (1 + r) ** (N - df_contrib['Year'] + 1)

# Principal Accumulation
p_401k_total = (df_contrib['Employee_401k_Contrib'] + df_contrib['Employer_401k_Match']).sum()
p_roth = df_contrib['Roth_IRA_Contrib'].sum()
p_taxable = df_contrib['Taxable_Brokerage_Contrib'].sum()

# Future Value Compounding (Gross)
fv_401k = np.sum((df_contrib['Employee_401k_Contrib'] + df_contrib['Employer_401k_Match']) * compound_factors)
fv_roth = np.sum(df_contrib['Roth_IRA_Contrib'] * compound_factors)
fv_taxable = np.sum(df_contrib['Taxable_Brokerage_Contrib'] * compound_factors)

# ==========================================
# 3. TAX ADJUSTED RETIREMENT NET WEALTH
# ==========================================
ordinary_income_tax = 0.22
cap_gains_tax = 0.15

# Traditional 401(k): Ordinary income tax on full withdrawal
tax_401k = fv_401k * ordinary_income_tax
net_401k = fv_401k - tax_401k

# Roth IRA: 100% Tax-Free
tax_roth = 0.0
net_roth = fv_roth

# Taxable Brokerage: Capital gains tax applied strictly to growth
gains_taxable = max(0, fv_taxable - p_taxable)
tax_taxable = gains_taxable * cap_gains_tax
net_taxable = fv_taxable - tax_taxable

# 4% Safe Withdrawal Rates
swr_monthly_401k = (net_401k * 0.04) / 12
swr_monthly_roth = (net_roth * 0.04) / 12
swr_monthly_taxable = (net_taxable * 0.04) / 12

# Construct Summary Results Table
df_summary = pd.DataFrame({
    'Account_Type': ['Traditional 401(k) + Match', 'Roth IRA', 'Taxable Brokerage'],
    'Total_Principal': [p_401k_total, p_roth, p_taxable],
    'Gross_FV_7pct': [fv_401k, fv_roth, fv_taxable],
    'Tax_Liability': [tax_401k, tax_roth, tax_taxable],
    'Net_After_Tax': [net_401k, net_roth, net_taxable],
    'Monthly_SWR_4pct': [swr_monthly_401k, swr_monthly_roth, swr_monthly_taxable]
})

print("==========================================================================")
print("             RETIREMENT PLAN ALTERNATIVES ANALYSIS SUMMARY                ")
print("==========================================================================")
print(df_summary.to_string(index=False, formatters={
    'Total_Principal': '${:,.2f}'.format,
    'Gross_FV_7pct': '${:,.2f}'.format,
    'Tax_Liability': '${:,.2f}'.format,
    'Net_After_Tax': '${:,.2f}'.format,
    'Monthly_SWR_4pct': '${:,.2f}'.format
}))

# ==========================================
# 4. ANNUAL GROWTH TRAJECTORY SIMULATION
# ==========================================
bal_401k, bal_roth, bal_taxable = [], [], []
curr_401k, curr_roth, curr_taxable = 0.0, 0.0, 0.0

for idx, row in df_contrib.iterrows():
    c_401k = row['Employee_401k_Contrib'] + row['Employer_401k_Match']
    c_roth = row['Roth_IRA_Contrib']
    c_taxable = row['Taxable_Brokerage_Contrib']
    
    curr_401k = (curr_401k + c_401k) * (1 + r)
    curr_roth = (curr_roth + c_roth) * (1 + r)
    curr_taxable = (curr_taxable + c_taxable) * (1 + r)
    
    bal_401k.append(curr_401k)
    bal_roth.append(curr_roth)
    bal_taxable.append(curr_taxable)

df_contrib['Bal_401k_Gross'] = bal_401k
df_contrib['Bal_Roth_Gross'] = bal_roth
df_contrib['Bal_Taxable_Gross'] = bal_taxable

# ==========================================
# 5. VISUALIZATION GENERATION
# ==========================================
plt.figure(figsize=(12, 6))
plt.plot(df_contrib['Year'], df_contrib['Bal_401k_Gross'] * (1 - ordinary_income_tax), 
         label='401(k) + Match (After 22% Tax)', color='#0284c7', linewidth=2.5)
plt.plot(df_contrib['Year'], df_contrib['Bal_Roth_Gross'], 
         label='Roth IRA (100% Tax-Free)', color='#10b981', linewidth=2.5)
plt.plot(df_contrib['Year'], df_contrib['Bal_Taxable_Gross'] - ((df_contrib['Bal_Taxable_Gross'] - df_contrib['Taxable_Brokerage_Contrib'].cumsum()) * cap_gains_tax), 
         label='Taxable Brokerage (After 15% Cap Gains)', color='#f59e0b', linewidth=2.5)

plt.title('30-Year Net Wealth Trajectory Across Retirement Accounts (After-Tax)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Contribution Year', fontsize=11)
plt.ylabel('Net Portfolio Value ($)', fontsize=11)
plt.gca().yaxis.set_major_formatter('${x:,.0f}')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=10, loc='upper left')
plt.tight_layout()
plt.show()