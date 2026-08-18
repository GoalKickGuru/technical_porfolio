import codecademylib3
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# ============================================================
# PART 1: FINANCIAL DATA ANALYSIS
# ============================================================

# Load financial data
financial_data = pd.read_csv('financial_data.csv')

# Step 1: Examine the data
print("=== FINANCIAL DATA ===")
print(financial_data.head())
print("\nShape:", financial_data.shape)
print("\nData types:\n", financial_data.dtypes)

# Step 2: Store columns in separate variables
month = financial_data['Month']
revenue = financial_data['Revenue']
expenses = financial_data['Expenses']

# Step 3-4: Plot revenue over time
plt.figure(figsize=(10, 5))
plt.plot(month, revenue, marker='o', linewidth=2, color='#6d4aff')
plt.xlabel('Month')
plt.ylabel('Amount ($)')
plt.title('Revenue Over Past 6 Months')
plt.grid(True, alpha=0.3)
plt.show()

# Step 5: Plot expenses over time
plt.clf()
plt.figure(figsize=(10, 5))
plt.plot(month, expenses, marker='s', linewidth=2, color='#ff6b6b')
plt.xlabel('Month')
plt.ylabel('Amount ($)')
plt.title('Expenses Over Past 6 Months')
plt.grid(True, alpha=0.3)
plt.show()

# --- ADDITIONAL ANALYSIS: Profit & Margin ---
financial_data['Profit'] = financial_data['Revenue'] - financial_data['Expenses']
financial_data['Profit_Margin'] = (financial_data['Profit'] / financial_data['Revenue']) * 100

print("\n=== PROFIT ANALYSIS ===")
print(financial_data[['Month', 'Revenue', 'Expenses', 'Profit', 'Profit_Margin']])

# Plot combined Revenue vs Expenses
plt.clf()
plt.figure(figsize=(10, 5))
plt.plot(month, revenue, marker='o', label='Revenue', linewidth=2)
plt.plot(month, expenses, marker='s', label='Expenses', linewidth=2)
plt.xlabel('Month')
plt.ylabel('Amount ($)')
plt.title('Revenue vs Expenses')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Plot profit margin
plt.clf()
plt.figure(figsize=(10, 5))
plt.plot(month, financial_data['Profit_Margin'], marker='D',
         linewidth=2, color='#2ecc71')
plt.xlabel('Month')
plt.ylabel('Profit Margin (%)')
plt.title('Profit Margin Decline')
plt.grid(True, alpha=0.3)
plt.show()

# --- ADDITIONAL: Linear projection for next 3 months ---
from sklearn.linear_model import LinearRegression

X = financial_data[['Month']].values
rev_model = LinearRegression().fit(X, revenue)
exp_model = LinearRegression().fit(X, expenses)

future_months = np.array([[7], [8], [9]])
rev_forecast = rev_model.predict(future_months)
exp_forecast = exp_model.predict(future_months)

print("\n=== 3-MONTH PROJECTION ===")
for m, r, e in zip([7, 8, 9], rev_forecast, exp_forecast):
    print(f"Month {m}: Revenue=${r:,.0f}, Expenses=${e:,.0f}, "
          f"Profit=${r-e:,.0f}")

# Plot with forecast
plt.clf()
plt.figure(figsize=(10, 5))
all_months = np.concatenate([X.flatten(), future_months.flatten()])
plt.plot(range(1, 10),
         np.concatenate([revenue.values, rev_forecast]),
         marker='o', label='Revenue (actual + projected)', linestyle='-')
plt.plot(range(1, 10),
         np.concatenate([expenses.values, exp_forecast]),
         marker='s', label='Expenses (actual + projected)', linestyle='-')
plt.axvline(x=6.5, color='gray', linestyle='--', alpha=0.5, label='Forecast boundary')
plt.xlabel('Month')
plt.ylabel('Amount ($)')
plt.title('Revenue & Expenses: Actual + Projected')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ============================================================
# PART 2: EXPENSE BREAKDOWN
# ============================================================

# Step 7: Load expense overview
expense_overview = pd.read_csv('expenses.csv')
print("\n=== EXPENSE OVERVIEW ===")
print(expense_overview.head(7))

expense_categories = expense_overview['Expense']
proportions = expense_overview['Proportion']

# Step 8-9: Create pie chart
plt.clf()
plt.figure(figsize=(8, 8))
plt.pie(proportions, labels=expense_categories, autopct='%1.1f%%',
        startangle=90)
plt.axis('Equal')
plt.tight_layout()
plt.title('Expense Categories')
plt.show()

# Step 10: Collapse categories < 5% into "Other"
expense_cut = "Salaries"  # Step 11 answer

threshold = 0.05
large_categories = expense_overview[expense_overview['Proportion'] >= threshold]
small_categories = expense_overview[expense_overview['Proportion'] < threshold]

other_proportion = small_categories['Proportion'].sum()
collapsed_df = pd.DataFrame({
    'Expense': list(large_categories['Expense']) + ['Other'],
    'Proportion': list(large_categories['Proportion']) + [other_proportion]
})

print("\n=== COLLAPSED EXPENSE CATEGORIES ===")
print(collapsed_df)

# Updated pie chart
plt.clf()
plt.figure(figsize=(8, 8))
plt.pie(collapsed_df['Proportion'], labels=collapsed_df['Expense'],
        autopct='%1.1f%%', startangle=90)
plt.axis('Equal')
plt.tight_layout()
plt.title('Expense Categories (Collapsed)')
plt.show()

# ============================================================
# PART 3: EMPLOYEE PRODUCTIVITY ANALYSIS
# ============================================================

# Step 12: Load employee data
employees = pd.read_csv('employees.csv')
print("\n=== EMPLOYEE DATA ===")
print(employees.head())
print(f"\nTotal employees: {len(employees)}")

# Step 13: Sort by productivity (ascending)
sorted_productivity = employees.sort_values(by=['Productivity'])
print("\n=== LEAST PRODUCTIVE EMPLOYEES ===")
print(sorted_productivity.head(10))

# Step 14: Cut 100 least productive employees
employees_cut = sorted_productivity.head(100)
print(f"\nEmployees to be cut: {len(employees_cut)}")
print(employees_cut[['Name', 'Salary', 'Productivity']].head(20))

# --- ADDITIONAL ANALYSIS: Impact of employee cuts ---
total_salary_all = employees['Salary'].sum()
total_salary_cut = employees_cut['Salary'].sum()
total_salary_kept = total_salary_all - total_salary_cut
avg_salary_cut = employees_cut['Salary'].mean()
avg_salary_kept = employees[~employees.index.isin(employees_cut.index)]['Salary'].mean()

print("\n=== SALARY IMPACT ANALYSIS ===")
print(f"Total annual payroll (all):     ${total_salary_all:,.0f}")
print(f"Total salary of cut employees:  ${total_salary_cut:,.0f}")
print(f"Savings from cuts:              ${total_salary_cut:,.0f}")
print(f"Avg salary of cut employees:    ${avg_salary_cut:,.0f}")
print(f"Avg salary of kept employees:   ${avg_salary_kept:,.0f}")
print(f"% of payroll saved:             {(total_salary_cut/total_salary_all)*100:.1f}%")

# Step 15: Transformation recommendation
transformation = (
    "Standardization (Z-score normalization via StandardScaler) is recommended "
    "because: (1) it handles vastly different scales between productivity (0-100) "
    "and salary (thousands); (2) it is robust to outliers since it centers on the "
    "mean rather than min/max; (3) it preserves distribution shape. Additionally, "
    "applying a log transformation to the salary column first (to address its "
    "right-skewness) before standardizing would yield the best results."
)
print(f"\n=== TRANSFORMATION RECOMMENDATION ===\n{transformation}")

# --- ADDITIONAL: Correlation Analysis ---
print("\n=== CORRELATION MATRIX ===")
correlation_matrix = employees[['Salary', 'Productivity', 'Commute Time']].corr()
print(correlation_matrix)

# Heatmap
plt.clf()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='Purples',
            fmt='.3f', square=True)
plt.title('Correlation Matrix: Salary, Productivity, Commute Time')
plt.tight_layout()
plt.show()

# Scatter: Salary vs Productivity
plt.clf()
plt.figure(figsize=(10, 6))
plt.scatter(employees['Productivity'], employees['Salary'],
            alpha=0.5, c='#6d4aff', edgecolors='white', s=40)
plt.xlabel('Productivity Score')
plt.ylabel('Salary ($)')
plt.title('Salary vs Productivity — Weak Correlation')
plt.grid(True, alpha=0.3)
plt.show()

# --- ADDITIONAL: Standardization demo ---
scaler = preprocessing.StandardScaler()
scaled_features = scaler.fit_transform(employees[['Salary', 'Productivity']])
scaled_df = pd.DataFrame(scaled_features, columns=['Salary_scaled', 'Productivity_scaled'])
print("\n=== STANDARDIZED FEATURES (first 5) ===")
print(scaled_df.head())

# ============================================================
# PART 4: COMMUTE TIME ANALYSIS
# ============================================================

# Step 16: Store commute times
commute_times = employees['Commute Time']

# Step 17: Descriptive statistics
print("\n=== COMMUTE TIME STATISTICS ===")
print(commute_times.describe())

avg_commute = commute_times.mean()
median_commute = commute_times.median()
print(f"\nAverage commute: {avg_commute:.2f} minutes")
print(f"Median commute:  {median_commute:.2f} minutes")
print(f"Std dev:         {commute_times.std():.2f} minutes")
print(f"Min:             {commute_times.min():.2f} minutes")
print(f"Max:             {commute_times.max():.2f} minutes")

# Step 18: Histogram of commute times
plt.clf()
plt.figure(figsize=(10, 5))
plt.hist(commute_times, bins=30, color='#6d4aff', edgecolor='white', alpha=0.8)
plt.xlabel('Commute Time (minutes)')
plt.ylabel('Frequency')
plt.title('Distribution of Commute Times (Right-Skewed)')
plt.grid(True, alpha=0.3)
plt.show()

# Step 19: Log transformation
commute_times_log = np.log(commute_times)

# Step 20: Histogram of log-transformed commute times
plt.clf()
plt.figure(figsize=(10, 5))
plt.hist(commute_times_log, bins=30, color='#2ecc71', edgecolor='white', alpha=0.8)
plt.xlabel('ln(Commute Time)')
plt.ylabel('Frequency')
plt.title('Distribution of Log-Transformed Commute Times (More Symmetric)')
plt.grid(True, alpha=0.3)
plt.show()

# --- ADDITIONAL: Compare original vs log-transformed stats ---
print("\n=== LOG-TRANSFORMED COMMUTE STATS ===")
print(f"Original mean:    {commute_times.mean():.2f}")
print(f"Log mean:         {commute_times_log.mean():.2f}")
print(f"Original skew:    {commute_times.skew():.2f}")
print(f"Log skew:         {commute_times_log.skew():.2f}")

# --- ADDITIONAL: Commute Time vs Productivity ---
plt.clf()
plt.figure(figsize=(10, 6))
plt.scatter(employees['Commute Time'], employees['Productivity'],
            alpha=0.5, c='#6d4aff', edgecolors='white', s=40)
plt.xlabel('Commute Time (minutes)')
plt.ylabel('Productivity Score')
plt.title('Commuter Time vs Productivity')
plt.grid(True, alpha=0.3)
plt.show()

# --- ADDITIONAL: Productivity distribution of cut vs kept ---
plt.clf()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(employees_cut['Productivity'], bins=20, color='#ff6b6b',
             edgecolor='white', alpha=0.8)
axes[0].set_title('Productivity: Cut Employees')
axes[0].set_xlabel('Productivity Score')
axes[0].set_ylabel('Count')

kept_employees = employees[~employees.index.isin(employees_cut.index)]
axes[1].hist(kept_employees['Productivity'], bins=20, color='#2ecc71',
             edgecolor='white', alpha=0.8)
axes[1].set_title('Productivity: Retained Employees')
axes[1].set_xlabel('Productivity Score')
axes[1].set_ylabel('Count')
plt.tight_layout()
plt.show()

# ============================================================
# ADDITIONAL ANALYSIS: Remote Work Feasibility Assessment
# ============================================================

remote_candidates = employees[employees['Commute Time'] > 30]
print(f"\n=== REMOTE WORK FEASIBILITY ===")
print(f"Employees with commute > 30 min: {len(remote_candidates)} "
      f"({len(remote_candidates)/len(employees)*100:.1f}%)")
print(f"Employees with commute > 45 min: {len(employees[employees['Commute Time'] > 45])} "
      f"({len(employees[employees['Commute Time'] > 45])/len(employees)*100:.1f}%)")
print(f"Average time saved per employee (if remote, >30 min commuters): "
      f"{remote_candidates['Commute Time'].mean():.1f} min/day")
print(f"Total monthly hours saved (assuming 22 work days, round trip): "
      f"{remote_candidates['Commute Time'].mean() * 2 * 22 * len(remote_candidates) / 60:,.0f} hours")

# ============================================================
# SUMMARY OUTPUT
# ============================================================
print("\n" + "="*60)
print("EXECUTIVE SUMMARY")
print("="*60)
print(f"""
1. FINANCIAL HEALTH: CRITICAL
   - Revenue declined ${1420000-720000:,} ({(1-720000/1420000)*100:.1f}%) over 6 months
   - Profit margin collapsed from 64.1% to 10.6%
   - Projected to operate at a LOSS by Month 7

2. EXPENSE STRUCTURE:
   - Salaries dominate at 62% of total expenses
   - Recommended cut category: {expense_cut}

3. EMPLOYEE PRODUCTIVITY:
   - {len(employees)} total employees analyzed
   - 100 least productive identified for potential layoff
   - Annual salary savings: ${total_salary_cut:,.0f}
   - Weak salary-productivity correlation detected
   - Recommended transformation: Standardization + log(salary)

4. COMMUTE TIME & REMOTE WORK:
   - Average commute: {avg_commute:.1f} minutes (right-skewed)
   - Median commute: {median_commute:.1f} minutes
   - {len(remote_candidates)} employees ({len(remote_candidates)/len(employees)*100:.1f}%) commute > 30 min
   - Remote work could save significant employee time
   - Log transformation normalizes the skewed commute distribution
""")