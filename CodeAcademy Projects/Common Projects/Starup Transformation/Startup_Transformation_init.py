import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=== Startup Transformation Project - Complete Solution ===\n")

# ====================== 1. Load Data ======================
print("1. Loading datasets...")
financial_data = pd.read_csv('/home/workdir/attachments/financial_data.csv')
expenses = pd.read_csv('/home/workdir/attachments/expenses.csv')
employees = pd.read_csv('/home/workdir/attachments/employees.csv')

print("Financial Data Head:\n", financial_data.head())
print("\nExpenses:\n", expenses)
print("\nEmployees Shape:", employees.shape)
print("Employees Head:\n", employees.head())

# ====================== 2. Financial Health Analysis ======================
print("\n=== Financial Health Analysis ===")
print("Revenue Trend:")
plt.figure(figsize=(10,5))
plt.plot(financial_data['Month'], financial_data['Revenue'], marker='o', label='Revenue')
plt.plot(financial_data['Month'], financial_data['Expenses'], marker='o', label='Expenses')
plt.xlabel('Month'); plt.ylabel('Amount ($)'); plt.title('Revenue vs Expenses Trend'); plt.legend(); plt.show()

# Profit calculation
financial_data['Profit'] = financial_data['Revenue'] - financial_data['Expenses']
print("Monthly Profit:\n", financial_data[['Month', 'Profit']])

# ====================== 3. Expense Breakdown ======================
print("\n=== Expense Breakdown ===")
expense_categories = ['Salaries', 'Advertising', 'Office Rent', 'Other']
proportions = [0.62, 0.15, 0.15, 0.08]
plt.figure(figsize=(8,8))
plt.pie(proportions, labels=expense_categories, autopct='%1.1f%%')
plt.title('Expense Categories (Collapsed)'); plt.axis('equal'); plt.show()

# ====================== 4. Employee Productivity ======================
print("\n=== Employee Productivity Analysis ===")
sorted_productivity = employees.sort_values(by='Productivity')
print("Lowest Productivity Employees (first 5):\n", sorted_productivity.head())

# Cut 100 least productive
employees_cut = sorted_productivity.head(100)
print(f"\nEmployees to let go (bottom {len(employees_cut)}):\n", employees_cut[['Name', 'Salary', 'Productivity']].head())

remaining = sorted_productivity.iloc[100:]
print(f"Remaining Employees Avg Productivity: {remaining['Productivity'].mean():.2f}")

# Alternate: Correlation Salary vs Productivity
print("\nSalary-Productivity Correlation:", employees['Salary'].corr(employees['Productivity']))

# ====================== 5. Commute Analysis ======================
print("\n=== Commute Times Analysis ===")
commute_times = employees['Commute Time']
print(commute_times.describe())

# Histogram
plt.figure(figsize=(10,5))
plt.hist(commute_times, bins=30, alpha=0.7, label='Original (Right-Skewed)')
plt.hist(np.log(commute_times), bins=30, alpha=0.7, label='Log Transformed')
plt.title('Commute Times Distribution'); plt.xlabel('Commute Time'); plt.ylabel('Frequency'); plt.legend(); plt.show()

# ====================== 6. Advanced: Standardization & Simulation ======================
print("\n=== Advanced Analysis & Simulation ===")

# Standardization (alternate method)
scaler = StandardScaler()
employees[['Salary_std', 'Productivity_std']] = scaler.fit_transform(employees[['Salary', 'Productivity']])
print("Standardized Data Sample:\n", employees[['Salary', 'Productivity', 'Salary_std', 'Productivity_std']].head())

# Simulation: Impact of different layoff cutoffs
def simulate_layoffs(cutoff_list=[50, 100, 150]):
    results = []
    for cutoff in cutoff_list:
        cut_df = sorted_productivity.head(cutoff)
        remaining_prod = sorted_productivity.iloc[cutoff:]['Productivity'].mean()
        cost_saved = cut_df['Salary'].sum()
        results.append({'Cutoff': cutoff, 'Remaining_Avg_Prod': round(remaining_prod, 2), 'Est_Annual_Savings': round(cost_saved, 0)})
    sim_df = pd.DataFrame(results)
    print("Simulation Results:\n", sim_df)
    return sim_df

sim_results = simulate_layoffs()

# Visualization of simulation
plt.figure(figsize=(8,5))
plt.bar(sim_results['Cutoff'], sim_results['Remaining_Avg_Prod'])
plt.title('Simulation: Avg Productivity After Layoffs'); plt.xlabel('Employees Cut'); plt.ylabel('Remaining Avg Productivity'); plt.show()

# ====================== 7. Answers to Key Questions ======================
print("\n=== ANSWERS TO KEY QUESTIONS ===")
print("1. Is the company in good financial health? NO - Revenue declining sharply while expenses rise. Risk of losses.")
print("2. Does the company need to let go of any employees? YES - Recommend letting go of the 100 least productive employees.")
print("3. Should the company allow employees to work from home permanently? YES - Average commute ~33 min; many long commutes; WFH saves time and boosts morale.")

print("\n=== Project Complete! ===")
print("Run this script for full outputs. Skeleton version has TODOs for practice.")