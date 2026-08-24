#!/usr/bin/env python
# coding: utf-8

# # Lab: Advertising Budget Optimization with Linear Regression
# 
# <figure>
#  <img src="./images/advertising_example.png"   style="width:600px;height:250px;">
# </figure>

# ## Goals
# In this lab you will:
# - Understand why linear regression matters for business decision-making
# - Implement the model $f_{w,b}$ for predicting sales from advertising spend
# - Learn how to calibrate your model to maximize ROI (Return on Investment)
# - Explore the consequences of over/under-estimating customer response

# ## Why This Matters: The Business Context
# Companies spend millions on advertising yearly. The critical question is:
# **"How much should we spend on advertising to achieve our sales goals?"**
# 
# Without data-driven insights, businesses risk:
# - **Overspending:** Wasting money on ads that don't convert
# - **Underspending:** Missing sales opportunities by not reaching enough customers
# - **Wrong channel allocation:** Putting money in ineffective marketing channels
# 
# Linear regression provides a mathematical framework to quantify the relationship
# between advertising investment and revenue outcome, enabling smarter budget decisions.

# ## Notation Summary
# 
# |Notation|Description|Python Variable|
# |:------|:----------|:--------------|
# | $m$ | Number of historical campaigns | `m`|
# | $x^{(i)}$ | Ad spend for campaign $i$ (in $1000s) | `x_train[i]`|
# | $y^{(i)}$ | Sales revenue for campaign $i$ (in $1000s) | `y_train[i]`|
# | $w$ | Revenue generated per $1000 ad spend (ROI slope) | `w`|
# | $b$ | Baseline sales without advertising | `b`|
# | $f_{w,b}(x)$ | Predicted revenue for spend $x$ | `f_wb`|

# ## Tools
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')


# # Problem Statement
# <img align="left" src="./images/business_problem.png"    style=" width:380px; padding: 10px; " /> 
# 
# Your company runs quarterly advertising campaigns and wants to optimize next quarter's budget.
# You have historical data from 5 campaigns showing ad spend and resulting sales revenue.
# 
# | Campaign | Ad Spend (1000s $) | Sales Revenue (1000s $) |
# | -------- | ------------------ | ----------------------- |
# | 1 | 2.0 | 12.5 |
# | 2 | 4.0 | 18.3 |
# | 3 | 6.0 | 24.7 |
# | 4 | 8.0 | 30.1 |
# | 5 | 10.0 | 35.8 |
# 
# **Business Question:** If we plan to spend $7000 on advertising next quarter,
# what sales revenue can we expect?
# 
# **Strategic Implications:**
# - If the forecast is too optimistic, you might miss revenue targets
# - If too pessimistic, you might allocate budget elsewhere unnecessarily
# - Accurate models directly impact financial planning and growth strategy

# Please run the following code cell to create your `x_train` and `y_train` variables.

# In[ ]:


# x_train: Historical advertising spend (in thousands of dollars)
# y_train: Corresponding sales revenue (in thousands of dollars)
# Each pair represents one completed campaign's outcome
x_train = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
y_train = np.array([12.5, 18.3, 24.7, 30.1, 35.8])
print(f"x_train (Ad Spend): {x_train}")
print(f"y_train (Sales Revenue): {y_train}")
print(f"\nUnit note: All values are in thousands of dollars")


# ### Understanding Training Examples
# 
# Each training example $(x^{(i)}, y^{(i)})$ represents a completed campaign:
# - We **observed** the outcome after spending $x^{(i)}$ on ads
# - We want to **learn the pattern** so we can predict future outcomes
# - This is supervised learning: we have both inputs (spend) and outputs (revenue)

# In[ ]:


# m = total number of historical campaigns in our dataset
m = x_train.shape[0]
print(f"Total historical campaigns available: {m}")

# Let's examine individual campaigns
i = 2  # Looking at 3rd campaign (index starts at 0)
x_i = x_train[i]
y_i = y_train[i]
print(f"\nCampaign {i+1}: Spent ${x_i:.0f}k → Generated ${y_i:.0f}k revenue")
print(f"ROI for this campaign: {(y_i/x_i):.1f}x return on ad spend")


# ### Calculating ROI Across All Campaigns
# 
# Return on Investment is crucial for business decisions:
# - ROI = Revenue / Ad Spend
# - An ROI of 4.0 means every $1 spent generates $4 in revenue
# - Campaigns with high ROI justify increased budgets

# In[ ]:


# Calculate ROI for each campaign
rois = y_train / x_train
print(f"\nIndividual Campaign ROIs: {rois}")
print(f"Average ROI across all campaigns: {np.mean(rois):.2f}x")
print(f"This suggests each $1k ad spend generates ~${np.mean(rois):.1f}k revenue")


# ### Visualizing the Data
# 
# Before building a model, visualize the relationship:
# - Is there a linear trend? (suggests linear regression is appropriate)
# - Are there outliers? (might need investigation)
# - How tightly do points cluster around the line? (indicates prediction certainty)

# In[ ]:


# Plot actual campaign outcomes
plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, marker='o', c='blue', s=100, alpha=0.7, label='Historical Campaigns')

# Add gridlines for easier reading
plt.grid(True, alpha=0.3)

# Customize labels with business context
plt.title('Advertising Spend vs Sales Revenue', fontsize=14, fontweight='bold')
plt.xlabel('Ad Spend (thousands of dollars)', fontsize=12)
plt.ylabel('Sales Revenue (thousands of dollars)', fontsize=12)

# Highlight specific campaigns for discussion
for i, (x, y) in enumerate(zip(x_train, y_train)):
    plt.annotate(f'Campaign {i+1}', xy=(x, y), xytext=(x+0.3, y+1), fontsize=9)

plt.legend()
plt.tight_layout()
plt.show()


# ## Model Function: The Linear Relationship
# 
# The core equation for simple linear regression:
# $$f_{w,b}(x) = wx + b \tag{1}$$
# 
# **Business Interpretation:**
# - $b$ (bias/intercept): Expected sales with ZERO advertising
#   - Could represent organic traffic, brand loyalty, word-of-mouth
#   - Important for understanding baseline business health
#   
# - $w$ (weight/slope): Additional revenue per unit of ad spend
#   - This is your **marginal return** on advertising
#   - Directly informs budget allocation decisions
#   - Higher $w$ = more efficient ad campaigns
# 
# **Example:** If $w=3$ and $b=7$, then:
# - $0$ spend → $7k baseline sales
# - $5k spend → $3(5)+7 = $22k predicted sales
# - Incremental effect of $5k spend → $15k additional revenue

# In[ ]:


def compute_model_predictions(x, w, b):
    """
    Computes predicted sales revenue given advertising spend.
    
    Args:
      x (ndarray (m,)): Ad spend values for m campaigns/data points
      w (float): Marginal revenue per dollar spent on ads
      b (float): Baseline revenue with zero ad spend
    
    Returns:
      f_wb (ndarray (m,)): Predicted revenue for each input in x
      
    Mathematical basis:
      f_wb[i] = w * x[i] + b
      This assumes a linear relationship between spend and revenue.
      
    Business rationale:
      - We assume each additional dollar spent generates constant incremental revenue
      - This holds true when markets aren't saturated and diminishing returns aren't yet active
      - For extended ranges, polynomial models may capture saturation effects better
    """
    m = x.shape[0]  # Number of data points to predict
    f_wb = np.zeros(m)  # Initialize array to store predictions
    
    # Loop through each data point
    for i in range(m):
        # Apply linear model formula: predicted = slope*spend + intercept
        f_wb[i] = w * x[i] + b
        
    return f_wb


# ## Exploring Different Parameter Values
# 
# The challenge: finding $w$ and $b$ that best fit historical data
# 
# **Approach 1:** Manual experimentation (what we'll do here)
# - Adjust $w$ and $b$, see how well the line fits
# - Good for understanding the model intuitively
# 
# **Approach 2:** Automated optimization (covered in gradient descent labs)
# - Algorithm finds optimal $w$ and $b$ mathematically
# - Minimizes prediction error (loss function)
# - Scalable to larger datasets

# In[ ]:


# START WITH INITIAL GUESS
w_guess = 2.0  # Each $1k spent generates $2k revenue
b_guess = 8.0  # $8k baseline sales without ads

print(f"Initial guess: w={w_guess}, b={b_guess}")

# Compute predictions for all campaigns with this guess
predictions = compute_model_predictions(x_train, w_guess, b_guess)

# Show actual vs predicted comparison
print(f"\n{'Campaign':<10} {'Spend':<8} {'Actual':<10} {'Predicted':<10} {'Error':<8}")
print("-" * 50)
for i in range(len(x_train)):
    error = predictions[i] - y_train[i]
    print(f"{i+1:<10} ${x_train[i]:.0f}k    ${y_train[i]:.0f}k       ${predictions[i]:.0f}k      ${error:+.1f}k")

# Calculate Mean Absolute Error (MAE) to measure overall fit
mae = np.mean(np.abs(predictions - y_train))
print(f"\nMean Absolute Error: ${mae:.2f}k")
print(f"Lower MAE = better model fit")


# ### Visualizing Model Fit
# 
# See how well your parameters explain the data

# In[ ]:


# Plot actual data points
plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, marker='o', c='red', s=100, alpha=0.7, 
            label='Historical Data', edgecolors='black')

# Plot model prediction line
plt.plot(x_train, predictions, c='blue', linewidth=2, 
         label=f'Model (w={w_guess}, b={b_guess})')

# Add annotations showing key metrics
plt.text(x_train[-1], predictions[-1]+3, 
         f'MAE: ${mae:.1f}k\nw={w_guess}\nb={b_guess}',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3),
         fontsize=11)

plt.title('Model Fit Visualization', fontsize=14, fontweight='bold')
plt.xlabel('Ad Spend (thousands of dollars)', fontsize=12)
plt.ylabel('Sales Revenue (thousands of dollars)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ### Challenge: Improve the Model
# 
# Your initial guess may not perfectly match the data. Try to find better $w$ and $b$ values:
# 
# **Guiding Principles:**
# - **Higher w:** Steeper line, predicts more revenue per dollar spent
# - **Higher b:** Line shifts up, predicts more baseline sales
# - **Goal:** Minimize distance between line and data points
# 
# **Hint:** Looking at the data, average ROI is ~3.0x. Try starting with w≈3.0

# In[ ]:


# IMPROVED PARAMETER VALUES
# Based on visual inspection and ROI analysis
w_optimized = 2.85  # Slightly less than average ROI due to diminishing returns
b_optimized = 7.2   # Lower baseline accounts for campaigns that drove most sales

print(f"Optimized parameters: w={w_optimized}, b={b_optimized}")

# Recompute predictions
optimized_predictions = compute_model_predictions(x_train, w_optimized, b_optimized)

# Recalculate MAE
optimized_mae = np.mean(np.abs(optimized_predictions - y_train))
print(f"Improved Mean Absolute Error: ${optimized_mae:.2f}k")
print(f"Improvement: ${mae - optimized_mae:.2f}k reduction in error ({((mae-optimized_mae)/mae*100):.1f}% better)")


# ## Making Strategic Predictions
# 
# Now use your trained model for business planning

# In[ ]:


# Scenario: Planning next quarter's advertising budget
planned_ad_spend = 7.0  # $7,000 planned spending

# Use model to predict expected revenue
predicted_revenue = w_optimized * planned_ad_spend + b_optimized

print("=" * 60)
print(f"BUDGET PLANNING SCENARIO")
print("=" * 60)
print(f"Planned Ad Spend: ${planned_ad_spend:.0f}k")
print(f"Predicted Sales Revenue: ${predicted_revenue:.1f}k")
print(f"Projected ROI: {predicted_revenue/planned_ad_spend:.2f}x")
print(f"Expected Profit Margin: {((predicted_revenue - planned_ad_spend)/predicted_revenue*100):.1f}%")
print("=" * 60)

# Save this prediction for later use
budget_prediction = predicted_revenue


# ### Sensitivity Analysis: What If Parameters Change?
# 
# In reality, marketing effectiveness varies. How robust are our predictions?

# In[ ]:


# Create a sensitivity table
scenarios = [
    ("Optimistic", w_optimized * 1.1, b_optimized * 1.1),   # 10% better performance
    ("Base Case", w_optimized, b_optimized),                 # Our calibrated model
    ("Conservative", w_optimized * 0.9, b_optimized * 0.9),  # 10% worse performance
    ("Worst Case", w_optimized * 0.8, b_optimized * 0.7),    # 20% decline
]

print(f"\n{'Scenario':<15} {'w':<8} {'b':<8} {'Predicted Rev':<12} {'ROI':<8} {'Status'}")
print("-" * 70)
for name, w_val, b_val in scenarios:
    rev = w_val * planned_ad_spend + b_val
    roi = rev / planned_ad_spend
    status = "✅" if roi > 3.0 else "⚠️" if roi > 2.0 else "❌"
    print(f"{name:<15} {w_val:<8.2f} {b_val:<8.2f} ${rev:<11.1f}k {roi:<8.2f}x {status}")


# ### Visualizing Multiple Scenarios
# 
# This helps stakeholders understand prediction uncertainty

# In[ ]:


plt.figure(figsize=(12, 6))

# Plot actual data
plt.scatter(x_train, y_train, marker='o', c='gray', s=120, alpha=0.6, 
            zorder=1, label='Historical Data')

# Plot each scenario
colors = ['green', 'blue', 'orange', 'red']
linestyles = ['solid', 'dashed', 'dotted', 'dashdot']

for i, (name, w_val, b_val) in enumerate(scenarios):
    scenario_preds = compute_model_predictions(x_train, w_val, b_val)
    plt.plot(x_train, scenario_preds, color=colors[i], linestyle=linestyles[i], 
             linewidth=2, alpha=0.7, label=name)

# Mark the planned budget point
plt.axvline(x=planned_ad_spend, color='purple', linestyle='--', 
            linewidth=1.5, alpha=0.5, label='Planned Spend')

plt.title('Revenue Forecast Under Different Performance Scenarios', fontsize=14, fontweight='bold')
plt.xlabel('Ad Spend (thousands of dollars)', fontsize=12)
plt.ylabel('Sales Revenue (thousands of dollars)', fontsize=12)
plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


# ## Advanced: Computing Optimal Parameters Mathematically
# 
# Instead of guessing, we can calculate the best-fit line using:
# **Ordinary Least Squares (OLS)** formulas

# In[ ]:


# OLS formulas for simple linear regression
# These minimize the sum of squared prediction errors

# Slope (w) calculation:
# Covariance of x,y divided by variance of x
w_calc = np.sum((x_train - np.mean(x_train)) * (y_train - np.mean(y_train))) / np.sum((x_train - np.mean(x_train))**2)

# Intercept (b) calculation:
# Average y minus slope times average x
b_calc = np.mean(y_train) - w_calc * np.mean(x_train)

print("=" * 60)
print("MATHEMATICALLY OPTIMIZED PARAMETERS")
print("=" * 60)
print(f"w (slope): {w_calc:.4f}")
print(f"b (intercept): {b_calc:.4f}")
print("=" * 60)

# Compare to manual calibration
print(f"\nDifference from manual calibration:")
print(f"w: {abs(w_calc - w_optimized):.4f} difference")
print(f"b: {abs(b_calc - b_optimized):.4f} difference")

# Calculate final MAE with optimal parameters
optimal_predictions = compute_model_predictions(x_train, w_calc, b_calc)
final_mae = np.mean(np.abs(optimal_predictions - y_train))
print(f"\nFinal Mean Absolute Error (optimal): ${final_mae:.2f}k")


# ## Business Decision Framework
# 
# With a validated model, here's how to make strategic decisions:

# In[ ]:


# Create budget recommendation function
def recommend_budget(target_revenue, w, b):
    """
    Calculates required ad spend to meet revenue target.
    
    Rearranges f = wx + b to solve for x:
    x = (target_revenue - b) / w
    
    Args:
      target_revenue (float): Desired sales revenue in thousands
      w (float): Model slope parameter
      b (float): Model intercept parameter
      
    Returns:
      recommended_spend (float): Required ad spend in thousands
    """
    if w <= 0:
        raise ValueError("Model slope must be positive for valid recommendations")
    
    spend = (target_revenue - b) / w
    return max(0, spend)  # Can't spend negative money


# Test various revenue targets
targets = [25, 35, 50]  # Thousand dollar revenue targets

print("\n" + "=" * 60)
print("BUDGET RECOMMENDATIONS FOR DIFFERENT REVENUE TARGETS")
print("=" * 60)
print(f"{'Target Revenue':<15} {'Required Spend':<15} {'Projected ROI':<12} {'Recommendation'}")
print("-" * 60)

for target in targets:
    required_spend = recommend_budget(target, w_calc, b_calc)
    projected_roi = target / required_spend if required_spend > 0 else float('inf')
    recommendation = "✅ Feasible" if required_spend < 20 else "⚠️ High Budget"
    print(f"${target:<14}k ${required_spend:<14.1f}k {projected_roi:<12.2f}x {recommendation}")

print("=" * 60)


# # Key Takeaways
# 
# ## 1. **Linear Models Enable Data-Driven Decisions**
# - Replace gut feeling with mathematical prediction
# - Quantify the impact of each dollar spent
# - Support budget justification to stakeholders
# 
# ## 2. **Model Quality Matters**
# - Poor parameters → wrong budget allocation → financial loss
# - Validate against historical data before deploying
# - Regular re-calibration as market conditions change
# 
# ## 3. **Understanding Uncertainty**
# - Single-point predictions hide variability
# - Sensitivity analysis shows risk exposure
# - Conservative estimates prevent overspending
# 
# ## 4. **Beyond Simple Linear Regression**
# - **Diminishing Returns:** After certain spend, each additional dollar returns less
# - **Multiple Channels:** Digital, TV, radio all affect revenue differently
# - **Time Effects:** Marketing impact decays or compounds over time
# - **External Factors:** Economy, competition, seasonality influence results
# 
# **Next Steps:** In subsequent labs, you'll learn gradient descent to automate
# finding optimal $w$ and $b$, and extend to multiple predictors (multivariate
# regression) for even more powerful business intelligence.


# In[ ]: