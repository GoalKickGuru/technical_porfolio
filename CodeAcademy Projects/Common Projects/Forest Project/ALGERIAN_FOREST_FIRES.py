"""
ALGERIAN FOREST FIRES - EXTENDED ANALYSIS
==========================================
Comprehensive statistical analysis with enhanced modeling and visualization
Author: Lumo Extended Analysis
Date: 2026-07-25
"""

# ============================================================================
# IMPORT LIBRARIES
# ============================================================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Set aesthetic parameters
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ============================================================================
# LOAD AND INITIAL DATA EXPLORATION
# ============================================================================
forests = pd.read_csv('forests.csv')

print("=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)
print(f"\nShape: {forests.shape[0]} observations × {forests.shape[1]} variables")
print(f"\nColumn types:\n{forests.dtypes}")
print(f"\nMissing values:\n{forests.isnull().sum()}")
print(f"\nFire occurrence distribution:")
print(forests['fire'].value_counts())
print(f"\nRegional distribution:")
print(forests['region'].value_counts())

# Encode binary and categorical variables
forests['fire_binary'] = forests['fire'].astype(int)
forests['region_encoded'] = forests['region'].map({'Bejaia': 0, 'Sidi Bel-abbes': 1})

# ============================================================================
# DESCRIPTIVE STATISTICS BY FIRE STATUS
# ============================================================================
print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS BY FIRE STATUS")
print("=" * 70)
descriptive_stats = forests.groupby('fire').agg([
    ('mean', 'mean'),
    ('std', 'std'),
    ('min', 'min'),
    ('max', 'max')
])
print(descriptive_stats.T.round(2))

# ============================================================================
# MULTICOLLINEARITY ANALYSIS WITH ENHANCED HEATMAP
# ============================================================================
quantitative_vars = ['temp', 'humid', 'wind', 'rain', 'FFMC', 'DMC', 'DC', 
                     'ISI', 'BUI', 'FWI']
corr_grid = forests[quantitative_vars].corr()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Heatmap of correlations
sns.heatmap(corr_grid, xticklabels=quantitative_vars, yticklabels=quantitative_vars, 
            annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=axes[0])
axes[0].set_title('Correlation Matrix - All Quantitative Variables', fontsize=14, fontweight='bold')
axes[0].set_xticklabels(quantitative_vars, rotation=45, ha='right')

# Strong correlations highlight
strong_corr = np.abs(corr_grid).unstack().reset_index()
strong_corr.columns = ['var1', 'var2', 'correlation']
strong_corr = strong_corr[strong_corr['var1'] != strong_corr['var2']]
strong_corr = strong_corr[strong_corr['correlation'] > 0.7].sort_values('correlation', ascending=False)

print("\n" + "=" * 70)
print("STRONG CORRELATIONS (>0.7)")
print("=" * 70)
for _, row in strong_corr.iterrows():
    print(f"{row['var1']} ↔ {row['var2']}: {row['correlation']:.3f}")

# Correlation network visualization concept
axes[1].scatter(range(len(quantitative_vars)), range(len(quantitative_vars)))
for i in range(len(quantitative_vars)):
    for j in range(i+1, len(quantitative_vars)):
        corr_value = abs(corr_grid.iloc[i, j])
        if corr_value > 0.5:
            axes[1].annotate('', xy=(j, i), xytext=(i, j),
                            arrowprops=dict(arrowstyle='-', color='red', alpha=0.3, 
                                          linewidth=corr_value*3))
axes[1].set_xticks(range(len(quantitative_vars)))
axes[1].set_yticks(range(len(quantitative_vars)))
axes[1].set_xticklabels(quantitative_vars, rotation=45, ha='right', fontsize=8)
axes[1].set_yticklabels(quantitative_vars, fontsize=8)
axes[1].set_title('Significant Correlations Network (>0.5)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Variable')
axes[1].set_ylabel('Variable')

plt.tight_layout()
plt.savefig('multicollinearity_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()
# ============================================================================
# REGIONAL COMPARISON VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 70)
print("REGIONAL COMPARISON ANALYSIS")
print("=" * 70)

# Box plots for all variables by region
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.ravel()

for idx, var in enumerate(quantitative_vars):
    sns.boxplot(x='region', y=var, data=forests, ax=axes[idx])
    axes[idx].set_title(f'{var} by Region', fontsize=10)
    axes[idx].set_xlabel('')
    
plt.suptitle('Weather Variables Distribution by Region', fontsize=16, fontweight='bold', y=1.0)
plt.tight_layout()
plt.savefig('regional_comparison_boxplots.png', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()

# ============================================================================
# HUMIDITY vs TEMPERATURE WITH REGION COLORING
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 8))

# Scatter plot by region
bejaia_data = forests[forests['region'] == 'Bejaia']
sidi_data = forests[forests['region'] == 'Sidi Bel-abbes']

ax.scatter(bejaia_data['temp'], bejaia_data['humid'], 
           c='#6d4aff', alpha=0.6, s=60, label='Bejaia', edgecolors='white', linewidth=0.5)
ax.scatter(sidi_data['temp'], sidi_data['humid'], 
           c='#ff6b35', alpha=0.6, s=60, label='Sidi Bel-abbes', edgecolors='white', linewidth=0.5)

# Add regression lines by region
bejaia_slope, bejaia_intercept, r_value, p_value, std_err = stats.linregress(
    bejaia_data['temp'], bejaia_data['humid'])
sidi_slope, sidi_intercept, r_value, p_value, std_err = stats.linregress(
    sidi_data['temp'], sidi_data['humid'])

temp_range = np.linspace(20, 42, 100)
ax.plot(temp_range, bejaia_intercept + bejaia_slope * temp_range, 
        color='#6d4aff', linewidth=3, linestyle='--', label=f'Bejaia: y={bejaia_intercept:.1f}+{bejaia_slope:.2f}x')
ax.plot(temp_range, sidi_intercept + sidi_slope * temp_range, 
        color='#ff6b35', linewidth=3, linestyle='--', label=f'Sidi: y={sidi_intercept:.1f}+{sidi_slope:.2f}x')

ax.set_xlabel('Temperature (°C)', fontsize=12)
ax.set_ylabel('Relative Humidity (%)', fontsize=12)
ax.set_title('Humidity vs Temperature by Region', fontsize=16, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Annotation for insights
ax.annotate(f'Bejaia correlation: r={r_value:.2f}', xy=(0.02, 0.98), xycoords='axes fraction',
           fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.annotate(f'Sidi Bel-abbes correlation: r={stats.linregress(sidi_data["temp"], sidi_data["humid"])[0]:.2f}', 
           xy=(0.02, 0.93), xycoords='axes fraction', fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()
plt.savefig('humidity_temperature_by_region.png', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()
# ============================================================================
# MULTIPLE LINEAR REGRESSION FOR HUMIDITY PREDICTION
# ============================================================================
print("\n" + "=" * 70)
print("MULTIPLE LINEAR REGRESSION - HUMIDITY MODEL")
print("=" * 70)

# Fit model with region as categorical
modelH = sm.OLS.from_formula('humid ~ temp + C(region)', data=forests).fit()
print(modelH.summary())

print("\n" + "-" * 70)
print("COEFFICIENT INTERPRETATION")
print("-" * 70)
print(f"Intercept: {modelH.params['Intercept']:.2f}")
print(f"Temperature coefficient: {modelH.params['C(region)[T.Sidi Bel-abbes]']:.2f}")
print(f"Region effect: Sidi Bel-abbes has {abs(modelH.params['C(region)[T.Sidi Bel-abbes]']):.1f}% lower humidity than Bejaia")

# Calculate R-squared and adjusted R-squared
print(f"\nR-squared: {modelH.rsquared:.3f}")
print(f"Adjusted R-squared: {modelH.rsquared_adj:.3f}")
print(f"F-statistic: {modelH.fvalue:.2f} (p-value: {modelH.f_pvalue:.2e})")

# ============================================================================
# RESIDUAL ANALYSIS
# ============================================================================
forests['humid_predicted_H'] = modelH.predict()
forests['residuals_H'] = forests['humid'] - forests['humid_predicted_H']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Residuals vs predicted
axes[0].scatter(forests['humid_predicted_H'], forests['residuals_H'], 
                alpha=0.6, edgecolors='black', linewidth=0.5)
axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0].set_xlabel('Predicted Humidity')
axes[0].set_ylabel('Residuals')
axes[0].set_title('Residuals vs Predicted Values', fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Q-Q Plot for normality
sm.qqplot(forests['residuals_H'], line='45', ax=axes[1])
axes[1].set_title('Q-Q Plot - Residual Normality Check', fontweight='bold')

# Histogram of residuals
axes[2].hist(forests['residuals_H'], bins=20, edgecolor='black', alpha=0.7)
axes[2].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[2].set_xlabel('Residuals')
axes[2].set_ylabel('Frequency')
axes[2].set_title('Distribution of Residuals', fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('humidity_model_residuals.png', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()
# ============================================================================
# FFMC PREDICTION WITH INTERACTION TERM (TEMP × FIRE)
# ============================================================================
print("\n" + "=" * 70)
print("INTERACTION MODEL - FFMC PREDICTION WITH FIRE STATUS")
print("=" * 70)

modelF = sm.OLS.from_formula('FFMC ~ temp * fire_binary', data=forests).fit()
print(modelF.summary())

print("\n" + "-" * 70)
print("EQUATION BREAKDOWN BY FIRE STATUS")
print("-" * 70)
print("Full Equation: FFMC = {beta0:.2f} + {beta1:.2f}*temp + {beta2:.2f}*fire + {beta3:.2f}*temp*fire".format(
    beta0=modelF.params['Intercept'],
    beta1=modelF.params['temp'],
    beta2=modelF.params['fire_binary'],
    beta3=modelF.params['temp:fire_binary']
))

print("\nFor NO FIRE (fire=0):")
print(f"  FFMC = {modelF.params['Intercept']:.2f} + {modelF.params['temp']:.2f}*temp")

print("\nFor FIRE (fire=1):")
print(f"  FFMC = {modelF.params['Intercept'] + modelF.params['fire_binary']:.2f} + {modelF.params['temp'] + modelF.params['temp:fire_binary']:.2f}*temp")

print("\n" + "-" * 70)
print("INTERACTION INTERPRETATION")
print("-" * 70)
print(f"The interaction term ({modelF.params['temp:fire_binary']:.2f}) indicates that:")
print(f"- At fire locations, temperature's effect on FFMC is WEAKER than at non-fire locations")
print(f"- Difference in slopes: {modelF.params['temp'] - (modelF.params['temp'] + modelF.params['temp:fire_binary']):.2f}")
print(f"- This suggests fire-prone areas have higher baseline FFMC but less temperature sensitivity")

# ============================================================================
# VISUALIZATION OF INTERACTION EFFECTS
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 8))

no_fire_data = forests[forests['fire_binary'] == 0]
fire_data = forests[forests['fire_binary'] == 1]

ax.scatter(no_fire_data['temp'], no_fire_data['FFMC'], 
           c='#6d4aff', alpha=0.5, s=50, label='No Fire', edgecolors='white')
ax.scatter(fire_data['temp'], fire_data['FFMC'], 
           c='#ff6b35', alpha=0.5, s=50, label='Fire Occurred', edgecolors='white')

# Regression lines
temp_range = np.linspace(22, 42, 100)
no_fire_line = modelF.params['Intercept'] + modelF.params['temp'] * temp_range
fire_line = (modelF.params['Intercept'] + modelF.params['fire_binary'] + 
             (modelF.params['temp'] + modelF.params['temp:fire_binary']) * temp_range)

ax.plot(temp_range, no_fire_line, color='#6d4aff', linewidth=3, 
        label=f'No Fire: slope={modelF.params["temp"]:.2f}')
ax.plot(temp_range, fire_line, color='#ff6b35', linewidth=3, 
        label=f'Fire: slope={modelF.params["temp"]+modelF.params["temp:fire_binary"]:.2f}')

ax.set_xlabel('Temperature (°C)', fontsize=12)
ax.set_ylabel('Fine Fuel Moisture Code (FFMC)', fontsize=12)
ax.set_title('FFMC vs Temperature by Fire Status (With Interaction)', fontsize=16, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ffmc_temp_interaction.png', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()
# ============================================================================
# POLYNOMIAL REGRESSION - FFMC vs HUMIDITY
# ============================================================================
print("\n" + "=" * 70)
print("POLYNOMIAL MODEL - FFMC AS FUNCTION OF HUMIDITY")
print("=" * 70)

forests['humid_squared'] = forests['humid'] ** 2
modelP = sm.OLS.from_formula('FFMC ~ humid + humid_squared', data=forests).fit()
print(modelP.summary())

print("\n" + "-" * 70)
print("QUADRATIC EQUATION")
print("-" * 70)
print(f"FFMC = {modelP.params['Intercept']:.2f} + {modelP.params['humid']:.3f}*humid + {modelP.params['humid_squared']:.4f}*humid²")

# Find vertex of parabola
vertex_humid = -modelP.params['humid'] / (2 * modelP.params['humid_squared'])
vertex_ffmc = modelP.params['Intercept'] + modelP.params['humid'] * vertex_humid + modelP.params['humid_squared'] * vertex_humid**2
print(f"\nVertex (minimum FFMC): humidity = {vertex_humid:.1f}%, FFMC = {vertex_ffmc:.1f}")

# Sample predictions
print("\nSample Predicted Values:")
humid_levels = [25, 35, 50, 60, 70, 85]
for h in humid_levels:
    ffmc_pred = modelP.params['Intercept'] + modelP.params['humid'] * h + modelP.params['humid_squared'] * h**2
    print(f"  Humidity {h}% → FFMC = {ffmc_pred:.1f}")

# ============================================================================
# VISUALIZATION OF QUADRATIC RELATIONSHIP
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 8))

ax.scatter(forests['humid'], forests['FFMC'], 
           c=forests['fire_binary'].map({0: '#6d4aff', 1: '#ff6b35'}),
           alpha=0.5, s=60, edgecolors='white', linewidth=0.5)

# Quadratic curve
humid_range = np.linspace(20, 95, 200)
ffmc_curve = modelP.params['Intercept'] + modelP.params['humid'] * humid_range + modelP.params['humid_squared'] * humid_range**2
ax.plot(humid_range, ffmc_curve, color='purple', linewidth=3, label='Quadratic Fit')

# Vertex marker
ax.scatter([vertex_humid], [vertex_ffmc], color='gold', s=200, marker='*', 
           label=f'Minimum FFMC at {vertex_humid:.0f}% humidity', zorder=5)

ax.set_xlabel('Relative Humidity (%)', fontsize=12)
ax.set_ylabel('Fine Fuel Moisture Code (FFMC)', fontsize=12)
ax.set_title('Non-Linear Relationship: FFMC vs Humidity (Quadratic Model)', fontsize=16, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Annotations
ax.annotate(f'R² = {modelP.rsquared:.3f}', xy=(0.02, 0.98), xycoords='axes fraction',
           fontsize=11, verticalalignment='top', 
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('ffmc_humidity_quadratic.png', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()
# ============================================================================
# LOGISTIC REGRESSION FOR FIRE CLASSIFICATION
# ============================================================================
print("\n" + "=" * 70)
print("LOGISTIC REGRESSION - FIRE OCCURRENCE PREDICTION")
print("=" * 70)

# Prepare features
features = ['temp', 'humid', 'wind', 'rain', 'FFMC', 'ISI', 'BUI']
X = forests[features]
y = forests['fire_binary']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# Evaluate
y_pred = log_model.predict(X_test)
y_pred_proba = log_model.predict_proba(X_test)[:, 1]

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Fire', 'Fire']))

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")

print("\n" + "-" * 70)
print("FEATURE COEFFICIENTS (Log Odds)")
print("-" * 70)
feature_importance = pd.DataFrame({
    'Feature': features,
    'Coefficient': log_model.coef_[0]
}).sort_values('Coefficient', ascending=False)
print(feature_importance.to_string(index=False))

print("\n" + "-" * 70)
print("KEY FINDINGS")
print("-" * 70)
positive_coefs = feature_importance[feature_importance['Coefficient'] > 0]['Feature'].tolist()
negative_coefs = feature_importance[feature_importance['Coefficient'] < 0]['Feature'].tolist()
print(f"Factors INCREASING fire risk: {positive_coefs}")
print(f"Factors DECREASING fire risk: {negative_coefs}")

# ============================================================================
# FEATURE IMPORTANCE VISUALIZATION
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 8))

colors = ['#ff6b35' if coef > 0 else '#6d4aff' for coef in feature_importance['Coefficient']]
bars = ax.barh(feature_importance['Feature'], feature_importance['Coefficient'], color=colors)

ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Coefficient (log odds)', fontsize=12)
ax.set_title('Logistic Regression Coefficients - Fire Risk Factors', fontsize=16, fontweight='bold')

# Add value labels
for i, (bar, coef) in enumerate(zip(bars, feature_importance['Coefficient'])):
    width = coef
    ax.text(width + (0.1 if coef > 0 else -0.1), bar.get_y() + bar.get_height()/2,
           f'{coef:.2f}', va='center', ha='left' if coef > 0 else 'right', fontsize=10)

plt.tight_layout()
plt.savefig('fire_risk_coefficients.png', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()
# ============================================================================
# FIRE WEATHER INDEX COMPONENT RELATIONSHIPS
# ============================================================================
print("\n" + "=" * 70)
print("FIRE WEATHER INDEX (FWI) MODELING")
print("=" * 70)

# FWI from ISI and BUI
modelFWI = sm.OLS.from_formula('FWI ~ ISI + BUI', data=forests).fit()
print(modelFWI.summary())

print("\n" + "-" * 70)
print("FWI COMPONENT EQUATION")
print("-" * 70)
print(f"FWI = {modelFWI.params['Intercept']:.2f} + {modelFWI.params['ISI']:.3f}*ISI + {modelFWI.params['BUI']:.3f}*BUI")

print(f"\nR² = {modelFWI.rsquared:.3f} - {modelFWI.rsquared*100:.1f}% of FWI variance explained by ISI and BUI")

# ============================================================================
# ADVANCED: ALL-PAIRS SCATTER MATRIX FOR KEY VARIABLES
# ============================================================================
key_vars = ['temp', 'humid', 'FFMC', 'ISI', 'BUI', 'FWI', 'fire_binary']
pair_plot_vars = forests[key_vars].copy()

g = sns.pairplot(pair_plot_vars, hue='fire_binary', 
                 diag_kind='kde', corner=True, height=2.5, aspect=1.2)
g.map_lower(sns.kdeplot, levels=4, cmap="Blues")
g.map_diag(sns.histplot, kde=True)

g.fig.suptitle('Pairwise Relationships - Fire Status Differences', 
               y=1.02, fontsize=16, fontweight='bold')
plt.savefig('pairplot_fire_status.png', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()
