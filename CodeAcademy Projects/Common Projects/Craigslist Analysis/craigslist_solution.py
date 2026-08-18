# Load libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Set seed
np.random.seed(1)

# Import data
housing = pd.read_csv('craigslist.csv')

# Fit model1
model1 = sm.OLS.from_formula('price ~ type + sqfeet + beds + baths', data=housing).fit()

# Fit model2
model2 = sm.OLS.from_formula('price ~ type + sqfeet + beds + baths + comes_furnished + laundry_options + parking_options + smoking_allowed', data=housing).fit()

# Fit model3
model3 = sm.OLS.from_formula('price ~ type + sqfeet + beds + baths + comes_furnished + laundry_options + parking_options + smoking_allowed + cats_allowed + dogs_allowed', data=housing).fit()

# Print R-squared for all models
print('model1 R-squared: ', model1.rsquared)
print('model2 R-squared: ', model2.rsquared)
print('model3 R-squared: ', model3.rsquared)

# Print adjusted R-squared for all models
print('model1 adj R-squared: ', model1.rsquared_adj)
print('model2 adj R-squared: ', model2.rsquared_adj)
print('model3 adj R-squared: ', model3.rsquared_adj)

# Run an F test comparing model2 and model3
from statsmodels.stats.anova import anova_lm
anova_results = anova_lm(model2, model3)
print(anova_results.round(2))

# Print log likelihood for all models
print('model1 log likelihood: ', model1.llf)
print('model2 log likelihood: ', model2.llf)
print('model3 log likelihood: ', model3.llf)

# Print AIC for all models
print('model1 AIC: ', model1.aic)
print('model2 AIC: ', model2.aic)
print('model3 AIC: ', model3.aic)

# Print BIC for all models
print('model1 BIC: ', model1.bic)
print('model2 BIC: ', model2.bic)
print('model3 BIC: ', model3.bic)

# Split housing data
indices = range(len(housing))
s = int(0.8*len(indices))
train_ind = np.random.choice(indices, size = s, replace = False)
test_ind = list(set(indices) - set(train_ind))
housing_train = housing.iloc[train_ind]
housing_test = housing.iloc[test_ind]

# Fit model2 with training data
model2_train = sm.OLS.from_formula('price ~ type + sqfeet + beds + baths + comes_furnished + laundry_options + parking_options + smoking_allowed', data=housing_train).fit()

# Fit model3 with training data
model3_train = sm.OLS.from_formula('price ~ type + sqfeet + beds + baths + comes_furnished + laundry_options + parking_options + smoking_allowed + cats_allowed + dogs_allowed', data=housing_train).fit()

# Calculate predicted price based on model2
fitted_mod2 = model2_train.predict(housing_test)

# Calculate predicted price based on model3
fitted_mod3 = model3_train.predict(housing_test)

# Calculate PRMSE for model2
true = housing_test.price
prmse2 = np.mean((true-fitted_mod2)**2)**.5

# Calculate PRMSE for model3
prmse3 = np.mean((true-fitted_mod3)**2)**.5

# Print PRMSE for both models
print('model2 PRMSE: ', prmse2)
print('model3 PRMSE: ', prmse3)