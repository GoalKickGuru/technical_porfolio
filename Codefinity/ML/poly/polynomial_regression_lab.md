# Extended Lab: Polynomial Regression with scikit-learn

## Learning Objectives

By the end of this lab you will be able to:

- Explain why a straight line often fails to capture curved relationships between a feature and a target, and how polynomial features fix that.
- Build a polynomial regression model using `PolynomialFeatures` + `LinearRegression`.
- Correctly transform new/unseen data with an *already-fitted* transformer.
- Evaluate a model with train/test splits and metrics (R², MSE) instead of eyeballing a plot.
- Diagnose underfitting and overfitting by varying the polynomial degree.
- Combine preprocessing and modeling into a single `Pipeline`.
- Apply regularization (Ridge) to control overfitting in high-degree polynomial models.

---

## 1. Background

Linear regression fits a straight line:

```
y = b0 + b1*x
```

Many real relationships are curved (e.g., house price vs. age, where price might drop quickly at first and then level off). Polynomial regression keeps the model "linear in its parameters" (so we can still use `LinearRegression`) but expands the *input features* into powers of `x`:

```
y = b0 + b1*x + b2*x^2 + ... + bn*x^n
```

`PolynomialFeatures(degree=n)` does this feature expansion for you — it takes a column like `age` and produces a new matrix with columns `[1, age, age^2, ..., age^n]`. The model itself is still `LinearRegression`, just trained on these expanded features.

**Key gotcha:** whatever transformation you apply to your training data (`X`), you must apply the *exact same fitted transformer* to any new data before predicting. That's why we call `poly.transform(X_new)` — never `poly.fit_transform(X_new)` — after the transformer has already been fit on the training set.

---

## 2. The Base Task (Warm-up)

This is the original exercise. Make sure you understand every line before moving to the extended tasks.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

df = pd.read_csv('https://codefinity-content-media.s3.eu-west-1.amazonaws.com/b22d1166-efda-45e8-979e-6c3ecfc566fc/houses_poly.csv')

# 1. Assign X as a DataFrame containing the 'age' column
X = df[['age']]
y = df['price']

n = 2  # degree of the polynomial

# 2. Create X_poly using PolynomialFeatures
poly = PolynomialFeatures(n)
X_poly = poly.fit_transform(X)

# 3. Build and train a LinearRegression model on the transformed features
model = LinearRegression()
model.fit(X_poly, y)

# 4. Reshape X_new into a 2-D array
X_new = np.linspace(0, 125, 200).reshape(-1, 1)

# 5. Preprocess X_new with the SAME fitted transformer
X_new_poly = poly.transform(X_new)

# Predict
y_pred = model.predict(X_new_poly)

# Visualize
plt.scatter(X, y, alpha=0.4)
plt.plot(X_new, y_pred, color='orange')
plt.xlabel('Age')
plt.ylabel('Price')
plt.title(f'Polynomial Regression (degree={n})')
plt.show()

# 6. Print the model's intercept and coefficients
print('Intercept:', model.intercept_)
print('Coefficients:', model.coef_)
```

**Checkpoint questions:**
1. Why is `X` written as `df[['age']]` (double brackets) instead of `df['age']`?
2. What does `X_poly`'s first column contain, and why?
3. Why do we call `.transform()` and not `.fit_transform()` on `X_new`?

*(Answers in Section 6.)*

---

## 3. Extended Task A — Compare Degrees and Spot Overfitting

Wrap the model-building logic in a loop over several degrees, and plot all the curves on the same scatter plot.

```python
degrees = [1, 2, 4, 8, 15]
X_new = np.linspace(df['age'].min(), df['age'].max(), 200).reshape(-1, 1)

plt.figure(figsize=(9, 6))
plt.scatter(X, y, alpha=0.3, label='data')

for d in degrees:
    poly_d = PolynomialFeatures(d)
    X_poly_d = poly_d.fit_transform(X)

    model_d = LinearRegression()
    model_d.fit(X_poly_d, y)

    X_new_poly_d = poly_d.transform(X_new)
    y_pred_d = model_d.predict(X_new_poly_d)

    plt.plot(X_new, y_pred_d, label=f'degree {d}')

plt.xlabel('Age')
plt.ylabel('Price')
plt.legend()
plt.title('Polynomial Regression: Effect of Degree')
plt.show()
```

**Task:** Run this and answer:
- Which degree looks like it *underfits* (too simple, misses the curve)?
- Which degree looks like it *overfits* (wiggles wildly, chases noise, especially near the edges of the data)?
- Which degree looks like the best balance?

---

## 4. Extended Task B — Quantify It: Train/Test Split + Metrics

Visual inspection is a good start, but you should always back it up with numbers on **held-out data**.

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

results = []

for d in range(1, 11):
    poly_d = PolynomialFeatures(d)
    X_train_poly = poly_d.fit_transform(X_train)
    X_test_poly = poly_d.transform(X_test)  # transform only, using the training fit

    model_d = LinearRegression()
    model_d.fit(X_train_poly, y_train)

    y_train_pred = model_d.predict(X_train_poly)
    y_test_pred = model_d.predict(X_test_poly)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    results.append((d, train_r2, test_r2, test_mse))

results_df = pd.DataFrame(results, columns=['degree', 'train_r2', 'test_r2', 'test_mse'])
print(results_df)

plt.figure(figsize=(8, 5))
plt.plot(results_df['degree'], results_df['train_r2'], marker='o', label='Train R²')
plt.plot(results_df['degree'], results_df['test_r2'], marker='o', label='Test R²')
plt.xlabel('Polynomial Degree')
plt.ylabel('R² score')
plt.legend()
plt.title('Train vs Test R² by Degree')
plt.show()
```

**Task:** Identify the degree where `train_r2` keeps climbing but `test_r2` starts to drop. That gap is the textbook signature of overfitting.

---

## 5. Extended Task C — Pipeline + Ridge Regularization

Chaining `PolynomialFeatures` and `LinearRegression` manually works, but a `Pipeline` prevents a very common bug: accidentally calling `fit_transform` on test data. It also makes regularization easy to bolt on.

```python
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

degree = 10  # deliberately high, to show regularization taming it

# Plain high-degree polynomial regression (likely overfits)
plain_pipeline = make_pipeline(
    PolynomialFeatures(degree),
    LinearRegression()
)
plain_pipeline.fit(X_train, y_train)

# Ridge-regularized version
ridge_pipeline = make_pipeline(
    PolynomialFeatures(degree),
    Ridge(alpha=10.0)
)
ridge_pipeline.fit(X_train, y_train)

X_new = np.linspace(df['age'].min(), df['age'].max(), 200).reshape(-1, 1)

y_plain_pred = plain_pipeline.predict(X_new)
y_ridge_pred = ridge_pipeline.predict(X_new)

plt.figure(figsize=(9, 6))
plt.scatter(X, y, alpha=0.3, label='data')
plt.plot(X_new, y_plain_pred, color='red', label=f'degree {degree}, no regularization')
plt.plot(X_new, y_ridge_pred, color='green', label=f'degree {degree}, Ridge(alpha=10)')
plt.legend()
plt.xlabel('Age')
plt.ylabel('Price')
plt.title('Taming Overfitting with Ridge Regularization')
plt.show()

print('Plain test R²:', r2_score(y_test, plain_pipeline.predict(X_test)))
print('Ridge test R²:', r2_score(y_test, ridge_pipeline.predict(X_test)))
```

**Task:** Try a few different `alpha` values (e.g. 0.1, 1, 10, 100). What happens to the curve as `alpha` grows very large? What happens as it approaches 0?

---

## 6. Answers to Checkpoint Questions (Section 2)

1. `df[['age']]` returns a **DataFrame** (2-D), which is what scikit-learn transformers/estimators expect as input. `df['age']` returns a **Series** (1-D) and would raise an error or behave unexpectedly when passed to `PolynomialFeatures`.
2. The first column of `X_poly` is all `1`s — it represents `age^0`, the bias/intercept term that `PolynomialFeatures` adds by default (`include_bias=True`).
3. Because `poly` was already **fit** on the training data (it learned the degree and the number of input features during `fit_transform(X)`). Calling `fit_transform` again on `X_new` would re-fit the transformer, which is not just unnecessary but dangerous — it can silently produce a different/incompatible feature space if `X_new` had different characteristics, and more generally it breaks the principle that test-time preprocessing must exactly mirror training-time preprocessing.

---

## 7. Challenge (Optional, No Starter Code)

1. Load the `houses_poly.csv` data and check if there's a second numeric feature besides `age`. If so, build a model using **both** features with `PolynomialFeatures(degree=2)` (note: with 2 input features, degree 2 will also create an interaction term like `age * other_feature`). Print `poly.get_feature_names_out()` to see what each coefficient corresponds to.
2. Use `GridSearchCV` to automatically search over `degree` (1–10) and `Ridge alpha` (e.g. `[0.01, 0.1, 1, 10, 100]`) using the pipeline from Section 5, and report the best combination.
3. Write a one-paragraph explanation (in your own words) of why polynomial regression is still called "linear regression" even though the resulting curve is not a straight line.

---

## 8. Summary Cheat Sheet

| Step | Code pattern | Common mistake to avoid |
|---|---|---|
| Select feature(s) | `X = df[['col']]` | Using single brackets → 1-D Series |
| Expand features | `poly = PolynomialFeatures(n); X_poly = poly.fit_transform(X)` | Calling `fit_transform` again later on new data |
| Fit model | `model.fit(X_poly, y)` | Fitting on raw `X` instead of `X_poly` |
| Prepare new data | `X_new.reshape(-1, 1)` | Forgetting new 1-D arrays need reshaping to 2-D |
| Transform new data | `poly.transform(X_new)` | Using `fit_transform` instead of `transform` |
| Evaluate | `r2_score`, `mean_squared_error` on a **test** set | Judging model quality from the training plot alone |
| Control complexity | Try multiple degrees; add `Ridge`/`Lasso` | Picking degree by best-looking training fit |
