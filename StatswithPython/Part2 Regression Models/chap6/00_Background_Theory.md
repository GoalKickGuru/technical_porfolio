# Simple Linear Regression with OLS — Background, Theory, and Limitations

This document is the companion reading for the lab set built around the three
source notebooks (*Ordinary Least Squares*, *Linear Model Assumptions*,
*Model Validation & Regression Variations*) and Chapter 6 of *Building
Statistical Models in Python*. Read this before (or alongside) the
**Extended Lab** notebook.

---

## 1. What problem are we solving?

Simple linear regression asks: given pairs of observations `(x_i, y_i)`,
is there a straight-line relationship

```
y = β0 + β1·x + ε
```

that lets us (a) describe how y changes with x, and (b) predict y for new
x values? `β0` is the intercept (value of y when x = 0), `β1` is the slope
(change in y per unit change in x), and `ε` is an unobserved error term that
absorbs everything the line doesn't explain — measurement noise, omitted
variables, natural randomness.

## 2. Ordinary Least Squares (OLS): deriving the estimator

We never observe the true `β0, β1`; we estimate them as `β̂0, β̂1` by
choosing the line that minimizes the **sum of squared errors (SSE)**:

```
S(β̂0, β̂1) = Σ (y_i − β̂0 − β̂1·x_i)²
```

Squaring is used instead of absolute value because it is differentiable
everywhere and penalizes large errors more heavily. Setting the partial
derivatives of `S` with respect to `β̂0` and `β̂1` to zero and solving the
resulting **normal equations** gives closed-form estimators:

```
β̂1 = Σ(x_i − x̄)(y_i − ȳ) / Σ(x_i − x̄)²
β̂0 = ȳ − β̂1·x̄
```

Intuition: `β̂1` is the (scaled) covariance between x and y; `β̂0` forces
the fitted line through the point of means `(x̄, ȳ)`. This is exactly the
`least_squares_method()` function in Notebook 1 — implementing it by hand
is the best way to internalize where the summary-table numbers come from.

**Matrix form.** With a design matrix `X` (a column of 1s for the intercept
plus the predictor column) and target vector `y`, the same result is:

```
β̂ = (XᵀX)⁻¹ Xᵀy
```

This is exactly what `statsmodels.OLS` computes internally, which is why
`sm.add_constant()` is mandatory — without an explicit column of ones,
`statsmodels` fits a line forced through the origin.

## 3. Coefficients of correlation (r) and determination (r²)

- **r** measures the strength and direction of the *linear* association
  between x and y, bounded in `[-1, 1]`. It is symmetric (swapping x and y,
  or rescaling either variable, does not change `|r|`).
- **r²** (coefficient of determination) is the proportion of the variance
  in y explained by x:

```
r² = 1 − SS_res / SS_tot
```

  where `SS_res = Σ(y_i − ŷ_i)²` and `SS_tot = Σ(y_i − ȳ)²`. In simple
  linear regression, `r²` is literally the square of the Pearson
  correlation coefficient — and in `statsmodels`, `R-squared` and
  `Adj. R-squared` are equal for a single-predictor model (the adjustment
  penalty only bites once you add more predictors).

**Correlation is not causation.** A high r² tells you the line fits well
in-sample; it says nothing about whether x *causes* y, or whether the
relationship will hold outside the observed range of x (extrapolation
risk).

## 4. The four assumptions of OLS

Violating these doesn't necessarily make the fitted line "wrong," but it
does invalidate the standard errors, p-values, and confidence intervals
that `statsmodels` reports — i.e., you can still get a slope estimate, but
you can no longer trust the inference built on top of it.

| # | Assumption | What it means | How to check |
|---|---|---|---|
| 1 | **Linearity** | The true relationship between x and y (or a transformation of them) is a straight line | Scatter plot of y vs. x; Residuals-vs-Fitted plot should show no curvature |
| 2 | **Normality of residuals** | The *errors*, not the raw x or y variables, are approximately Gaussian | Histogram of residuals; Normal Q-Q plot |
| 3 | **Homoscedasticity** | Residual variance is constant across the range of fitted values | Residuals vs. Fitted / Scale-Location plot — look for a "megaphone" shape |
| 4 | **Independence of errors** | One observation's error tells you nothing about another's | No plot proves this — it comes from the sampling design; for ordered/time data, check the Durbin-Watson statistic and ACF/PACF of residuals |

A subtlety worth remembering (and often missed): **x and y themselves do
not need to be normally distributed** — only the residuals do. Notebook 2
demonstrates this directly by fitting a line through two skewed variables
and showing the residuals still come out approximately normal.

## 5. Reading the `statsmodels` OLS summary table

Key fields and what they tell you:

- **coef**: the estimated β̂ for each regressor (including `const`).
- **std err**: the standard error of that coefficient — smaller means a
  more precisely estimated effect.
- **t, P>|t|**: the t-statistic and p-value for the null hypothesis
  `β = 0`. A small p-value (conventionally < 0.05) suggests the predictor
  has a statistically significant relationship with y.
- **[0.025, 0.975]**: the 95% confidence interval for the coefficient. If
  it doesn't contain 0, that agrees with a significant p-value.
- **R-squared / Adj. R-squared**: variance explained (see §3).
- **F-statistic / Prob (F-statistic)**: tests whether the model as a whole
  (all slopes = 0) is significant — with one predictor this duplicates the
  t-test on that predictor.
- **Durbin-Watson**: tests for first-order autocorrelation in the
  residuals (see §6). Ranges roughly 0–4; near 2 = no autocorrelation,
  near 0 = strong positive autocorrelation, near 4 = strong negative
  autocorrelation.
- **Omnibus / Prob(Omnibus), Jarque-Bera / Prob(JB), Skew, Kurtosis**:
  joint and individual tests of residual normality.
- **Cond. No.**: the condition number of the design matrix; large values
  (rule of thumb >30, and certainly the 2000+ seen in the macrodata
  example) flag numerical instability, often from multicollinearity or
  very different variable scales.

## 6. Diagnosing problems: the four residual plots

1. **Residuals vs. Fitted** — checks linearity. Want a flat, structureless
   band of points around zero. A curved LOWESS trend line signals a
   missing nonlinear term.
2. **Normal Q-Q** — checks residual normality. Points should hug the
   45° reference line; systematic bowing away at the tails indicates
   skew or heavy tails.
3. **Scale-Location** (√|standardized residuals| vs. fitted) — checks
   homoscedasticity. A flat trend = constant variance; a rising/falling
   trend (funnel shape) = heteroscedasticity.
4. **Residuals vs. Leverage** (with Cook's distance) — flags influential
   outliers: points that are both extreme in x (**high leverage**) and
   poorly fit (**large residual**) exert outsized influence on β̂. Cook's
   distance combines both; values above ~0.5–1 warrant investigation.

## 7. Serial correlation, Durbin-Watson, ACF/PACF, differencing

When data are collected sequentially (e.g., quarterly macro data), nearby
observations are often correlated with each other, violating the
independence assumption.

- **Durbin-Watson (DW) statistic** tests for *lag-1* autocorrelation in the
  residuals: `DW ≈ 2(1 − ρ̂₁)`. DW well below 2 ⇒ positive autocorrelation;
  well above 2 ⇒ negative autocorrelation. Critical values come from
  published Durbin-Watson tables (indexed by `n` and number of
  regressors, `k`).
- **ACF (Autocorrelation Function)** shows the raw correlation between a
  series and its lagged versions, without controlling for intermediate
  lags. A slowly-decaying ACF is the signature of a trending
  (non-stationary) series.
- **PACF (Partial Autocorrelation Function)** shows the correlation at
  each lag *after* removing the effect of shorter lags — it's the tool
  used to identify autoregressive order.
- **First-order differencing** (`np.diff(series, n=1)`) — replacing each
  value with the change from the previous value — is a simple way to
  remove a trend and often resolves much of the serial correlation, at
  the cost of losing one observation and changing what the model
  estimates (a *rate-of-change* relationship rather than a *levels*
  relationship).

## 8. Model validation strategies

- **Train/test split**: fit on a random subset (e.g., 75%), evaluate the
  same coefficients' predictive performance on the held-out 25%. Similar
  performance (and similar coefficients) on both partitions is reassuring
  evidence the model generalizes.
- **Naïve baseline comparison**: compare the model's Mean Absolute Error
  (MAE) to that of a "predict the mean" naïve model. If the fitted model
  doesn't clearly beat guessing the mean every time, it isn't adding
  value.
- **Holdout MAE vs. training MAE**: a much larger error on held-out data
  than on training data is a red flag for overfitting (less of a concern
  with a single-predictor OLS model, but the habit generalizes to more
  complex models).

## 9. Limitations of this lab / what NOT to conclude

- **Simple linear regression only models one predictor.** Anything omitted
  from the model that also affects y will show up as bias in β̂ or as
  structure in the residuals (confounding). Multiple regression
  (Chapter 7 material) is the natural next step.
- **Correlation ≠ causation**, even with a significant p-value and high
  r². The macroeconomic example (investment vs. disposable income) is
  illustrative, not a causal claim — both series plausibly respond to a
  common set of business-cycle drivers.
- **Time-ordered data breaks the independence assumption** almost by
  construction. OLS with autocorrelated residuals still produces unbiased
  point estimates of β but the standard errors (and therefore p-values and
  confidence intervals) are unreliable — usually too small, giving false
  confidence. The lab shows differencing as a quick patch, but the
  chapter (and this lab) is explicit that a dedicated time-series model
  (AR/ARMA, Chapter 10 material) is the statistically correct tool once
  autocorrelation is confirmed.
- **Outlier handling is not automatic.** Cook's distance and leverage flag
  suspicious points, but deciding whether to drop, transform, or keep them
  requires domain judgment — a statistically "extreme" point may be the
  most important observation in the dataset (e.g., a recession quarter).
- **Diagnostic plots are visual, not hypothesis tests** (except
  Durbin-Watson, Omnibus, and Jarque-Bera, which are). Two people can
  reasonably disagree about whether a Q-Q plot "looks normal enough."
  Treat these as a checklist for further inquiry, not a pass/fail gate.
- **Small-sample behavior.** With `n` in the dozens (as in several
  synthetic examples), normality and homoscedasticity tests have low
  power — absence of evidence of a violation is not strong evidence of
  its absence.

## 10. What this lab CAN simulate successfully

- Recovering known ground-truth `β0, β1` from noisy synthetic data, which
  builds intuition for sampling variability in OLS estimates.
- Demonstrating, with controlled synthetic examples, exactly what each
  assumption violation *looks like* in a residual plot (skew, outliers,
  heteroscedastic "funnels," serial correlation) — because we know the
  ground truth, we can confirm the diagnostics are working correctly
  before applying them to real, messier data.
- Walking through a full real-data workflow (statsmodels macrodata) from
  raw scatter plot → fitted model → diagnostics → autocorrelation
  detection → differencing → re-fitting → train/test validation, which is
  representative of a real applied-regression workflow end to end.
- Comparing a fitted OLS model against a trivial naïve baseline, a
  practice that generalizes to any predictive-modeling project regardless
  of model complexity.

## 11. Where this lab intentionally stops short

- No multiple regression, regularization, or variable selection (planned
  as the "next chapter" topic in the source material).
- No formal treatment of heteroscedasticity-robust standard errors (e.g.,
  `HC3`) or Generalized Least Squares — the lab identifies the problem but
  doesn't fix it within the OLS framework.
- No full ARIMA/ARMA modeling of the autocorrelated series — that's
  flagged as a follow-up topic, not solved here.
- No causal inference framework (instrumental variables, RCTs, DAGs) —
  the lab is about *descriptive/predictive* linear regression, not causal
  estimation.
