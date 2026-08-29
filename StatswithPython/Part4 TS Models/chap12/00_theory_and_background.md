# Multivariate Time Series: Theory, Background, and Lab Scope

This document is the companion reading for the lab notebooks in this pack
(`01_extended_lab_skeleton.ipynb`, `02_extended_lab_solutions.ipynb`,
`03_cheat_sheet.ipynb`, `04_reusable_template.ipynb`). It expands on the
material in *Building Statistical Models in Python*, Chapter 12
("Multivariate Time Series"), covering cross-correlation, ARIMAX, and VAR
modeling — and is explicit about what these models can and cannot do.

---

## 1. Why multivariate time series?

A univariate model (AR, MA, ARMA, ARIMA, SARIMA) uses only a variable's own
past to forecast its future. That's a strong restriction: in the real world,
almost every series of interest is influenced by other measured processes —
temperature affects wind speed, consumption affects investment, one stock
affects another. A **multivariate time series** is a vector-valued sequence

```
X_t = [x_t,0, x_t,1, ..., x_t,N]
```

where each `x_t,i` is one univariate "covariate" series observed at the same
time steps. The two questions this lab answers are:

1. **Are these series related, and at what lag?** → cross-correlation.
2. **How do I build a forecasting model that uses that relationship?** →
   ARIMAX (one endogenous variable, several exogenous drivers) and VAR
   (several variables, all treated symmetrically, each explained by lags of
   all the others).

---

## 2. Cross-correlation (CCF)

### 2.1 Definition

For two (stationary) series `X` and `Y`, the sample cross-correlation
function at lag `k` is

```
rho_k(X, Y) = sum_{t=1}^{n-k} (X_t - X_bar)(Y_{t-k} - Y_bar)
              --------------------------------------------------
              sqrt(sum (X_t - X_bar)^2) * sqrt(sum (Y_t - Y_bar)^2)
```

`Y` is said to *lag* `X` by `k` steps (equivalently `X` *leads* `Y`). When
`X = Y` this reduces to the ordinary autocorrelation function (ACF).

`scipy.signal.correlate` computes the full (unnormalized) cross-correlation
in one call; we normalize by `std(X) * std(Y) * n` to get values in
`[-1, 1]`, matching the CCF definition above.

### 2.2 Significance band

Just like the ACF, a cross-correlation is treated as "significantly
different from zero" when

```
|rho_k(X, Y)| > 1.96 / sqrt(N)
```

for a 95% confidence level (use 1.645 for 90%, 2.576 for 99%), where `N` is
the sample size. This is an approximation that assumes the series are
white noise under the null — in practice, if `X` or `Y` has its own
autocorrelation (almost always true), the *effective* sample size is
smaller than `N`, so this band is optimistic (too narrow). Treat it as a
rough guide, not a rigorous test — see §6 (limitations).

### 2.3 Reading a CCF plot

* A **single spike** at lag `k > 0` (positive side) → `X` at time `t-k`
  predicts `Y` at time `t`; useful as an exogenous predictor with that lag.
* **Oscillating, decaying peaks** at multiples of some period → both series
  share a seasonal/periodic structure (e.g., a daily cycle sampled hourly
  will show CCF peaks near lag ±24, ±48, ...). This is what you'll see
  between wind speed and temperature in this lab: an oscillation with a
  ~12–24 hour period, reflecting the shared diurnal cycle rather than one
  variable causing the other at a single sharp lag.
* **A CCF that looks like two AR processes correlated only at lag 0** →
  the relationship may be spurious/contemporaneous and driven by a shared
  trend; check stationarity (§4) before trusting the CCF at all.

### 2.4 Cross-correlation is not causation

A strong CCF at lag `k` tells you `X_{t-k}` and `Y_t` move together; it does
not tell you `X` *causes* `Y`. Both could be driven by a third, unobserved
factor (a classic case: two trending series will cross-correlate at nearly
every lag purely because of the shared trend). Always check stationarity
before interpreting a CCF, and consider Granger-causality tests (§7) as a
(still imperfect) way to talk about lead/lag structure more formally.

---

## 3. ARIMAX

### 3.1 From ARIMA to ARIMAX

Recall the (differenced) ARIMA(p, d, q) equation:

```
y'_t = c + sum_{i=1}^{p} phi_i * y'_{t-i} + e_t + sum_{j=1}^{q} theta_j * e_{t-j}
```

ARIMAX adds a linear combination of `r` **exogenous** regressors `X`:

```
y'_t = c + sum_i phi_i y'_{t-i} + e_t + sum_j theta_j e_{t-j} + sum_{k=1}^{r} beta_k X_{t,k}
```

The `X_{t,k}` terms are typically **lagged versions of covariate series**
(chosen via the CCF, §2), not the covariate at time `t` itself — using the
covariate's value *at* time `t` would require knowing its future value at
forecast time, which usually isn't available (see §6.1).

### 3.2 Choosing which lags to include

1. Compute the CCF between the target (endogenous) series and each candidate
   exogenous series, out to some reasonable number of lags (e.g., 48 hours
   for hourly data — two full daily cycles).
2. Mark the lags whose |CCF| exceeds the significance band.
3. Create a lagged column for each such lag with `series.shift(-lag)` (a
   *negative* shift moves future values backward so that the exogenous
   column at row `t` holds `x_{t-lag}` — see the worked example in the lab).
4. Collect all lagged columns into an exogenous matrix `X` and fit
   `pmdarima.auto_arima(y, X, ...)`, or `statsmodels`
   `SARIMAX(y, exog=X, order=(p,d,q))`.

### 3.3 Feature selection / multicollinearity

Multiple lags of the *same* underlying series (e.g., `temp_lag_13`,
`temp_lag_24`, `temp_lag_37`) are themselves autocorrelated, so they are
often highly collinear with each other. Symptoms: inflated standard errors,
unstable or sign-flipping coefficients, high condition number. Two common
remedies used in this lab:

* **Iterative p-value elimination** — drop the exogenous term with the
  largest p-value, refit, repeat until all remaining terms are significant
  at your chosen level. (This is what the book does; it's simple but
  greedy and can remove a jointly-informative pair of terms one at a time.)
* **Variance Inflation Factor (VIF)** — a more direct multicollinearity
  diagnostic (`statsmodels.stats.outliers_influence.variance_inflation_factor`);
  the extended lab adds this as a cross-check.

### 3.4 What the coefficients mean

Each `beta_k` in the SARIMAX summary is the expected change in `y_t` for a
one-unit increase in that particular *lagged* exogenous column, holding the
ARMA structure fixed. Because the columns are lags of the same variable at
different offsets, don't interpret each coefficient as if it came from an
independent experiment — read them jointly.

---

## 4. Stationarity (needed by both ARIMAX and VAR)

Both models assume (weak) stationarity: constant mean, constant variance,
and autocovariance that depends only on lag, not on absolute time. A trending
or seasonally-drifting series must be **differenced** (or otherwise
detrended) before modeling.

* **Visual check**: an ACF that decays very slowly (many significant lags,
  no clear cutoff) is a classic trend/unit-root signature.
* **Augmented Dickey-Fuller (ADF) test**: `statsmodels.tsa.stattools.adfuller`.
  Null hypothesis: a unit root is present (series is non-stationary). A
  small p-value (e.g. < 0.05) lets you reject the null and treat the series
  as stationary.
* **Fix**: first-difference (`np.diff` / `series.diff()`) is the usual first
  attempt; re-run ADF on the differenced series to confirm.

For VAR specifically: **every** variable in the system must be stationary
(or the whole system treated with a cointegrated/VECM approach, which is out
of scope here — see §6.3).

---

## 5. VAR (Vector Autoregression)

### 5.1 The model

Unlike ARIMAX, VAR treats every variable symmetrically: there is no
endogenous/exogenous split. Each variable is regressed on lagged values of
**itself and every other variable in the system**. For the two-variable
VAR(1) case:

```
y_t1 = (1 - phi11) mu1 - phi12 mu2 + phi11 y_{t-1,1} + phi12 y_{t-1,2} + e_t1
y_t2 = -phi21 mu1 + (1 - phi22) mu2 + phi21 y_{t-1,1} + phi22 y_{t-1,2} + e_t2
```

or in compact zero-mean matrix form: `y_t = Phi_1 y_{t-1} + e_t`.

Because VAR is fully autoregressive (no external drivers), **no future
values of any variable are needed to forecast** — a genuine advantage over
ARIMAX, whose exogenous terms are really just *lagged* covariates chosen so
that forecasting never needs a future covariate value either. The
philosophical difference is that in VAR there's no assumption that any one
variable is exogenous / causally prior to the others.

### 5.2 Model-building workflow (the six steps used in the lab)

1. **Visual inspection** — line plot every series; look for trend/seasonality.
2. **Stationarity + order selection** — ADF test each series; difference if
   needed; use ACF/PACF (Yule-Walker) of each *differenced* series to get an
   initial sense of AR order per variable.
3. **Cross-correlation between candidate inputs and the target** — decide
   whether any series should be shifted so that its peak correlation sits at
   lag 0 (this only matters if you want to reduce required VAR order; it is
   not a VAR requirement).
4. **Grid search over (p, q)** — fit `statsmodels.tsa.statespace.varmax.VARMAX`
   for a range of `(p, q)` and select by AIC/BIC, then sanity-check
   coefficient significance (too many insignificant terms → drop to a lower
   order).
5. **Test forecast** — hold out the last `h` points, forecast, and compare to
   actuals with a plotted confidence interval.
6. **Production forecast** — forecast beyond the end of the observed sample.

### 5.3 Reading a VARMAX summary

The summary is organized as one regression "block" per equation (one per
variable). `L1.varname` coefficients under "Results for equation X" describe
how a one-unit change in `varname` one period ago shifts `X` today. The
**error covariance matrix** at the bottom shows how much the residuals of
different equations move together — nonzero, significant cross-covariances
mean there is instantaneous (same-period) co-movement your model's lagged
terms don't capture.

### 5.4 Undoing differencing

If you difference before fitting, remember the model produces forecasts of
the *differenced* series. Reconstructing level forecasts requires
accumulating (cumulative-summing) the differenced forecasts starting from the
last known level, or picking a transform (e.g., log then difference) that
back-transforms with a closed form (exponentiate a cumulative sum of
log-differences).

---

## 6. Limitations — what this lab does *not* solve

Be explicit with yourself (and anyone reading your results) about these:

### 6.1 ARIMAX needs *future* exogenous values, or lagged ones only

The classical ARIMAX/SARIMAX formulation assumes you *have* the exogenous
regressor's value at the forecast horizon. If you use only lagged exogenous
terms (as we do here, chosen via CCF), you sidestep this — but only up to
the smallest lag used. If your smallest significant lag is 2 and you want to
forecast 5 steps ahead, you don't have the lagged exogenous values for steps
3–5 either, and you must either forecast the exogenous series itself
(compounding error) or restrict your horizon to the smallest lag used.

### 6.2 Both CCF and the 1.96/√N band assume near-white-noise inputs

As noted in §2.2, the standard significance band understates the true
sampling variability when the series are autocorrelated (which time series
almost always are). Treat CCF "significance" as a screening heuristic, not
a hypothesis test with a trustworthy p-value. Always sanity-check any lag
you keep by looking at whether it survives in the fitted model's own
coefficient p-values.

### 6.3 VAR assumes stationarity, not cointegration

If two non-stationary series share a long-run equilibrium relationship
(cointegration), naively differencing and fitting VAR on the differences
throws away that long-run information. The correct tool there is a **Vector
Error Correction Model (VECM)** — out of scope for this lab, but worth
knowing the name for when you hit trending macro/financial data that looks
cointegrated (test with `statsmodels.tsa.vector_ar.vecm.coint_johansen`).

### 6.4 Linear models only capture linear co-movement

Both ARIMAX and VAR are linear in the (possibly lagged/transformed)
regressors. Nonlinear relationships (regime switches, threshold effects,
volatility clustering) will not be captured — you'd reach for threshold
VAR, Markov-switching models, or nonlinear/ML approaches (e.g., gradient
boosted trees on engineered lag features, or recurrent/temporal
architectures) instead.

### 6.5 Small-sample / high-order interactions

VAR parameter count grows as `(#variables)^2 * order`, so with few
variables and modest order you can still overfit a few hundred
observations, especially with quarterly/annual macro data. Watch AIC/BIC
selection and out-of-sample error, not just in-sample fit.

### 6.6 Forecast intervals assume Gaussian, homoskedastic innovations

The `summary_frame` confidence intervals from `SARIMAX`/`VARMAX` come from
the model's assumed Gaussian innovations. Heavy-tailed or heteroskedastic
residuals (check Jarque-Bera and heteroskedasticity tests in the summary
output) mean the printed intervals are approximate at best.

---

## 7. Extensions used in the lab beyond the source chapter

To get more mileage out of the same two datasets, the extended lab and
solutions notebooks add a few standard practices that the book chapter
doesn't cover in depth:

* **Train/test (out-of-sample) evaluation with a proper holdout**, rather
  than fitting and forecasting on the same data used for lag selection.
* **Variance Inflation Factor (VIF)** as a second multicollinearity
  diagnostic alongside p-value elimination.
* **Granger causality tests** (`statsmodels.tsa.stattools.grangercausalitytests`)
  to add statistical language to "leading indicator" claims from the CCF.
* **Impulse response functions (IRF)** from the fitted VAR, to show how a
  one-time shock to one variable propagates through the system over time —
  one of the most common real-world uses of VAR (e.g., "what happens to
  investment if consumption jumps by one unit today?").
* **Rolling-origin (walk-forward) cross-validation** sketch for time series,
  since a single train/test split can be misleading for short series.

---

## 8. What this kind of lab *can* successfully simulate

Given the linear, Gaussian-innovation assumptions above, this workflow is a
good fit for:

* Any **short-to-medium-horizon forecasting** problem where recent history
  (yours and correlated series') is genuinely informative — weather-driven
  demand, short-term energy/wind-power forecasting, quarter-to-quarter
  macroeconomic dynamics, sensor/IoT streams with known physical coupling.
* Problems where you want an **interpretable** model (coefficients with
  units and confidence intervals) rather than a black box — VAR/ARIMAX
  summaries are directly readable by a domain expert.
* **Diagnostic / relationship-discovery** work: "does A lead B, and by how
  much?" even before you commit to a specific forecasting model.
* Small numbers of variables (VAR) — 2 to ~6 series is the comfortable
  range before parameter count and interpretability both degrade.

It is a poor fit for:

* Long-horizon forecasting where exogenous variables must themselves be
  forecast forward (compounding uncertainty invisibly).
* Highly nonlinear, regime-switching, or structurally-breaking systems
  (financial crises, policy shocks) — VAR/ARIMAX will systematically
  under/over-shoot around such events, exactly as seen with the 2008
  recession example in the book's forecast plot.
* Systems with many (dozens+) of candidate covariates, where lag selection
  by hand via CCF becomes impractical — that's the regime for regularized
  regression (e.g., LASSO on a large lag-feature matrix) or dedicated
  high-dimensional VAR estimators.
