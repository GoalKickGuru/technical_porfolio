# Background & Theory: Time Series Diagnostics

*Companion reading for the lab, skeleton, cheat sheet, template, and solutions notebooks in this bundle.*

## 1. What makes time-series data different

Most introductory statistics and machine learning assumes independent samples: each row of a dataset is unrelated to the others. A time series breaks that assumption on purpose. A time series is a sequence of measurements of the *same* quantity taken over time,

$$X = x_1, x_2, \dots, x_{t-1}, x_t$$

and because the samples are ordered and related, a value's position in time — its **lag** — becomes meaningful information. That relatedness is called **serial correlation**, or when a variable is compared to its own past, **autocorrelation**. It is both the reason time-series analysis needs its own toolkit and the reason that toolkit can be useful: if the past contains information about the future, that information can be modeled and forecast.

Time series can be univariate (one variable tracked over time, e.g. daily temperature) or multivariate (several variables tracked together, e.g. temperature, humidity, and rainfall). This lab bundle focuses on the univariate case, with cross-correlation as a bridge toward relating two series to each other.

## 2. The two goals of time-series analysis

1. **Identifying patterns** — is the series random, or does it have trend, seasonality, or cyclical structure?
2. **Forecasting** — using identified patterns to predict future values, always with an honest accounting of uncertainty (prediction intervals, error rates). No forecasting method is infallible, and results should be communicated with appropriate humility about model uncertainty.

Everything in this lab bundle is in service of goal 1: you cannot responsibly attempt goal 2 without first knowing whether a series is noise, trending, seasonal, or a mix, and whether it has been transformed into a form a model can actually use.

## 3. Mean, signal, and ergodicity

The sample mean of a series is $\bar{X} = \frac{1}{n}\sum_{t=1}^{n} x_t$. In time-series language the mean is often called the **signal** — the predictable component you're trying to isolate from the noise.

A key nuance: whether a single realization's statistics can be trusted to represent the whole underlying process depends on **ergodicity**.

- An **ergodic** process produces consistent statistical output regardless of when you sample it; the sample mean converges to the true population mean as sample size grows, and one long stretch is representative of the process as a whole.
- A **non-ergodic** process does not have this property — statistics computed from one phase of the process's output need not resemble another phase. A machine whose output drifts as parts wear down (until recalibration) is a physical example: measuring its behavior this month doesn't guarantee anything about its behavior next month, no matter how many samples you take this month.

Ergodicity matters practically because most classical time-series models (AR, MA, ARIMA, and their relatives) implicitly assume something close to ergodic, stationary behavior in the transformed series they're fit to. If the underlying process is not ergodic (e.g. it undergoes structural regime changes), no amount of differencing or data will make a single fixed model trustworthy across all regimes.

## 4. Variance and the white-noise baseline

Variance in a time series is the dispersion of values around the (possibly time-constant) mean. For a stationary process, the variance of the sample mean itself depends on the series' own autocorrelation structure:

$$\mathrm{Var}(\bar{X}) = \frac{\sigma^2}{n}\sum_{k=-(n-1)}^{n-1}\left(1 - \frac{|k|}{n}\right)\rho_k$$

When there is no autocorrelation — the **white-noise** case — this collapses to the familiar $\mathrm{Var}(\bar{X}) = \sigma^2/n$, and an average is a perfectly good "model" because there are no correlated errors to exploit.

**White noise** is formally a series with independent samples, constant (finite) variance, and zero mean. It is the null hypothesis of "nothing to model": if a raw series looks like white noise, there's no pattern to forecast beyond the mean; if a fitted model's *residuals* look like white noise, that's a good sign the model has extracted everything predictable and left only irreducible randomness behind.

### The Ljung-Box test

The Ljung-Box test formalizes the white-noise check as a hypothesis test:

- $H_0$: the data are independently distributed (no serial correlation across the tested lags)
- $H_a$: the data show serial correlation somewhere within the tested horizon

$$Q = n(n+2)\sum_{k=1}^{h}\frac{\hat\rho_k^2}{n-k} \;\sim\; \chi^2_h \text{ under } H_0$$

A large $Q$ (small p-value) rejects $H_0$; the test statistic follows a chi-square distribution and, by construction, weights more recent lags more heavily than distant ones. It can be applied either directly to raw data (to check whether the data itself is white noise) or to a fitted model's residuals (to check whether the model has captured all the exploitable structure). In `statsmodels`, `acorr_ljungbox()` implements this.

**Caveat:** a hypothesis test run at one lag/sample size can produce false positives or negatives at the stated significance rate (e.g. about 5% of the time at $\alpha=0.05$, even on genuine white noise). Pair it with a visual ACF check and, where practical, multiple lags or repeated samples rather than trusting a single p-value in isolation.

## 5. Autocorrelation (ACF) and partial autocorrelation (PACF)

For lag $k>0$, the (sample) autocorrelation function is:

$$r_k = \frac{\sum_{t=k+1}^{n}(y_t - \bar{y})(y_{t-k}-\bar{y})}{\sum_{t=1}^{n}(y_t-\bar y)^2}$$

The ACF measures **raw** correlation between a value and its lag-$k$ predecessor, without adjusting for the lags in between — this is why a strongly trending series shows a slowly-decaying ACF: correlation "leaks through" every intermediate lag. The **partial** autocorrelation function (PACF) corrects for this by measuring the *direct* relationship between $y_t$ and $y_{t-k}$ after netting out the effect of all shorter lags.

This distinction is the practical reason both plots matter together:

- The **PACF**'s cutoff point suggests a candidate **autoregressive (AR)** order — an AR(p) process typically shows a PACF that drops to (statistically) zero after lag $p$.
- The **ACF**'s cutoff/decay shape suggests a candidate **moving-average (MA)** order — an MA(q) process typically shows an ACF that drops to zero after lag $q$, while its PACF tails off more gradually.
- A slowly, near-linearly decaying ACF at low lags is itself diagnostic: it usually signals a non-stationary (trending) series that should be differenced before ACF/PACF are used for order selection at all.

As a rule of thumb, AR/MA orders much beyond about 5 are usually a sign of overfitting a short or noisy sample rather than genuine long-range structure, though the "right" order always depends on the analyst's judgment and the specifics of the process.

## 6. Differencing

A **first-order difference** is $Y_t' = Y_t - Y_{t-1}$ — a discrete numerical derivative, and a simple linear (low-pass) filter that removes a linear trend from the mean while passing through higher-frequency variation. It is implemented directly by `numpy.diff(x, n=1)` and removes one observation from the series (an unavoidable cost of numerical differencing).

Differencing is applied when the mean is not constant — e.g., a monotonically increasing or decreasing signal. It can be repeated if one difference is not enough, but **over-differencing** is a real risk: differencing a series more than it needs typically introduces spurious negative autocorrelation at low lags (most visibly, a strong negative spike at lag 1) rather than removing more genuine structure. Watching the ACF/PACF before and after each difference — and stopping once the trend/near-unit-root behavior is gone — helps avoid this.

For data with a repeating periodic pattern (e.g. monthly data with an annual cycle), a **seasonal difference** $Y_t' = Y_t - Y_{t-s}$ (with $s$ the period, e.g. 12) targets that periodicity directly; a first-order difference alone will often leave seasonal spikes in the ACF untouched.

When the *variance* — not just the mean — grows with the level of the series (multiplicative seasonality or heteroskedastic growth), a **log transform** is a common complement: it converts a growing-amplitude seasonal or trending pattern into something closer to a constant-amplitude additive one, which differencing can then act on more effectively.

## 7. Cross-correlation (CCF) and lead/lag relationships

The cross-correlation function extends autocorrelation to *two* different series, letting you check whether one series' current value is more related to the other series' past, present, or future. For two series $i,j$ at lag $k$:

$$\hat\rho_{i,j}(k) = \frac{\sum_{t=1}^{n-k}(x_{t,i}-\bar x_i)(x_{t+k,j}-\bar x_j)}{\sqrt{\sum_{t=1}^{n}(x_{t,i}-\bar x_i)^2}\sqrt{\sum_{t=1}^{n}(x_{t+k,j}-\bar x_j)^2}}$$

If the CCF peaks at a nonzero lag $k$, that's evidence one series **leads** the other by $k$ steps — practically useful for building regression or forecasting models where an early-warning variable (e.g. advertising spend) predicts a later outcome (e.g. sales). Once the lag is identified, `pandas.Series.shift()` is used to align the leading series so a same-timestep model can use it. As with autocorrelation, a dominant trend in both series can produce a large but practically uninformative CCF everywhere; differencing both series first (to strip the shared trend) before computing the CCF is standard practice, mirroring what's done for the ACF.

It's also worth separating **statistical significance** from **practical significance**: a CCF value can sit outside a computed confidence band and still be small enough (e.g., ~0.2) that it explains very little of the variance — worth noting, but not necessarily worth building a forecasting strategy around on its own. Pearson's correlation coefficient can supplement this by measuring the overall long-run linear association between two trending series, separate from the lag-by-lag CCF picture.

## 8. Stationarity

Stationarity (specifically, weak/covariance stationarity) is the single most important precondition for classical time-series forecasting models. A series is stationary if, for all $t$:

1. **Constant mean:** $E[X_t] = \mu$
2. **Constant variance:** $\mathrm{Var}[X_t] = \sigma^2$
3. **Autocorrelation depends only on lag distance, not on time itself:** $\mathrm{Cov}(X_{t_1}, X_{t_2})$ depends only on $t_2 - t_1$

Series with a trend or seasonality violate these by construction, since the trend/seasonal component makes the mean (and often the variance) time-dependent. Stationarity can be assessed in three complementary ways, each catching different problems:

- **Visual:** plotting the raw series, a seasonal decomposition (trend/seasonal/residual via `seasonal_decompose`), rolling mean/std, and grouped boxplots (e.g. by year) to see level and spread drifting over time.
- **Descriptive/hypothesis-test based on autocorrelation:** the Ljung-Box test applied to the raw series, since strong, dominant autocorrelation across many lags is itself a symptom of a non-stationary (typically trending) series.
- **Formal unit-root/stationarity hypothesis tests:**
  - **Augmented Dickey-Fuller (ADF):** $H_0$ = series has a unit root (non-stationary); a small p-value rejects $H_0$, i.e. is evidence *for* stationarity.
  - **KPSS:** $H_0$ = series is stationary; a small p-value rejects $H_0$, i.e. is evidence *for* non-stationarity — the **opposite** null hypothesis from ADF.

Because their nulls point in opposite directions, using ADF and KPSS together is more robust than either alone:

| ADF verdict | KPSS verdict | Interpretation |
|---|---|---|
| stationary | stationary | Strong evidence of stationarity |
| non-stationary | non-stationary | Strong evidence of non-stationarity — transform before modeling |
| stationary | non-stationary | Possibly trend-stationary — consider detrending rather than differencing |
| non-stationary | stationary | Ambiguous/borderline — inspect visually and consider one difference |

The general workflow this lab bundle practices is: **visualize → test → transform (difference / seasonal-difference / log) → re-test → inspect ACF/PACF of the transformed series → propose AR/MA order**. This is exactly the ladder that precedes ARIMA/SARIMA model fitting.

## 9. What this lab can simulate successfully

The datasets and synthetic simulations used across the lab, skeleton, template, and solutions notebooks were chosen to make specific, textbook-clean diagnostic signatures visible:

- **Pure white noise** (`np.random.normal`) — cleanly demonstrates a "nothing to model" ACF and passing Ljung-Box test.
- **A stationary AR(1)/AR(2) process** built by explicit recursion (`x_t = phi * x_{t-1} + eps_t`) — cleanly demonstrates a PACF that cuts off at the true order, since the data-generating process is exactly an AR model.
- **A random walk** (`np.cumsum` of noise) — cleanly demonstrates classic non-stationarity (near-unit-root ACF decay, ADF failing to reject a unit root) and shows that a single first difference restores white noise, exactly matching the textbook ARIMA(0,1,0) case.
- **A variance regime shift and a mean regime shift** — cleanly demonstrates non-ergodic-flavored behavior and why full-sample statistics can mislead.
- **A synthetic leading-indicator pair** (one series shifted and scaled to build the other) — cleanly demonstrates a CCF with an unambiguous, known-ground-truth peak lag, and how shifting resolves it.
- **Macro-economic series with a genuine long-run trend** (`realinv`, `realdpi` from `statsmodels`) — a real (if small and old) dataset showing the ACF-dominated-by-trend problem and how first-differencing exposes shorter-range structure.
- **Monthly airline passenger counts** — a real dataset with both trend and clear multiplicative seasonality, ideal for demonstrating why differencing alone is often not enough and why log + seasonal differencing is a standard combination.

Because most of these series are *constructed* to have a known ground truth (a known AR order, a known lag, a known transform that achieves stationarity), the lab is well suited for **building intuition and verifying that a diagnostic method correctly recovers a known answer** — which is exactly what you want before trusting the same method on a real series where the answer is unknown.

## 10. Limitations — what this lab does *not* cover, and what to watch for

- **No model fitting yet.** This bundle stops at diagnostics: what transform to apply and what AR/MA order to try. It does not fit or evaluate ARIMA/SARIMA models, compute forecasts, or assess forecast accuracy (e.g. out-of-sample error, prediction intervals). That is the natural next step (an "ARIMA models" follow-up lab).
- **Univariate focus, with only elementary multivariate tools.** Cross-correlation is included, but genuinely multivariate methods (vector autoregression, cointegration, Granger causality with full inference, multiple regression with time-series errors) are out of scope here.
- **Small, well-behaved sample sizes.** The synthetic series (300–1000 points) and the classic datasets used are small by modern standards. Diagnostics that look clean at this scale (e.g. a clearly cutting-off PACF) can be noisier and harder to read on very large or very short real-world series, or on series sampled at irregular intervals — this bundle assumes fixed-interval sampling throughout, consistent with the scope described in the source material.
- **Idealized data-generating processes.** The synthetic AR/seasonal/random-walk series are generated from clean, known formulas. Real data is rarely this cooperative — it can mix multiple regimes, have missing values, outliers, measurement error, structural breaks, and non-linear dynamics that none of the linear tools here (ACF/PACF/Ljung-Box/ADF/KPSS/CCF) are designed to detect. A clean diagnostic verdict on synthetic data is a **best case**, not a guarantee that the same method will be equally decisive on messy real-world data.
- **Hypothesis tests have known blind spots.** ADF has relatively low power against near-unit-root-but-stationary alternatives (it can fail to reject non-stationarity even when the series is technically stationary but close to a unit root); KPSS can over-reject stationarity in the presence of strong positive autocorrelation if the bandwidth/lag settings aren't well chosen. Neither test "proves" its alternative — they provide evidence to weigh alongside the visual checks, not a final verdict on their own.
- **Ergodicity is not directly testable from one sample.** The lab illustrates the *concept* with a constructed regime-shift series where the ground truth is known, but in practice you cannot definitively prove or disprove ergodicity from a single realization — you can only look for symptoms (drifting statistics across sub-samples) that make you suspicious enough to seek additional context (domain knowledge, multiple realizations, or a change-point analysis).
- **Manual CCF band is an approximation.** The confidence bands used in the CCF function are a large-sample z-score approximation ($\pm z/\sqrt n$), not an exact finite-sample confidence interval; treat CCF "significance" as a rough guide, especially with the CCF's high multiple-comparison risk when scanning many lags at once.
- **Seasonal period must be known or guessed.** The reusable template assumes you already know (or can reasonably guess) the seasonal period (e.g. 12 for monthly data); detecting an unknown seasonal period automatically (e.g. via spectral analysis) is not covered here.

## 11. Where to go next

Once a series has been diagnosed (transform identified, approximate stationarity achieved, candidate AR/MA orders proposed from the ACF/PACF), the natural next steps are:

1. Fit candidate AR/MA/ARMA/ARIMA models (and SARIMA if seasonality remains) using the identified `(p, d, q)` — and `(P, D, Q, s)` for seasonal terms.
2. Check the *residuals* of the fitted model with the same white-noise toolkit used here (ACF, Ljung-Box) — a well-specified model should leave residuals indistinguishable from noise.
3. Compare candidate models with information criteria (AIC/BIC) and out-of-sample validation, and quantify forecast uncertainty with prediction intervals rather than relying on point forecasts alone.
4. For genuinely related pairs of series (beyond simple lead/lag), consider multivariate extensions (VAR, cointegration testing) rather than single-equation regression with a shifted leading indicator.
