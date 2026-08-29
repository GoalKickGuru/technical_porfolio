# Background Theory: Survival Analysis

*Companion document to the survival-analysis lab notebooks (Kaplan-Meier, Exponential, Cox Proportional
Hazards), based on Huy H., "Building Statistical Models in Python" (2023), Ch. 13–14.*

## 1. What survival analysis is for

Survival analysis (also called time-to-event, reliability, or duration analysis depending on the field)
studies the time until an event of interest occurs — death, mechanical failure, customer churn,
recidivism, etc. What sets it apart from ordinary regression is **censoring**: for some subjects, the
study ends, or the subject drops out, before the event happens. Their true event time is unknown, but
they still contribute information ("this person survived at least this long").

## 2. Censoring

| Type | Definition | Example |
|---|---|---|
| Left censoring | The event happened before observation began | A student already past puberty onset when the study starts |
| Right censoring | The event has not happened by the time observation ends | A patient still alive when a 5-year study concludes |
| Interval censoring | The event is known to have happened between two check-ins, exact time unknown | HIV test negative last year, positive this year |

Right censoring is further split into:
- **Type I** — study ends at a fixed calendar time; subjects still event-free at that point are censored.
- **Type II** — study ends after a fixed *number* of events occurs, regardless of elapsed time.

All three lab notebooks assume **non-informative right censoring**: the reason a subject is censored
(study ended, moved away) is unrelated to their underlying risk of the event. If censoring were
*informative* (e.g., sicker patients are more likely to drop out), every method below would be biased.

For subject *i*, let *T_i* be the true event time and *C_i* the censoring time. We observe
`Y_i = min(T_i, C_i)` and a status indicator `δ_i = 1` if the event was observed, `0` if censored.

## 3. Survival function, hazard function, hazard ratio

- **Survival function** `S(t) = P(T > t)` — probability of surviving past time *t*. `S(0) = 1`,
  `S(∞) = 0`, and it is non-increasing.
- **Hazard function** `h(t) = lim_{Δt→0} P(t < T ≤ t+Δt | T > t) / Δt` — the instantaneous event rate at
  time *t*, given survival to *t*. Think of it as the probability of the event happening "right now,"
  conditional on not having happened yet.
- **Cumulative hazard** `H(t) = ∫₀ᵗ h(u) du`, related to survival by `S(t) = exp(−H(t))`.
- **Hazard ratio (HR)** — the ratio of hazards between two groups (or per unit change in a covariate).
  HR = 3 means one group has 3× the instantaneous risk of the event at any given moment, *given* it has
  survived that long.

## 4. Kaplan-Meier estimator (non-parametric)

$$\\hat S(t) = \\prod_{i:\\, t_i \\le t} \\frac{n_i - d_i}{n_i}$$

where `n_i` is the number at risk just before time `t_i` and `d_i` is the number of events at `t_i`. The
estimator is a product of conditional survival probabilities at each observed event time — hence the
characteristic step-function shape, dropping only at times when an event actually occurred.

**Why it matters:** Kaplan-Meier makes *no assumption* about the shape of the hazard or survival curve.
It is the standard first pass at any survival dataset, and the standard baseline every parametric or
regression model should be checked against visually.

**What it cannot do:** it does not use covariates beyond simple group splits, and it gives no single
number ("the effect of age") — only a curve per subgroup, plus optional pairwise/multi-group log-rank
tests for whether curves differ.

## 5. Exponential model (parametric)

The exponential distribution arises from a **Poisson process** with constant rate `λ`: events occur
independently over time at rate `λ = Y/t`. Its density, CDF, and survival function are:

$$f(t) = \\lambda e^{-\\lambda t}, \\quad F(t) = 1 - e^{-\\lambda t}, \\quad S(t) = e^{-\\lambda t}$$

Because `h(t) = λ` is **constant**, the cumulative hazard is linear: `H(t) = λt`, and `S(t) = e^{-H(t)}`.
`λ` is estimated by maximum likelihood from the observed durations and event indicators.

**Why it matters:** it is the simplest parametric survival model, provides a closed-form hazard ratio
between groups (`λ_A / λ_B`), and is a natural bridge from non-parametric to regression-based (Cox)
survival modeling.

**What it cannot do well:** the constant-hazard assumption is strong and frequently wrong — most real
hazards rise or fall over time (e.g., cancer risk generally worsens with stage/time; mechanical failure
often follows a "bathtub curve"). Always compare an exponential fit against Kaplan-Meier visually and
against richer parametric families (Weibull, log-normal, log-logistic) by AIC before trusting it, as done
in the extended lab notebook.

## 6. Cox Proportional Hazards model (semi-parametric)

$$h(t) = h_0(t)\\, e^{(\\beta_1 X_1 + \\beta_2 X_2 + \\dots + \\beta_n X_n)}$$

`h_0(t)` is an unspecified ("nonparametric") baseline hazard, and the covariates act **multiplicatively**
on it through `exp(βX)`. Because `h_0(t)` is never estimated in closed form (Cox's partial-likelihood
trick sidesteps it), the model is "semi-parametric": parametric in the covariate effects, non-parametric
in the baseline shape.

`exp(β_j)` is the **hazard ratio** for a one-unit increase in covariate `X_j`, holding all other
covariates fixed.

### Assumptions (all three matter — none are automatically satisfied)

1. **Proportional hazards** — the hazard ratio between any two covariate levels stays *constant* over the
   whole follow-up period. This is testable via Schoenfeld residuals (`check_assumptions()` in
   lifelines) and is **not checked anywhere in the original book chapter** — the extended lab adds this
   step explicitly.
2. **Independence of survival across subjects** — one subject's risk doesn't depend on another's (no
   contagion, shared shocks, or clustering effects left unmodeled).
3. **Non-informative censoring** — censoring is unrelated to underlying risk (see Section 2).

### Workflow used in the lab (following the NIH-style sequence in the book)

1. State the null hypothesis per covariate (no effect on survival).
2. Estimate a baseline Kaplan-Meier curve to understand the raw data.
3. Run a log-rank test to confirm groups differ before bothering to model covariates.
4. Fit the Cox model; read `coef`, `exp(coef)`, confidence intervals, p-values.
5. Predict survival curves for individuals (train and holdout) from their covariate values.

## 7. Comparing the three models

| | Kaplan-Meier | Exponential | Cox PH |
|---|---|---|---|
| Parametric? | No | Fully parametric | Semi-parametric |
| Covariates | No (subgroup-only) | No (fit per group) | Yes, multivariate |
| Hazard shape assumption | None | Constant | None for baseline; *proportional* across covariates |
| Typical use | Descriptive, quick group comparison | Simple parametric summary, simulation | Risk-factor analysis with interpretable hazard ratios |

## 8. Limitations of this lab, and what it can/cannot successfully demonstrate

**What this type of lab is good at simulating/teaching:**
- The mechanics and interpretation of all three model families on real, if small, public datasets.
- The *workflow* of survival analysis: censoring-aware estimation → group comparison → regression →
  prediction.
- How to read and communicate a hazard ratio, a survival curve, a log-rank p-value, and a concordance
  index.
- The value of comparing a fitted parametric shape against the non-parametric Kaplan-Meier baseline
  rather than trusting it blindly.

**What it cannot reliably demonstrate, and why:**
- **Causal inference.** The Stanford heart transplant `transplant` variable is not randomized — patients
  who die early cannot receive a transplant, so comparing "transplant" vs. "no transplant" as a fixed
  baseline covariate suffers from immortal-time bias. The book/lab's Cox hazard ratio for `transplant`
  should be read as an association, not a causal treatment effect, unless the covariate is re-modeled as
  time-varying or a formal causal method (e.g., landmark analysis, IPTW) is used.
- **Rigorous assumption checking.** The book chapter never tests the proportional-hazards assumption,
  the single assumption most likely to be violated in practice. The extended lab adds this check, but a
  sample this small (n=90–171) has limited power to detect real violations — a "pass" is weak evidence,
  not proof.
- **Generalization beyond the sample.** Both datasets are small, single-site, and decades old (1960s–70s
  clinical data). Coefficients, medians, and hazard ratios are specific to these cohorts and should not
  be treated as universal clinical facts.
- **Model robustness.** The lab uses a single train/holdout split with no cross-validation or bootstrap
  resampling of confidence intervals; reported concordance and coefficients will have non-trivial
  sample-to-sample variability that a single run does not reveal.
- **Handling of time-varying covariates or competing risks.** All three models here treat covariates as
  fixed at baseline and assume a single event type. Real applications (a patient who can die *or* be
  transplanted *or* be lost to follow-up, with transplant status itself changing mid-study) often need
  `CoxTimeVaryingFitter` or competing-risks methods, which are outside the scope of this lab.

**Bottom line:** treat this lab as building fluency with the mechanics and honest interpretation of
survival models — not as a template for drawing clinical or causal conclusions from small, historical,
non-randomized datasets.

## 9. Further reading
- lifelines documentation: https://lifelines.readthedocs.io/en/latest/index.xhtml
- Cox PH modeling steps referenced in the book follow the outline at:
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7876211/
