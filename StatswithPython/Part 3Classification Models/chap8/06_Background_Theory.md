# Background Theory: Discrete Response Regression Models

*Companion reading for the Probit/Logit, Multinomial Logit, Poisson, and Negative Binomial lab
notebooks. This document explains the "why" behind each model, where the math comes from, what each
model assumes, where those assumptions break down, and what kinds of real-world problems each model
can (and can't) be trusted to simulate or predict.*

---

## 1. Why not just use linear regression for everything?

Ordinary least squares (OLS) regression assumes:
- the response variable is continuous and (conditional on the predictors) approximately normal,
- errors have constant variance (homoscedasticity),
- predictions are unbounded — a line can output any real number.

None of this holds for the outcomes in this lab:

| Outcome type | Example | Why OLS fails |
|---|---|---|
| Binary (0/1) | admitted / not admitted | predictions can fall outside [0,1]; errors can't be normal with a 2-point response; variance of a 0/1 variable depends on its mean (heteroscedastic by construction) |
| Unordered categorical (3+ levels) | species, product category | there's no meaningful "distance" between category codes (species 2 is not "more" than species 1) |
| Count (0, 1, 2, ...) | rentals per week, claims per policy | predictions can go negative; variance typically grows with the mean; discreteness invalidates the normal-error assumption |

Each family of models in this lab exists to fix a specific one of these violations by choosing an
appropriate **probability distribution** for the response and an appropriate **link function**
connecting the linear predictor $z = \beta_0 + \beta_1 x_1 + \dots$ to the mean of that distribution.
This is the core idea of the **Generalized Linear Model (GLM)** framework, of which logit, probit,
Poisson, and negative binomial regression are all special cases.

---

## 2. Logit and Probit Regression (Binary Outcomes)

### 2.1 Derivation

Start from the naive linear probability model:
$$P(y=1 \mid x) = \beta_0 + \beta_1 x + \epsilon$$
This has two problems: predicted probabilities can fall outside $[0,1]$, and the variance of a
Bernoulli response ($p(1-p)$) is not constant, violating homoscedasticity.

**Step 1 — model the odds instead of the probability.** The odds $\frac{P}{1-P}$ range over
$[0, \infty)$, which is closer to what a linear model can produce, but still can't go negative.

**Step 2 — model the log-odds ("logit").** $\ln\frac{P}{1-P}$ ranges over $(-\infty, \infty)$, which
now matches the range of a linear predictor exactly:
$$\ln\frac{P}{1-P} = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n = z$$

Solving for $P$ gives the **logistic (sigmoid) function**:
$$P = \frac{e^z}{1+e^z} = \frac{1}{1+e^{-z}}$$

This is the CDF of the **logistic distribution**. An equally valid alternative is to use the CDF of
the **standard normal distribution**, $\Phi(z)$, which gives the **probit** model instead. Both
functions are S-shaped, both map $(-\infty,\infty) \to (0,1)$, and in practice they produce very
similar fitted probabilities — differing mainly in the tails and in how coefficients are scaled.

### 2.2 Estimation

Both models are fit via **maximum likelihood estimation (MLE)**, not least squares, because the
likelihood of a Bernoulli outcome is a product of $P^{y}(1-P)^{1-y}$ terms, and the log-likelihood of
this is what's actually maximized numerically (iteratively, since there's no closed-form solution
like OLS's normal equations).

### 2.3 Interpreting coefficients

- **Logit**: $e^{\beta_j}$ is the multiplicative change in the **odds** of $y=1$ per one-unit increase
  in $x_j$, holding other variables fixed. This makes logit coefficients directly interpretable as
  odds ratios — a major reason it's preferred in epidemiology, credit scoring, and social science.
- **Probit**: coefficients are in "z-score" units of a latent normal variable and are not directly
  interpretable as odds ratios or probabilities. Use **marginal effects**
  ($\partial P/\partial x_j$, typically evaluated at the mean of the data) to get an
  interpretable "percentage-point change in probability" statement.

### 2.4 Assumptions and limitations

- **Independence of observations.** Logit/probit assume each observation's outcome is independent
  given the predictors. Clustered data (repeated measures, panel data, students within schools)
  violates this and requires clustered standard errors, mixed-effects/hierarchical extensions, or
  GEE — not covered in this lab.
- **Correct functional form / linearity in the log-odds.** The model assumes the *log-odds* (not the
  probability) is a linear function of the predictors. Curvature in the true relationship needs
  polynomial terms, splines, or interaction terms — the model won't warn you if this is violated.
- **No perfect separation.** If a predictor (or combination of predictors) perfectly predicts the
  outcome, MLE does not converge to finite estimates — coefficients and standard errors blow up. This
  is a common failure mode with small, clean datasets (as in the book's 31-row admissions example).
- **Sensitivity to sample size and class balance.** With small or highly imbalanced samples,
  coefficient estimates are unstable and standard errors are unreliable, even though `statsmodels`
  will still happily print a summary table.
- **No causal claims for free.** Logit/probit models correlational relationships. Interpreting a
  coefficient as a causal effect requires the same assumptions (no omitted confounders, no reverse
  causality, no selection bias) needed for causal inference from any observational regression.

### 2.5 What this model family is good for / not good for

**Good for:** binary classification and probability estimation problems where you have a
theoretically justified, mostly linear (in log-odds) relationship — credit default risk, marketing
response modeling, medical diagnosis screening, churn prediction, A/B test outcome analysis.

**Not good for / be cautious with:** highly nonlinear decision boundaries (consider tree-based
models or neural nets instead, at the cost of interpretability), extremely rare-event outcomes
without correction (e.g., 1-in-100,000 events — consider rare-events logistic regression
corrections), or settings with strong temporal/spatial autocorrelation without an appropriate
extension.

---

## 3. Multinomial Logit (3+ Unordered Categories)

### 3.1 Derivation

Multinomial logit generalizes binary logit by picking a **baseline category** $k_0$ and estimating
one log-odds equation per remaining category:
$$\ln\frac{P(y=k)}{P(y=k_0)} = \beta_{0k} + \beta_{1k}x_1 + \dots, \quad k \ne k_0$$

Exponentiating and normalizing so probabilities sum to 1 across all $K$ categories gives the
**softmax function**:
$$P(y=k) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

### 3.2 A key structural assumption: IIA

Multinomial logit relies on the **Independence of Irrelevant Alternatives (IIA)** assumption: the
relative odds of choosing category $A$ over category $B$ do not depend on what other categories are
available. This is a strong assumption. The classic illustrative failure is the "red bus / blue bus"
problem: if people are indifferent between a red bus and a blue bus (perfect substitutes), adding a
blue bus option shouldn't change the odds of car vs. (either) bus 2-to-1 — but naive multinomial logit
can predict an implausible shift. When your categories include near-substitutes, consider nested
logit or mixed logit models instead (not covered in this lab).

### 3.3 Interpreting coefficients

Exponentiated coefficients are **relative risk ratios** relative to the baseline category — "a
1-unit increase in $x$ multiplies the odds of category $k$ vs. the baseline by $e^{\beta_k}$." There
is no single, model-wide "odds ratio" the way there is in binary logit; every ratio is
baseline-relative.

### 3.4 Assumptions and limitations

- Same independence-of-observations and linearity-in-log-odds caveats as binary logit, applied
  per category.
- **IIA**, as discussed above — check whether your categories are truly distinct choices before
  trusting the model with substitute-heavy category sets.
- **Data requirements scale with the number of categories.** Each additional category needs its own
  set of coefficients estimated, so multinomial logit needs meaningfully more data than binary logit
  to reach the same precision, especially for rare categories.
- **Ordered categories are a different model.** If your categories have a natural order (low/medium/
  high satisfaction), multinomial logit throws away that structure. Use ordered logit/probit
  (`statsmodels.miscmodels.ordinal_model.OrderedModel`) instead — it will be more efficient and more
  correctly specified.

### 3.5 What this model family is good for / not good for

**Good for:** unordered categorical outcomes with a small-to-moderate number of well-separated
categories — species/type classification, product category prediction, mode-of-transport choice
(when alternatives are sufficiently distinct), diagnosis among a defined set of conditions.

**Not good for:** very high-cardinality categories (hundreds of classes — consider a different
architecture entirely), ordered categories (use ordered logit/probit), or choice sets with strong
substitution patterns among alternatives (use nested/mixed logit).

---

## 4. Poisson Regression (Counts)

### 4.1 Derivation

The **Poisson distribution** models the number of events in a fixed interval, given they occur
independently at a constant average rate $\lambda$:
$$P(Y=k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

To connect $\lambda$ to covariates while keeping predictions non-negative, we use a
**log link** — modeling $\ln(\lambda)$, rather than $\lambda$ itself, as linear in the predictors:
$$\ln(\lambda) = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n
\iff \lambda = e^{\beta_0 + \beta_1 x_1 + \dots}$$

Because $\lambda = e^{(\cdot)}$ is always positive regardless of the linear predictor's sign or
magnitude, predictions can never go negative — solving the core problem that OLS has with count data.

### 4.2 The defining assumption: equidispersion

The Poisson distribution has the unusual property that its **mean equals its variance**:
$\text{Var}(Y) = E[Y] = \lambda$. This is a strong, testable, and frequently violated assumption for
real count data (see Section 5).

### 4.3 Interpreting coefficients

$e^{\beta_j}$ is the multiplicative change in the **expected count** per one-unit increase in $x_j$,
holding other variables fixed — directly analogous to the odds-ratio interpretation in logit
regression, but applied to a rate/count instead of odds.

### 4.4 The offset / exposure term

Many real count problems have a natural "exposure" — the opportunity for events to occur — that
differs across observations: total time observed, population size, area, number of trials. Ignoring
this conflates *rate* with *raw count*. The standard fix is an **offset**:
$$\ln(\lambda_i) = \ln(\text{exposure}_i) + \beta_0 + \beta_1 x_{1i} + \dots$$
implemented by passing `offset=np.log(exposure)` to the model, with its coefficient fixed at exactly
1 (unlike a regular covariate). This lab's bike-sharing case study sidesteps this by using per-week
*mean* daily counts, but many real applications (insurance claims per policy-year, defects per batch
size, incidents per site-year) require it explicitly — see Part 3 of the extended lab notebook.

### 4.5 Assumptions and limitations

- **Equidispersion** (mean = variance) — violated very often; see Section 5.
- **Independence of events** — the Poisson process assumption that events occur independently of one
  another. Contagion effects (one event making a subsequent event more likely — e.g., a machine
  failure cascading into more failures) violate this and cause overdispersion.
- **Correct exposure/offset specification** — omitting a needed offset silently biases coefficients
  toward reflecting differences in exposure rather than differences in the true underlying rate.
- **Excess zeros.** Many real count datasets have far more zeros than a Poisson distribution would
  predict (e.g., number of doctor visits — most people have zero). This needs a **zero-inflated
  Poisson** or **hurdle model**, not covered in this lab, but worth knowing exists.
- **Log-linearity.** The model assumes the log of the expected count is linear in the predictors;
  as with logit, curvature needs explicit modeling (polynomial terms, splines, interactions).

### 4.6 What this model family is good for / not good for

**Good for:** modeling rates and counts with mild-to-no overdispersion — number of customer visits,
number of defects per unit given a well-controlled process, event counts over a fixed, well-defined
exposure window when the equidispersion assumption roughly holds.

**Not good for:** count data with heavy overdispersion (use negative binomial — Section 5), count
data with an excess of zeros beyond what the exposure would suggest (use zero-inflated/hurdle
models), or non-independent event processes (e.g., contagious/clustered failures) without additional
structure.

---

## 5. Negative Binomial Regression (Overdispersed Counts)

### 5.1 Why it's needed

In real data, $\text{Var}(Y) > E[Y]$ is common — **overdispersion**. Common causes include:
- unobserved heterogeneity between subjects (some subjects have a systematically higher underlying
  rate than others, even after controlling for measured predictors),
- contagion/clustering of events,
- omitted predictors that would have explained some of the variance.

Fitting Poisson regression to overdispersed data still gives **consistent point estimates** of the
coefficients (the mean structure is correctly specified), but the **standard errors are too small**,
because Poisson's likelihood assumes less variability than the data actually has. This means p-values
will be **overstated in significance** — you'll see too many "significant" predictors, and confidence
intervals will be falsely narrow.

### 5.2 The model

Negative binomial regression keeps the same log-linear mean structure as Poisson but adds a
dispersion parameter $\alpha$ (via a Poisson-Gamma mixture interpretation: each observation's rate
$\lambda_i$ is itself drawn from a Gamma distribution around the "true" mean, adding extra spread):
$$\text{Var}(Y) = \mu + \alpha \mu^2$$
As $\alpha \to 0$, this reduces exactly to the Poisson variance $\mu$, so Poisson is a special case /
nested model of negative binomial.

### 5.3 Estimating alpha: the auxiliary OLS regression

The lab uses the Cameron & Trivedi auxiliary-regression approach:
1. Fit Poisson regression to get fitted means $\hat\mu_i$.
2. Compute $\dfrac{(y_i - \hat\mu_i)^2 - y_i}{\hat\mu_i}$ for each observation.
3. Regress this quantity on $\hat\mu_i$ **with no intercept**.
4. The slope coefficient is the estimated $\alpha$; its t-test is a formal test of
   $H_0: \alpha = 0$ (Poisson is adequate) vs. $H_1: \alpha \ne 0$ (overdispersion present).

This estimated $\alpha$ can then be plugged into `sm.GLM(..., family=NegativeBinomial(alpha=...))`,
or `alpha` can instead be estimated jointly with the regression coefficients via full MLE using
`sm.NegativeBinomial(...).fit()` — the two approaches usually agree closely but are not numerically
identical.

### 5.4 Assumptions and limitations

- **Conceptually, this model is for counts arising from a (quasi-)fixed number of Bernoulli trials**
  (per the source text) — e.g., number of failures before a fixed number of successes. In practice it
  is used more broadly as a flexible overdispersed-count model, but it's worth knowing this is a
  looser use of the model than its textbook derivation.
- **It corrects variance, not the mean structure.** If your mean structure (choice of predictors,
  functional form) is wrong, negative binomial regression will not fix that — it only relaxes the
  variance=mean constraint.
- **Still assumes independence between observations** and log-linearity in the predictors, same as
  Poisson.
- **Doesn't handle excess zeros specifically** either — a zero-inflated negative binomial model
  exists for count data that is *both* overdispersed *and* zero-heavy.
- **Can be numerically less stable** than Poisson, particularly with small samples or with an
  `alpha` estimate very close to 0 (in which case you're better off just using Poisson).

### 5.5 What this model family is good for / not good for

**Good for:** overdispersed count data common in insurance claims modeling, healthcare utilization
counts, ecological abundance counts, and manufacturing defect counts where variability between
subjects/units exceeds what a pure Poisson process would produce.

**Not good for:** underdispersed data (variance < mean — rare, but negative binomial can't represent
it; consider a generalized Poisson or Conway-Maxwell-Poisson model instead), data dominated by
structural zeros (use zero-inflated variants), or situations where you haven't actually verified
overdispersion is present — always run the auxiliary test first rather than assuming.

---

## 6. Cross-Cutting Limitations of the Whole Model Family (GLMs in General)

1. **Correlational, not causal, by default.** All four models estimate association between
   predictors and outcomes under a given functional form. None of them, on their own, establish
   causation — that requires either a randomized experiment or careful causal-inference design
   (instrumental variables, difference-in-differences, matching, etc.) layered on top.

2. **Extrapolation risk.** Every model in this lab is a global parametric form (log-odds is linear;
   log-count is linear). Predictions far outside the range of the training data's predictors can be
   wildly wrong even if the model fits well within that range — none of these models "know" when
   they're extrapolating.

3. **Multicollinearity** among predictors inflates standard errors and destabilizes coefficient
   estimates in all four model families, exactly as in OLS. Check variance inflation factors (VIFs)
   before trusting individual coefficient interpretations.

4. **Model selection ≠ truth discovery.** Comparing AIC/BIC/pseudo-R²/accuracy across models tells
   you which model fits *this sample* better under a specific criterion — it does not guarantee
   you've found the "true" data-generating process, especially with correlated or omitted predictors.

5. **Small-sample instability.** Every example in the source textbook chapter uses very small
   training/test sets (31 training rows, 5 test rows for the admissions example; 150 rows for Iris).
   MLE-based inference (all the models here) is asymptotically justified — that is, its nice
   properties (unbiasedness, correct standard errors, normal sampling distributions for test
   statistics) are guaranteed as sample size grows large, not for small samples. Treat small-sample
   results as illustrative, not as strong statistical evidence.

6. **These are not deep learning models.** They cannot automatically discover nonlinear interactions
   or feature representations the way tree ensembles or neural networks can. Their major
   countervailing advantage is **interpretability**: every coefficient has a precise, defensible
   statistical meaning (odds ratio, relative risk ratio, rate multiplier), which matters enormously
   in regulated domains (credit, healthcare, insurance, criminal justice) where "the model said so"
   is not an acceptable explanation.

---

## 7. What Can Realistically Be Simulated and Learned in This Lab

**Can be simulated successfully:**
- Clean, correctly-specified binary/multinomial/count data-generating processes (as in the extended
  lab's simulated datasets) — these let you verify that estimation recovers known true coefficients,
  building intuition for how sample size affects estimation precision.
- Overdispersion, by construction, via Poisson-Gamma mixtures — this lets you *see* the negative
  binomial correction working exactly as theory predicts.
- The practical mechanics of model fitting, diagnostic checking, and comparison workflows that
  transfer directly to real projects (train/test splitting, dispersion checks, AIC/BIC comparison,
  ROC/AUC evaluation).

**Cannot be fully captured by this lab (be aware, for real-world work):**
- **Genuine unobserved confounding** in real data — simulated data has no hidden variables by
  construction, so exercises here can't teach you to detect confounding you don't already know about.
- **Non-independent / clustered / longitudinal data structures** — none of the datasets used here
  have repeated measures on the same subject or a network/spatial dependence structure; real
  applications with such structure need mixed-effects or GEE extensions.
- **Model misspecification in the wild** — real data rarely follows a textbook-clean Poisson,
  logistic, or negative binomial process; it usually has some combination of excess zeros,
  measurement error, non-stationarity over time, and nonlinear relationships that these base models
  don't address. Treat this lab as the *foundation* for those extensions, not the end of the road.
- **Fairness and bias considerations.** None of the datasets here have the kind of sensitive-attribute
  and historical-bias issues that real deployed models (e.g., admissions, credit, criminal justice
  risk scores) must be audited for. Production use of any of these techniques on human-subject data
  should include a fairness/bias audit that is entirely outside the scope of this statistical lab.

---

## 8. Suggested Next Steps Beyond This Lab

- **Ordered logit/probit** for ordered categorical outcomes (`statsmodels.miscmodels.ordinal_model`).
- **Zero-inflated and hurdle models** for count data with excess zeros
  (`statsmodels.discrete.count_model`).
- **Mixed-effects / hierarchical extensions** (e.g., `statsmodels.genmod.generalized_estimating_equations`,
  or `pymer4`/`lme4`-style tools) for clustered or repeated-measures data.
- **Regularized variants** (L1/L2-penalized logistic regression via `sklearn`) for high-dimensional
  predictor sets.
- **Causal inference layered on top of GLMs** (propensity score methods, instrumental variables) if
  the goal shifts from prediction/association to causal effect estimation.
