# Background Theory, Limitations, and Simulation Scope

### Companion document to the Extended Lab, Skeleton, Cheat Sheet, Template, and Solutions notebooks
### Topic: Multiple Linear Regression → Feature Selection → Shrinkage → Dimension Reduction

---

## 1. Purpose of this document

The notebooks in this lab pack are hands-on. This document is the "why" behind them: the statistical theory each
technique rests on, the assumptions that make the theory valid, where those assumptions break down in practice,
and — most importantly — an honest account of what this kind of lab can and cannot teach you about real-world
modeling.

---

## 2. Multiple Linear Regression (MLR): the foundation

### 2.1 The model

Simple linear regression (SLR) models a response `y` as a linear function of one explanatory variable `x`:

```
y = β0 + β1*x + ε
```

Multiple linear regression extends this to `p` explanatory variables:

```
y = β0 + β1*x1 + β2*x2 + ... + βp*xp + ε
```

`β0` is the intercept; each `βj` is the *partial* effect of `xj` on `y` — the change in the expected value of `y`
for a one-unit increase in `xj`, **holding all other variables in the model constant**. This "holding constant"
clause is the entire reason MLR is more useful than a series of separate SLRs: it isolates each variable's
association with the target from the associations of the other included variables.

The model is fit by **ordinary least squares (OLS)**: choose the `β` coefficients that minimize the residual sum
of squares,

```
RSS = Σ (yi - ŷi)²   where ŷi = β0 + β1*xi1 + ... + βp*xip
```

OLS has a closed-form solution, `β̂ = (XᵀX)⁻¹Xᵀy`, which exists and is unique as long as `XᵀX` is invertible —
this single fact is the mathematical root of the multicollinearity problem discussed below.

### 2.2 The four classical assumptions

1. **Linearity** — the true relationship between each `xj` and `y` (conditional on the others) is linear. Checked
   informally with scatter plots of `y` against each `xj`, and more rigorously with partial-regression
   ("added variable") plots.
2. **Normality of residuals** — the errors `ε` are normally distributed. This matters most for the validity of
   confidence intervals and hypothesis tests (the `P>|t|` column), less for the point estimates themselves. OLS
   coefficient estimates remain unbiased under non-normal errors as long as the other assumptions hold; only the
   inferential machinery (p-values, CIs) is affected. Checked with a residual histogram and a QQ-plot.
3. **Homoscedasticity** — the variance of the residuals is constant across the range of fitted values. Violation
   (heteroscedasticity) doesn't bias the coefficients but does bias the standard errors, making p-values and
   confidence intervals unreliable. Checked with a residuals-vs-fitted plot: a "megaphone" shape signals trouble.
4. **Independent errors** — residuals are not correlated with each other. This is primarily a concern for
   time-ordered or spatially clustered data (e.g. repeated measurements on the same subject). Checked with the
   Durbin-Watson statistic for ordered data (values near 2 suggest no strong autocorrelation).

MLR adds a fifth, practical concern that SLR doesn't have:

5. **Limited multicollinearity** — the explanatory variables should not be too strongly linearly related to each
   other.

### 2.3 Multicollinearity and the Variance Inflation Factor (VIF)

When two or more predictors are highly correlated, `XᵀX` becomes nearly singular. The regression can still be
fit, but:
- Coefficient estimates become highly sensitive to small changes in the data (high variance).
- Standard errors inflate, making it hard to detect real effects (low statistical power) even when the *joint*
  effect of the correlated block of variables is strong and precisely estimated.
- Individual coefficient signs and magnitudes can become uninterpretable or counter-intuitive.

The **Variance Inflation Factor** for predictor `j` quantifies this:

```
VIF_j = 1 / (1 - R²_j)
```

where `R²_j` is the R² from regressing `xj` on all the *other* predictors. If `xj` can be perfectly predicted
from the others, `R²_j → 1` and `VIF_j → ∞`. A VIF of 1 means no correlation with the other predictors; VIF ≥ 5
(some practitioners use 10) is the conventional threshold for concern. The standard remedy demonstrated in this
lab is **iterative elimination**: repeatedly drop the variable with the highest VIF and recompute, until all
remaining VIFs are acceptable. This is simple and effective but is not the only remedy — ridge regression and PCR
(Sections 4–5) handle multicollinearity without discarding variables outright.

### 2.4 Categorical variables and dummy encoding

A categorical variable with `L` levels is represented by `L - 1` binary "dummy" columns; one level is chosen as
the **reference level** and gets no dummy column of its own. The coefficient on a dummy variable is the shift in
the expected value of `y` for that level relative to the reference level, holding all other variables constant.
This is why, mathematically, a categorical variable only ever shifts the regression surface by a constant amount
per level — it cannot introduce curvature or interaction on its own (that requires explicitly adding interaction
terms, which the base MLR model in this lab does not do).

---

## 3. Feature Selection

Feature selection addresses a practical question: given many candidate predictors, which subset produces the
best, most trustworthy model? Two philosophies are covered:

### 3.1 Statistical methods

- **Correlation ranking**: fast, cheap, and a reasonable first filter, but limited to linear, pairwise,
  numeric-variable relationships. It ignores interactions between predictors and cannot handle categorical
  variables (Pearson's correlation is undefined for unordered categories).
- **Forward / backward / stepwise selection**: iteratively add (or remove) variables based on p-value thresholds.
  These methods are intuitive and were historically dominant, but have well-documented statistical problems: they
  perform many implicit hypothesis tests without correcting for multiple comparisons, they are greedy (a variable
  excluded early can never re-enter in forward selection), and the resulting p-values and confidence intervals
  from the *final* selected model are no longer valid, because the model was chosen using the same data used to
  test it. This is why the source material — and modern practice generally — treats these as illustrative rather
  than recommended.

### 3.2 Performance-based methods

- **Recursive Feature Elimination with Cross-Validation (RFECV)**: fits the model repeatedly on nested subsets of
  features, ranks features by their contribution to model performance (not statistical significance), and uses
  **cross-validation** — splitting data into training and validation folds — to select the subset size that
  generalizes best to unseen data. This directly targets the quantity you usually care about (predictive
  performance on new data) rather than a proxy (a p-value in one specific fitted model).

### 3.3 Comparing models

Two workhorse metrics recur throughout:
- **MSE / RMSE** — mean (root) squared error; penalizes large errors disproportionately; same units as `y` when
  using RMSE.
- **MAPE** — mean absolute percentage error; easier to communicate to non-technical audiences ("the model is off
  by about 12% on average") but undefined/unstable when `y` can be zero or near-zero.

---

## 4. Shrinkage (Regularization)

### 4.1 The bias-variance trade-off

Every model's expected prediction error can be decomposed (conceptually) into **bias** (systematic error from
oversimplifying the true relationship) and **variance** (sensitivity of the fitted model to which particular
training sample it saw). OLS is the unique **unbiased** linear estimator with minimum variance *among unbiased
estimators* — but "minimum variance among unbiased estimators" is not the same as "minimum total error." Shrinkage
methods deliberately introduce a small amount of bias in exchange for a often much larger reduction in variance,
which can lower total expected error, especially when predictors are numerous or correlated.

### 4.2 Ridge Regression (L2 penalty)

Minimizes `RSS + λ * Σ βj²`. As `λ → 0`, ridge recovers OLS; as `λ → ∞`, all coefficients shrink toward (but never
exactly to) zero. Ridge is ideal when you believe most or all predictors contribute some real signal and you
mainly want to tame variance from multicollinearity — it keeps every variable in the model.

### 4.3 LASSO Regression (L1 penalty)

Minimizes `RSS + λ * Σ |βj|`. Geometrically, the L1 penalty's constraint region has corners on the coordinate
axes; the least-squares solution often lands exactly on one of those corners, which is why LASSO can zero out
coefficients entirely — performing automatic variable selection as a side effect of regularization, unlike ridge.

### 4.4 Elastic Net

Blends both penalties: `RSS + λ*[(1-α)/2 * Σβj² + α * Σ|βj|]`. The mixing parameter `α` (called `l1_ratio` in
scikit-learn) interpolates between pure ridge (`α=0`) and pure LASSO (`α=1`). Elastic Net tends to outperform
LASSO when predictors are highly correlated in groups, because LASSO tends to arbitrarily pick one variable from
a correlated group and zero out the rest, while Elastic Net can retain the whole group with similar coefficients.

### 4.5 Choosing λ (and why this lab differs from the source text)

The source material illustrates ridge/LASSO/Elastic Net with a single, manually chosen `alpha`. This lab instead
uses `RidgeCV` / `LassoCV` / `ElasticNetCV`, which select `λ` (and, for Elastic Net, `α`) via cross-validation —
the standard, defensible approach in practice. A hand-picked `alpha` is useful for *illustrating* the mechanics of
shrinkage but should not be the basis for a real deployed model.

**Critical practical requirement:** all shrinkage methods are scale-sensitive, because the penalty term sums raw
coefficient magnitudes/squares. A variable measured in the thousands (e.g. income) would be penalized very
differently than one measured in single digits (e.g. number of children) if left unscaled. **Standardizing all
predictors before fitting is mandatory**, not optional, for these methods.

---

## 5. Dimension Reduction: PCA, PCR, and PLS

### 5.1 Principal Component Analysis (PCA)

PCA re-expresses a set of `p` correlated variables as `p` new, **uncorrelated** variables (principal components),
ordered by how much of the total variance in the original data they explain. The first component is the direction
of maximum variance; each subsequent component is the direction of maximum remaining variance, subject to being
orthogonal to all previous components. Because the components are constructed purely from the covariance
structure of `X`, **PCA never looks at the target variable `y`** — it is an unsupervised technique.

### 5.2 Principal Component Regression (PCR)

PCR is a two-step recipe: (1) run PCA on the (standardized) predictors, (2) fit an ordinary linear regression
using the first `k` principal components as the predictors instead of the original variables. Because the
components are uncorrelated by construction, PCR sidesteps multicollinearity entirely without discarding any
original variable's information (each component is a specific linear combination of *all* original variables,
weighted by that variable's contribution to that direction of variance).

The number of components `k` is a hyperparameter, chosen by cross-validation (as in the extended lab and
template) rather than an arbitrary cutoff. The trade-off: PCR is harder to interpret, since a "coefficient" now
applies to an abstract direction in feature space rather than a single named variable.

**A leakage bug worth calling out explicitly:** because PCA is a *fitted* transformation (it estimates component
directions from data), it must be fit on the training data only and then *applied* (not re-fit) to the test data.
The source notebook's example re-fits PCA on the test set before evaluating test performance — this leaks test-set
structure into the "test" evaluation and makes the reported test error unrealistically optimistic. The lab and
template notebooks in this pack fix this by using `.transform()`, not `.fit_transform()`, on the test set.

### 5.3 Partial Least Squares (PLS) Regression — an extension beyond the source material

PLS is closely related to PCR but builds its components **supervised by the target**: each PLS component is
chosen to maximize covariance with `y`, not just variance within `X`. In practice this often means PLS reaches
comparable or better predictive accuracy with **fewer** components than PCR needs, because every PLS component is
guaranteed to carry some predictive signal, whereas a high-variance PCA component might happen to be irrelevant to
`y`.

---

## 6. Limitations of this lab (read before drawing broad conclusions)

This lab is designed to teach *mechanics* clearly. Several simplifications make that possible, and it's important
to know what they are:

1. **Single train/test splits.** Most of the lab evaluates models on one fixed train/test split. Real
   generalization performance has its own variance — a different random split can meaningfully change which model
   "wins." Proper practice uses repeated cross-validation or nested CV for model *selection* and a held-out test
   set only for a final, one-time performance estimate.
2. **Clean, tabular, complete data.** The diabetes dataset arrives pre-cleaned and pre-scaled; Hitters requires
   only a `.dropna()`. Real datasets bring inconsistent encodings, mixed types, outliers, non-random
   missingness, and label errors that no amount of algorithmic sophistication fixes automatically.
3. **Purely linear relationships assumed.** Every model here (OLS, ridge, LASSO, Elastic Net, PCR, PLS) is
   fundamentally linear in the (possibly transformed) predictors. None of them will capture genuine non-linear
   or threshold effects unless you engineer polynomial/interaction features yourself first.
4. **No interaction terms.** The MLR model as built only allows each predictor to shift `y` independently — it
   cannot represent "the effect of X1 depends on the level of X2" without explicitly adding an `X1*X2` term,
   which this lab does not do.
5. **Correlational, not causal.** A statistically significant, well-behaved coefficient tells you about
   association conditional on the other included variables — not that changing `xj` would *cause* a change in
   `y`. Causal claims require a causal design (randomization, instrumental variables, natural experiments, etc.),
   which is out of scope here.
6. **Small-to-moderate, low-dimensional datasets.** Diabetes (442 rows × 10 features), a synthetic dataset
   (2000 × 20), and Hitters (263 rows × ~19 features) are all comfortably within the regime where these classical
   methods work well and quickly. None of this lab addresses the computational or statistical challenges of
   genuinely high-dimensional data (`p` in the thousands or millions) or big-data-scale training.
7. **Synthetic substitute in Part C.** For network-independence, the shrinkage section uses `make_regression`
   instead of a real-world dataset like California housing. This is excellent for *seeing the mechanism* (because
   you know the ground-truth informative features), but it means the specific numeric results don't tell you
   anything about real housing markets — only about how ridge/LASSO/Elastic Net behave in a controlled setting.
8. **No hyperparameter search beyond `alpha` / component count.** Real projects often also tune things like the
   choice of cross-validation scheme, feature engineering choices, or ensemble/non-linear alternatives (random
   forests, gradient boosting) as competing baselines — this lab stays within the classical linear-modeling
   toolbox by design.
9. **No deployment, monitoring, or drift considerations.** Fitting and evaluating a model once is not the same as
   maintaining one in production, where data distributions shift over time and periodic re-validation is needed.

---

## 7. What this lab *can* teach you reliably

- The mechanical, hands-on process of diagnosing an MLR model against its assumptions.
- Why and how multicollinearity is detected (VIF) and one standard remedy (iterative elimination).
- The practical difference between p-value-driven and cross-validation-driven feature selection, and why the
  field has shifted toward the latter.
- How ridge, LASSO, and Elastic Net trade bias for variance, visualized directly via coefficient paths.
- The mechanics and correct, leakage-free implementation of PCR, plus how PLS differs from it.
- How to build a single, fair, apples-to-apples comparison across several modeling approaches on the same data.
- Transferable code patterns (the reusable template) for applying this exact workflow to a new regression dataset.

## 8. What this lab *cannot* teach you, and where to go next

- How to handle messy, real-world data ingestion and cleaning at scale.
- How to detect or model non-linear relationships (splines, tree-based methods, neural networks).
- How to reason about or establish causality.
- How to validate a model's fairness, robustness to distribution shift, or behavior in production.
- How to choose among linear models and fundamentally different model families (e.g., gradient-boosted trees)
  for a given real business problem — that requires broader model comparison than this lab's linear-only scope.

If you want to build on this lab, natural next steps are: (a) add polynomial/interaction features and re-run the
same diagnostic and shrinkage pipeline, (b) replace the linear estimator inside the template's functions with a
tree-based model to compare non-linear performance, and (c) practice the same workflow on a dataset with real
missingness and messy categorical encodings to build data-cleaning skills alongside the modeling skills.
