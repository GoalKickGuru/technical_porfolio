# Background Theory: Non-Parametric Hypothesis Testing

This document is the "why" behind the lab: the statistical theory underneath
each test, the assumptions each one actually relaxes (and the ones it
*doesn't*), and — most importantly — a candid discussion of what this class of
methods can and cannot be used to simulate or conclude.

---

## 1. Why Non-Parametric Tests Exist

Parametric tests (t-tests, ANOVA, Pearson correlation) get their statistical
power from assuming a specific family of distributions for the data —
typically the normal distribution — and estimating a small number of
parameters (mean, variance) that fully describe it. When that assumption
holds, parametric tests are the more powerful choice: they extract more
information per data point than a rank-based alternative.

The trouble is that real data frequently violate these assumptions:

- **Small samples** where you can't verify normality with any confidence.
- **Skewed or heavy-tailed distributions** (counts, reaction times, income,
  defect counts).
- **Ordinal data** (survey Likert scales, rankings) where the numbers encode
  order but not consistent intervals — a mean is not really meaningful.
- **Outliers** that a mean/variance-based test is highly sensitive to.

Non-parametric ("distribution-free") tests replace the raw values with
**ranks**, use **resampling** (permutation), or work directly with **counts**
so that the null distribution of the test statistic doesn't depend on
assuming a particular data-generating distribution.

**Critical caveat repeated throughout the source material:** non-parametric
tests still require **independent observations**. Nothing here rescues you
from clustered, autocorrelated, or otherwise dependent data — that assumption
is never optional.

---

## 2. The Logic of a P-Value (Permutation Test)

A p-value is the probability, under the null hypothesis, of observing a test
statistic at least as extreme as the one actually observed. To compute one you
need (a) an observed statistic and (b) the *null distribution* of that
statistic.

The permutation test constructs the null distribution directly from the data:
if the null hypothesis (the two groups come from the same distribution) is
true, then the group labels are exchangeable — shuffling which observations
are "labeled" group A vs. group B shouldn't systematically change the
statistic. So:

1. Compute the observed statistic (e.g., mean(A) − mean(B)).
2. Repeatedly permute the labels and recompute the statistic on each
   permutation. This builds an empirical approximation of the null
   distribution.
3. The p-value is the proportion of permuted statistics at least as extreme as
   the observed one.

This works for **any** statistic you can define in code — mean difference,
median difference, a correlation, a custom weighted score — which is the main
reason to reach for a permutation test over a named test: flexibility.

**Scaling limitation:** the number of *distinct* permutations grows
factorially. A dataset of size 12 already has ~479 million permutations; by a
few dozen samples, exhaustive enumeration is computationally infeasible.
`scipy.stats.permutation_test` handles this by defaulting to a large but
finite number of **random** permutations (Monte Carlo approximation) rather
than the exhaustive set, which is accurate but introduces a small amount of
Monte Carlo noise in the p-value (mitigated by using more resamples).

---

## 3. Rank-Based Tests: Rank-Sum, Signed-Rank, Kruskal-Wallis

All three of these convert raw values into **ranks** before computing a test
statistic. Ranking throws away the exact magnitude of differences but
preserves order — which is exactly the information non-parametric tests are
willing to bet on.

### 3.1 Mann-Whitney U / Wilcoxon Rank-Sum Test

**Null hypothesis:** the two independent samples come from the same
distribution (commonly interpreted as "no difference in location" when the
shapes of the two distributions are similar).

**Procedure:**
1. Pool both samples and rank all values together (ties get the *average* of
   the ranks they'd otherwise occupy).
2. Sum the ranks belonging to one group — this is the test statistic *T*
   (equivalently, U, computed as `T - n(n+1)/2` for that group).
3. For small samples, an *exact* permutation-based distribution of T is used.
   For larger samples, T is approximately normal, so a Z-score is formed:

   `Z = (T − Mean(T)) / SD(T)`

   where `Mean(T) = n_T * R̄` and `SD(T) = s_R * sqrt(n_T * n_O / (n_T + n_O))`
   (R̄ and s_R being the mean and standard deviation of the corrected ranks).
4. The p-value comes from the standard normal distribution (or the exact
   distribution for small n).

**What it actually tests:** strictly, whether one distribution stochastically
dominates the other (P(X > Y) ≠ 0.5). It's only cleanly interpretable as "a
shift in the median/location" when the two distributions have similar shape
and spread — if the shapes differ substantially (e.g., one is much more
skewed), a significant Mann-Whitney result could reflect a difference in
shape rather than a location shift.

### 3.2 Wilcoxon Signed-Rank Test

**Null hypothesis:** the median of the paired differences is zero.

**Procedure:**
1. Compute the difference for each pair; drop pairs with a difference of zero.
2. Rank the *absolute* differences (average ranks for ties).
3. Sum the ranks belonging to the positive differences — this is the test
   statistic *S*.
4. For larger n, S is approximately normal:
   `Mean(S) = n(n+1)/4`, `SD(S) = sqrt(n(n+1)(2n+1)/24)`,
   `Z = (S − Mean(S)) / SD(S)`.
5. p-value from the normal distribution (or exact for small, tie-free n —
   which is what `scipy.stats.wilcoxon` uses by default).

**Assumption it keeps:** the distribution of the *differences* should be
(approximately) symmetric around the median under the null — the test is not
fully assumption-free, it just relaxes normality of the raw scores.

### 3.3 Kruskal-Wallis Test

**Null hypothesis:** all k independent groups have the same median (more
precisely, come from the same distribution).

**Procedure (conceptually):** rank all observations across all groups
together, then compute a statistic (denoted H) based on how far each group's
average rank is from the overall average rank. H approximately follows a
chi-square distribution with k−1 degrees of freedom under the null.

**What it doesn't do:** like ANOVA's F-test, a significant Kruskal-Wallis
result only tells you "not all groups are equal" — it doesn't say *which*
pairs differ. That requires a **post-hoc** procedure: pairwise Mann-Whitney U
tests with a multiple-comparisons correction (Bonferroni, in this lab; Dunn's
test is a more rigorous alternative used in practice) to control the
inflated false-positive rate from running many pairwise tests.

---

## 4. Chi-Square Tests

Both chi-square tests below share the same underlying statistic:

`χ² = Σ (O − E)² / E`

where O is an observed frequency and E is the frequency expected under the
null hypothesis. This statistic follows (approximately) a chi-square
distribution with degrees of freedom determined by the structure of the
table.

### 4.1 Goodness-of-Fit

Tests whether the observed counts across the levels of a **single** categorical
variable match a hypothesized (expected) distribution. Degrees of freedom =
k − 1 for k categories.

### 4.2 Test of Independence

Tests whether **two** categorical variables are associated, using a
contingency (cross-tab) table. Expected counts are computed from the row and
column totals: `E_ij = (row total_i × column total_j) / grand total`. Degrees
of freedom = (rows − 1)(columns − 1).

**Yates' continuity correction** (optional, `correction=True` in
`chi2_contingency`) shifts each observed count 0.5 toward its expected value
before computing χ². It compensates for the chi-square distribution being a
continuous approximation to a discrete process, and matters most when
expected cell counts are small (rule of thumb: below ~10). With large samples
it changes almost nothing.

**Both chi-square tests are always right-tailed:** the null hypothesis
corresponds to χ² = 0 (observed exactly matches expected); any departure,
regardless of direction, pushes χ² upward, so only the upper tail signals
evidence against the null.

**Sample-size sensitivity:** with very large samples (the Texas crash dataset
in this lab has over a million observations), even a trivially small
association becomes "statistically significant." This is why **Cramér's V**
(a normalized effect size, `sqrt(χ² / (n × min(rows−1, columns−1)))`) is
essential alongside the p-value — it tells you the *strength* of the
association independent of sample size, on a 0–1 scale comparable across
studies.

---

## 5. Spearman's Rank Correlation

Pearson's correlation coefficient measures the strength of a **linear**
relationship between two continuous variables and is sensitive to outliers
and non-linearity. Spearman's rank correlation applies the same formula to
the **ranks** of the data instead of the raw values:

`r_s = S_xy / sqrt(S_xx × S_yy)`

(equivalently, for data with no or few ties: `r_s = 1 − 6Σd_i² / (n(n²−1))`,
where d_i is the difference in ranks for each pair.)

Because it operates on ranks, Spearman's r_s detects any **monotonic**
relationship (not just linear ones) and is robust to outliers — an extreme
value only shifts a rank by a small, bounded amount, unlike its effect on a
raw mean or covariance.

**Limitation with small samples:** the standard p-value for Spearman's rho
relies on an asymptotic approximation that becomes less reliable as n shrinks.
A **bootstrap confidence interval** (resampling pairs with replacement and
recomputing rho many times) gives an honest picture of how much uncertainty
remains — with n=7, as in this lab's judges example, that interval is wide
even though the point estimate looks convincingly large.

---

## 6. Assumptions These Tests *Do* and *Don't* Relax

| Assumption | Parametric test | Non-parametric alternative |
|---|---|---|
| Normal distribution | Required | Relaxed |
| Equal variance | Required (or Welch correction) | Mostly relaxed, but Mann-Whitney/Kruskal-Wallis interpretation as a location shift assumes *similar shape* |
| Independent observations | Required | **Still required — never relaxed** |
| Interval/ratio scale | Required (to compute meaningful means) | Not required — ranks work for ordinal data |
| Large sample size | Helps asymptotics but not strictly required | Small-sample exact methods exist (permutation, exact Wilcoxon) precisely because these tests are built for smaller/messier data |

---

## 7. Limitations of This Family of Tests

1. **Lower statistical power when parametric assumptions actually hold.** If
   your data genuinely are normal with equal variance, a t-test or ANOVA will
   detect a true effect with a smaller sample than the rank-based equivalent.
   Non-parametric tests are an insurance policy, not a strictly better default.

2. **Ranking discards information.** Two datasets with very different
   magnitudes of difference can produce the same rank-based statistic if the
   *order* of values is the same. This is precisely what buys robustness to
   outliers, but it also means these tests answer "is there a difference in
   rank/location" rather than "how big is the difference" — that's why effect
   sizes (rank-biserial r, eta-squared, Cramér's V, Cohen's w) are a required
   companion, not an optional extra.

3. **Ties reduce power and complicate variance formulas.** All the normal
   approximations used above (Z-scores for Mann-Whitney/Wilcoxon) assume a
   tie-correction to the variance term; with many tied values, exact or
   permutation-based methods are safer than the approximation.

4. **Omnibus tests (Kruskal-Wallis, chi-square independence) don't localize
   the effect.** They tell you *that* something differs, not *where*. You need
   a follow-up procedure (post-hoc pairwise tests, standardized residuals in a
   contingency table) to pin down the source.

5. **These tests don't generalize easily to multivariate or model-based
   questions.** None of the seven tests in this lab can answer "how does Y
   change as a function of X1, X2, and X3 simultaneously," control for
   confounders, or produce a predictive model. For that you need regression
   methods — ordinal logistic regression, rank-based regression, or
   generalized linear models — which is the natural next topic after this
   chapter's material.

6. **Chi-square tests need reasonably sized expected cells.** The
   chi-square distribution is an asymptotic approximation; with expected
   frequencies below ~5 (some say ~10 for small df), the approximation
   degrades and an exact test (e.g., Fisher's exact test for a 2×2 table) is
   more appropriate.

7. **Independence is assumed, never tested by these methods.** If
   observations are naturally clustered (e.g., repeated measurements on the
   same subject that aren't explicitly paired, students nested within
   classrooms), none of these tests are valid without modification — you'd
   need a clustered/hierarchical approach instead.

---

## 8. What This Type of Lab *Can* Simulate Successfully

- **Small-sample, real-world-shaped scenarios:** count data, skewed
  distributions, ordinal survey-style data, and paired before/after designs —
  exactly where non-parametric methods are the intended tool, and where the
  book's and this lab's synthetic examples (defect counts, treatment scores,
  shift comparisons) are realistic stand-ins for actual operational data.
- **The full inferential workflow** for a two-group or paired comparison:
  assumption-checking → test selection → running the test → effect size →
  plain-English conclusion. This mirrors real analyst work closely.
- **Contingency-table reasoning** for categorical data at any scale, from a
  handful of survey responses to a dataset with over a million rows (as in
  the Texas crash example) — the mechanics don't change with scale, only the
  interpretation of practical vs. statistical significance does.
- **Power/sample-size planning** for the chi-square goodness-of-fit case,
  which is directly transferable to real study design questions ("how many
  responses do I need to detect this size of imbalance").

## 9. What This Type of Lab *Cannot* Simulate Successfully

- **Multivariable relationships.** None of these tests handle more than one
  predictor or covariate adjustment; you can't use this toolkit to ask
  "controlling for department, does shift still predict output quality?"
- **Time-series or longitudinal dependence.** These tests assume independent
  observations (or simple pairing for exactly two time points via Wilcoxon).
  Anything with autocorrelation, trends over many time points, or
  subject-level random effects needs different machinery (e.g., the Friedman
  test for >2 repeated measures, or mixed-effects models beyond that).
- **Precise effect-magnitude estimation with tight confidence bounds** from
  very small samples. The tests will still run and produce a p-value, but as
  the Part 7 bootstrap exercise shows, a small-n correlation or rank-sum
  result can carry much more uncertainty than the point estimate alone
  suggests — this lab is well-suited to *teaching* that lesson, but not to
  producing decision-grade precision from tiny samples.
- **Causal claims.** Like any hypothesis test, rejecting a null hypothesis of
  "no association/no difference" is not evidence of a causal mechanism absent
  a designed experiment (randomization) or a causal-inference framework layered
  on top.

---

## 10. References

- Nguyen, Huy Hoang. *Building Statistical Models in Python* (2023), Chapter 5:
  Non-Parametric Tests.
- SciPy documentation: `scipy.stats.permutation_test`, `mannwhitneyu`,
  `wilcoxon`, `kruskal`, `chi2_contingency`, `spearmanr`.
- Statsmodels documentation: `statsmodels.stats.gof.chisquare`,
  `statsmodels.stats.power.GofChisquarePower`.
- OpenIntro Statistics dataset repository (`gpa_iq.rda`), used in Part 2 of
  the lab.
