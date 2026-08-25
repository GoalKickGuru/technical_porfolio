# Background & Theory: Data Types, Distributions, and Resampling Methods

*Companion document to the lab notebooks: `1_extended_lab.ipynb`, `2_cheat_sheet.ipynb`,
`3_skeleton_notebook.ipynb`, `4_solutions_notebook.ipynb`, `5_reusable_template.ipynb`.*

This document explains the **theory behind every technique used in the lab**, why the
techniques are structured the way they are, and — just as important — **where they break
down**: what the lab's simulations can and cannot tell you about real-world data.

---

## 1. Why data types come first

Every statistic is really a claim about which operations are meaningful to perform on a
set of numbers or labels. Stevens' (1946) typology — **nominal, ordinal, interval, ratio**
— ranks data by which arithmetic operations are valid:

| Type | Valid operations | Invalid operations | Example |
|---|---|---|---|
| Nominal | =, ≠ | <, >, +, −, ×, ÷ | machine model, blood type |
| Ordinal | =, ≠, <, > | +, −, ×, ÷ | star rating, education level |
| Interval | =, ≠, <, >, +, − | ×, ÷ (no true zero) | temperature (°C/°F), calendar year |
| Ratio | =, ≠, <, >, +, −, ×, ÷ | — (all valid) | income, wait time, weight |

**Why this matters for the rest of the lab:** every technique downstream — mean,
variance, bootstrapping, permutation testing, transformations — implicitly assumes you
are working with interval or ratio data (or, for the mean/median specifically, ordinal-or-
higher data where "middle" and "distance" are meaningful). Applying a mean to nominal
codes (e.g., averaging ZIP codes) produces a number, but not a meaningful one. The lab's
first module exists to build the habit of checking this *before* reaching for `np.mean()`.

**Limitation:** Stevens' typology is a simplification. Real variables often sit in gray
zones — Likert scales (1–5 agreement ratings) are technically ordinal but are routinely
treated as interval data in practice because the alternative (ordinal-only statistics) is
often too weak for practical analysis. The lab does not resolve this debate; it teaches
the classification so you can make an informed, defensible choice for your own data.

---

## 2. Descriptive statistics: center, spread, and shape

### 2.1 Center: mode, median, mean
- **Mode** — the only center measure valid for nominal data (works via the equality
  operation alone).
- **Median** — the middle value once sorted; valid for ordinal data and above; **robust
  to outliers** because it only depends on rank order, not magnitude.
- **Mean** — the arithmetic average; valid for interval/ratio data; **sensitive to
  outliers** because every value contributes proportionally to its magnitude.

### 2.2 Spread: range, IQR, Tukey fences, variance, standard deviation
- **Range** (max − min) is intuitive but as outlier-sensitive as the mean.
- **IQR** (Q3 − Q1) is the median's spread analogue — robust, but only reflects the
  middle 50% of the data, so it can miss real structure in the tails.
- **Tukey fences** (`Q1 − 1.5×IQR`, `Q3 + 1.5×IQR`) are a *heuristic*, not a statistical
  test, for flagging potential outliers. The constant 1.5 is a convention (John Tukey's
  original choice), not a derived optimum — treat flagged points as "worth
  investigating," not automatically "wrong" or "removable."
- **Variance / standard deviation** use every data point and are the foundation for
  parametric methods (t-tests, regression, confidence intervals based on the normal
  distribution). The sample formula divides by `N − 1` (Bessel's correction) rather than
  `N` to produce an unbiased estimator of the *population* variance from a sample —
  this is why `ddof=1` matters in NumPy.

### 2.3 Shape: skewness and kurtosis
- **Skewness** quantifies asymmetry. Right-skew (positive) means a long right tail (e.g.,
  income); left-skew (negative) means a long left tail.
- **Kurtosis** (as computed by `scipy.stats.kurtosis`, which reports *excess* kurtosis)
  quantifies tail weight relative to a normal distribution: positive means heavier tails
  (more extreme values than normal), negative means lighter tails.

**Limitation:** skewness and kurtosis estimates are themselves noisy for small samples
(roughly n < 30), and the lab's exercises sometimes compute them on samples that small
(e.g., the capstone's n=9–12 groups). Treat those numbers as *descriptive*, not as a
rigorous shape test — a formal test (D'Agostino's K², Shapiro-Wilk) is more appropriate
when the sample is small and the conclusion matters.

---

## 3. The normal distribution, the Empirical Rule, and the Central Limit Theorem

The normal (Gaussian) distribution is the backbone of classical (parametric) statistics
because so many parametric tests — t-tests, ANOVA, linear regression's error
assumptions, Pearson correlation's significance test — derive their formulas and
p-values under the assumption that some underlying quantity is normally distributed.

**The Empirical Rule** (68/95/99.7) is a direct consequence of the normal distribution's
shape and gives quick sanity checks: for genuinely normal data, ~68% of observations
fall within one standard deviation of the mean, ~95% within two, ~99.7% within three.

**The Central Limit Theorem (CLT)** is the deeper reason normal-distribution-based
methods work even when raw data isn't normal: *the sampling distribution of the mean*
(not the individual data points) approaches normality as sample size `n` grows,
**regardless of the shape of the original population**, provided the population has
finite variance. This is what the lab's CLT demo shows by drawing repeated samples from
a starkly non-normal exponential population and watching the distribution of sample
means become bell-shaped.

**Why this matters for bootstrapping (Module 3):** bootstrapping's validity rests on the
CLT. When you resample and compute a mean thousands of times, the resulting
distribution of bootstrap means is expected to approximate normality (or at least a
smooth, well-behaved shape) even if the original data was skewed — which is exactly why
bootstrapping is a popular tool for building confidence intervals on non-normal data.

**Limitations of the CLT in practice:**
- Convergence to normality is a limit as `n → ∞`; for small `n` (rule of thumb: n < 30,
  though this depends heavily on how skewed/heavy-tailed the population is), the
  sampling distribution can still be visibly non-normal. The lab's n=5 CLT panel is
  deliberately included to show this — it does *not* look normal yet.
- The CLT requires **finite variance**. Populations with extremely heavy tails (e.g.,
  certain financial return distributions, some power-law phenomena) may converge much
  more slowly, or the classical CLT may not apply in the usual form at all.
- The CLT describes the sampling distribution of the **mean** specifically. Other
  statistics (median, correlation, ratios, max/min) have their own — sometimes very
  different — convergence behavior and are not automatically "CLT-safe."

---

## 4. Bootstrapping

**Core idea:** treat your observed sample as a stand-in for the population, and estimate
the sampling distribution of a statistic by repeatedly resampling **with replacement**
from that sample, recomputing the statistic each time.

**What it's good for:**
- Confidence intervals for statistics that don't have a simple closed-form formula
  (medians, ratios, custom metrics, correlation coefficients).
- Small-sample inference when you can't safely assume normality but the sample is
  reasonably representative of the population.

**What the lab's implementation does:**
- `frac=0.5, replace=True` (as in the book) resamples half the data with replacement —
  an arbitrary but common choice; `frac=1.0, replace=True` (full-size resampling) is the
  more standard textbook definition of the bootstrap and is what the reusable template
  and cheat sheet default to. Both are valid, but **be consistent within a single
  analysis** and state which you used, since resample size affects the width of the
  resulting distribution.
- Percentile-based confidence intervals (2.5th/97.5th percentile for a 95% CI). This is
  the simplest bootstrap CI method; more advanced variants (BCa — bias-corrected and
  accelerated, or the studentized bootstrap) correct for skew in the bootstrap
  distribution itself and are preferable for rigorous work, but are out of scope here.

**Critical limitations:**
1. **Bootstrapping cannot manufacture information that isn't in your sample.** If your
   original 18-person sample is not representative of the population (biased
   collection, missing subgroups), every bootstrap resample inherits that same bias —
   bootstrapping estimates the sampling variability of your statistic, not the accuracy
   of your original sample.
2. **It performs poorly for very small samples** (rule of thumb: single digits to low
   teens, exactly the size used in this lab's Duncan-dataset and capstone examples).
   With n=9, there are only `2^9 − 1 = 511` distinct possible bootstrap resamples (up to
   ordering) — the bootstrap distribution is discretized and can understate true
   uncertainty. The lab's Exercise 3.2 is designed to make this visible.
3. **It performs poorly for statistics driven by extreme values** (max, min, or heavy-
   tailed quantities), because resampling with replacement cannot generate values more
   extreme than what's already in the sample.
4. Bootstrapped correlation coefficients (Module 3, Exercise 3.4-style analysis) assume
   the paired structure of the data is meaningful — always resample **rows**, not
   columns independently, or you destroy the very relationship you're trying to measure.

---

## 5. Permutations, combinations, and permutation testing

**Permutations vs. combinations** are counting tools, not inferential statistics on
their own — but they underlie a genuinely powerful *inferential* method: **permutation
testing**.

**Permutation testing logic:** under the null hypothesis that two groups come from the
same underlying distribution, the group *labels* are arbitrary — swapping them around
shouldn't systematically change a summary statistic like the difference in means. By
repeatedly shuffling the pooled data and re-splitting it into groups of the original
sizes, we build up an empirical **null distribution** of the test statistic. The
proportion of shuffles that are as extreme as (or more extreme than) the *actual*
observed statistic is the p-value.

**Why it's attractive:** unlike a t-test, it makes **no distributional assumption**
(no normality requirement) — it only assumes the two samples are exchangeable under the
null hypothesis. This makes it a natural complement to the normality checks in Module 2.

**Limitations:**
1. **Computational, not exact, for continuous data with large n.** With small, fully
   discrete datasets (like `A=[3,5,4]`, `B=[43,41,56,78,54]`), the *true* number of
   distinct permutations is finite and could in principle be enumerated exactly; the
   lab instead uses Monte Carlo shuffling (`n_iter=10000`), which introduces a small
   amount of simulation noise into the p-value. Increasing `n_iter` reduces — but never
   eliminates — this noise.
2. **Exchangeability, not just "no normality assumption," is the real requirement.**
   If group membership correlates with some other structural factor (e.g., professional
   vs. blue-collar workers were sampled in different years, with inflation-adjusted vs.
   nominal dollars), shuffling labels doesn't correctly represent the null hypothesis
   and the test's validity breaks down.
3. **It tests for *a* difference in the chosen statistic (usually the mean), not
   equivalence of the full distributions.** Two groups can have identical means but
   very different variances or shapes — a permutation test built around the mean
   difference will not detect that.
4. Choosing `alternative='greater'/'less'` versus `'two-sided'` **must be decided before
   looking at the data**; picking the direction after seeing which way the observed
   difference points inflates the false-positive rate (a form of p-hacking).

---

## 6. Transformations

Transformations re-express a variable on a different scale to make it more symmetric
(closer to normal) — useful before applying tools that assume normality, or simply to
make relationships more linear/interpretable.

| Transform | Effect | Requires |
|---|---|---|
| `log(x)` | Compresses large values much more than small ones; strongest de-skewing | x > 0 |
| `sqrt(x)` | Milder compression than log | x ≥ 0 |
| `cbrt(x)` | Similar to sqrt but defined for negative/zero values too | any real x |
| `log1p(x)` | Like log but safe when x can be exactly 0 | x > −1 |
| Box-Cox | Automatically finds the power transform (λ) that best normalizes the data | x > 0 |

**Limitations:**
1. **Transformations change the units and the interpretation of results.** A t-test or
   regression coefficient on `log(income)` answers a question about *multiplicative*,
   not additive, differences in income — that reframing must be communicated, not
   glossed over.
2. **Not all skew is "fixable."** Multimodal data (two or more distinct sub-populations
   mixed together) will not become unimodal-normal no matter which power transform you
   apply; the right fix there is often to model the sub-populations separately, not to
   transform.
3. **Box-Cox's "optimal" λ is optimal for normality of *this* sample**, not necessarily
   for downstream model performance, and it can behave unstably with very small samples
   or samples containing near-zero values (hence adding a small constant in the lab's
   code, which itself slightly distorts the transform — a known, accepted trade-off).
4. Always transform, then re-check assumptions and outliers **on the transformed
   scale** — an "outlier" flagged by Tukey fences on raw income data may not be an
   outlier at all on the log scale, and vice versa.

---

## 7. What this lab can and cannot simulate successfully

**What the simulations reliably demonstrate:**
- The Empirical Rule and CLT convergence, because `numpy`/`scipy` random generators
  produce exact, arbitrarily large samples from known theoretical distributions — this
  is the ideal setting to *see* asymptotic theory in action.
- The behavior of bootstrapping and permutation testing under controlled conditions
  (known true difference, known sample sizes), which is exactly how these methods'
  original developers validated them (Efron's bootstrap papers used simulation studies
  for the same reason).
- The effect of skew and the mechanics of each transformation, since we control the
  generating distribution (Beta, Gamma, exponential) precisely.

**What the simulations cannot tell you, and where the analogy to real data breaks down:**
1. **Real data is rarely i.i.d. (independent and identically distributed).** Machine
   output over time may have autocorrelation (today's reading depends on yesterday's
   maintenance), seasonal effects, or drift — none of which the lab's `np.random.*`
   generators include unless explicitly modeled. Every bootstrap/permutation result in
   this lab implicitly assumes independence; real processes should be checked for this
   (e.g., via autocorrelation plots) before applying these methods naively.
2. **Simulated "ground truth" isn't available with real data.** In the capstone, we know
   the *true* generating distributions (Gamma with specific parameters) because we
   built them — this lets us sanity-check the statistical methods against a known
   answer. With real machine-output data, there is no such ground truth; a
   statistically significant permutation-test result is evidence, not proof, and
   should be corroborated with domain knowledge (e.g., a mechanical reason Model C
   would run hotter/faster).
3. **Sample sizes in the lab (n=7 to n=21) are deliberately small to match the
   textbook's teaching examples.** This is good for seeing bootstrap/permutation
   mechanics clearly and quickly, but it also means every "significant" result in this
   lab should be read as a **worked example of the method**, not as a claim you should
   generalize to real professional-vs-blue-collar income differences today — the
   Duncan dataset is decades old and was collected under a specific historical
   methodology (a 1950s-era model built from survey and NORC prestige data), which the
   book itself notes should be treated as "a sample," not the current population.
4. **Confounding and measurement error are absent by construction.** Randomly generated
   data has no confounding variables, no missing-data mechanism, and no measurement
   noise beyond what we deliberately inject. Real-world analyses need additional steps
   (missing-data handling, confound adjustment, sensitivity analysis) this lab does not
   cover.
5. **Multiple comparisons are not addressed.** If you run permutation tests or
   bootstrap CIs across many variables or many group pairs (as one might in an
   exploratory real-world analysis), the chance of at least one false positive rises
   with each additional test. The lab presents each test in isolation; a real analysis
   with many comparisons would need a correction (e.g., Bonferroni, Benjamini-Hochberg).

**Bottom line:** this lab is an excellent environment for *learning the mechanics and
intuition* of these methods — because the generating process is known and controllable,
you can directly observe theorems (CLT, Empirical Rule) working and watch resampling
methods succeed or visibly struggle (small-n bootstrap instability, discretized
resampling). It is not, by itself, a template for publishing real-world statistical
conclusions — for that, add independence checks, larger/representative samples,
uncertainty about the generating process, and correction for multiple comparisons.

---

## References / Further Reading
- Efron, B. & Tibshirani, R. (1993). *An Introduction to the Bootstrap.*
- Good, P. (2005). *Permutation, Parametric, and Bootstrap Tests of Hypotheses.*
- Stevens, S. S. (1946). "On the Theory of Scales of Measurement." *Science*, 103(2684).
- Huy, H. (2023). *Building Statistical Models in Python*, Chapter 2 (source material
  for this lab's original examples and datasets).
