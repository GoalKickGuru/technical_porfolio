# Background Theory: Z-Scores, Z-Tests, and Statistical Power

This document is the conceptual companion to the lab notebooks. Read it before (or alongside)
`01_extended_lab.ipynb`. It covers the theory behind every formula used in the lab, the
assumptions each method depends on, where the method breaks down, and what you can and cannot
legitimately conclude from simulations built on top of it.

---

## 1. The hypothesis-testing framework

Every test in this lab follows the same four-step recipe:

1. **State H₀ and Hₐ.** H₀ is the "nothing interesting is happening" claim (a parameter equals
   some value). Hₐ is its negation, and can be two-sided (`≠`) or one-sided (`<` or `>`).
2. **Pick α (the significance level)** — the probability you are willing to accept of rejecting
   H₀ when it is actually true. Common choices: 0.10, 0.05, 0.01. There is nothing sacred about
   0.05; it is a convention, not a law of nature.
3. **Compute a test statistic and a p-value (or compare the statistic to a critical value).**
   The p-value is the probability, *assuming H₀ is true*, of observing a statistic at least as
   extreme as the one you got.
4. **Decide:** if p ≤ α (equivalently, if the statistic falls in the critical region), reject H₀.
   Otherwise, fail to reject H₀. "Fail to reject" is not "accept" — absence of evidence is not
   evidence of absence.

### Type I and Type II errors

|                        | H₀ actually True        | H₀ actually False        |
|------------------------|--------------------------|---------------------------|
| **Don't reject H₀**    | Correct (prob. 1−α)      | Type II error (prob. β)   |
| **Reject H₀**          | Type I error (prob. α)   | Correct — "power" (1−β)   |

- **Type I error (α)**: a false positive. You claim an effect exists when it doesn't. This is a
  knob you set directly when you choose α.
- **Type II error (β)**: a false negative. You miss a real effect. β is *not* a free parameter —
  it falls out of your sample size, effect size, variability, and the α you chose. This is exactly
  what power analysis quantifies.
- **Power = 1 − β**: the probability of correctly detecting an effect that is really there. A
  power of 0.80 means that if the true effect is exactly the size you assumed, you'll detect it
  80% of the time across repeated studies of that design.

Increasing α inflates power (and Type I error). Increasing sample size inflates power without
touching α. This trade-off is the entire reason power analysis exists: it lets you choose a
sample size that gives you acceptable power *before* you collect data, rather than discovering
after the fact that your study was too small to detect anything.

---

## 2. The Z-score and the sampling distribution of the mean

A **z-score** re-expresses a single value in standard-deviation units relative to a distribution:

  z = (x − x̄) / σ

It answers "how many standard deviations is this point from the mean?" and lets you use one
universal table (the standard normal table) for any normally distributed variable regardless of
its original units.

A **z-statistic** is the analogous idea applied to a *sample mean* rather than a single
observation. The Central Limit Theorem tells us that, for a large enough sample, the sampling
distribution of x̄ is approximately normal with:

  E(x̄) = μ            (the sampling distribution is centered on the true population mean)
  σ_x̄ = σ / √n         (the "standard error" — spread shrinks as n grows)

So the test statistic for a one-sample test of the mean is:

  z = (x̄ − μ₀) / (σ / √n)

This is the same z-score formula, just applied to the mean of a sample instead of to a single
observation, using the *standard error* instead of the raw standard deviation.

### Why n matters twice
Sample size enters the picture in two independent ways that are easy to conflate:
1. It shrinks the standard error, which makes the test statistic larger (and the test more
   sensitive) for a fixed observed difference.
2. It is a direct input into every power/sample-size calculation later in the lab.

---

## 3. One-sample, two-sample, and proportion Z-tests

All of these are variations on "take a difference, divide by its standard error, compare to the
standard normal distribution":

| Test                        | Statistic                                                                 | What varies |
|------------------------------|----------------------------------------------------------------------------|-------------|
| One-sample mean               | z = (x̄ − μ₀) / (σ/√n)                                                     | one mean vs. a fixed value |
| Two-sample mean (independent)| z = (x̄₁ − x̄₂ − Ω) / √(σ₁²/n₁ + σ₂²/n₂)                                    | difference of two means |
| One-proportion                | z = (p̂ − p₀) / √(p₀(1−p₀)/n)                                              | one proportion vs. a fixed value |
| Two-proportion (pooled)       | z = (p̂₁ − p̂₂) / √(p̄(1−p̄)(1/n₁ + 1/n₂)), p̄ = pooled proportion            | difference of two proportions |

In every case the recipe is: **(estimate − hypothesized value) / standard error of the estimate**.
Once you see that pattern, t-tests, z-tests for proportions, and even more exotic tests you'll meet
later (chi-square, F) all feel like the same idea with a different standard-error formula and a
different reference distribution.

### p-values vs. critical values
These are two equivalent ways to reach the same decision:
- **p-value approach**: compute the probability of a statistic this extreme or more, compare to α.
- **Critical value approach**: compute the z-value that corresponds to α (via the inverse CDF,
  `norm.ppf`), and compare your observed statistic directly to that boundary.

For a two-tailed test, remember to split α across both tails (α/2 each) and to double the
one-tailed p-value.

---

## 4. Statistical power and effect size

**Effect size** is a standardized measure of "how big is the difference, in units that don't
depend on the original scale." The lab uses **Cohen's d** for means:

  d = |μ₁ − μ₂| / σ_pooled,   σ_pooled = √[ (n₁σ₁² + n₂σ₂²) / (n₁ + n₂) ]

Rough conventional benchmarks (Cohen, 1988): d ≈ 0.2 small, 0.5 medium, 0.8 large. These are
starting points for discussion, not universal truths — what counts as a "meaningful" effect is a
domain judgment, not a statistical one.

For chi-square goodness-of-fit tests, the analogous standardized effect size is **Cohen's w**,
computed from two sets of category proportions (`statsmodels.stats.gof.chisquare_effectsize`).

For ANOVA-style comparisons across k groups, Cohen's f plays the same role as d.

### The four-way relationship
Power, α, effect size, and sample size are linked — fix any three and the fourth is determined.
This is why `solve_power()` functions in `statsmodels` accept `None` for exactly one argument:
you're always solving for whichever one you leave open. The two most common uses in practice:
- **Given α, desired power, and an assumed/estimated effect size → solve for required n.**
  (Planning a study before collecting data — the responsible way to size an experiment.)
- **Given α, n, and effect size → solve for achieved power.**
  (Checking, after the fact or for a pilot, whether a design was ever capable of detecting the
  effect you care about.)

### Properties of power worth memorizing
- Power increases as sample size increases.
- Power increases as the true effect size increases (bigger differences are easier to detect).
- Power increases as α increases (a looser Type I error threshold makes rejection easier).
- Power increases as the variability (σ) of the underlying data decreases.

---

## 5. Why t-tests, chi-square, and F-tests show up alongside z-tests

The lab notebooks also touch `TTestPower`, `TTestIndPower`, `GofChisquarePower`, `FTestPower`,
and `FTestAnovaPower` from `statsmodels`. They belong in the same family:

- **t-test** is the z-test's sibling for when the *population* standard deviation is unknown and
  must be estimated from the sample — which is the normal state of affairs in real research. The
  t-distribution has heavier tails than the normal distribution, which is the mathematical
  correction for the extra uncertainty of estimating σ. As n grows, the t-distribution converges
  to the normal, and t-tests and z-tests give increasingly similar answers.
- **Chi-square goodness-of-fit** tests whether observed category counts match hypothesized
  proportions (e.g., market share by brand). Its power calculation uses Cohen's w and the number
  of bins instead of a mean difference.
- **F-test** compares two variances directly (or, in ANOVA, compares between-group to
  within-group variance across 2+ groups simultaneously). Power analysis for ANOVA
  (`FTestAnovaPower`) generalizes the two-group z/t-test power logic to k groups.

Seeing all of these side by side in one lab is intentional: the same four-way relationship
(power ↔ α ↔ effect size ↔ n) governs every one of them, only the specific effect-size formula and
reference distribution change.

---

## 6. Assumptions and limitations — read this before trusting any result

Every method in this lab rests on assumptions. Violating them doesn't necessarily make the
answer "wrong" by a huge margin, but it does mean the reported p-value, critical value, or power
number is no longer exactly what it claims to be.

### Assumptions behind the z-test specifically
1. **Population standard deviation (σ) is known.** In real problems this is almost never true
   — you'd typically use a t-test instead, substituting the sample standard deviation `s` for σ.
   The lab uses the z-test because σ is *given* in the textbook scenarios, which is a
   simplification for teaching purposes, not a realistic default.
2. **Normality.** The z-test assumes the underlying population (or, for large n, the sampling
   distribution of the mean via the CLT) is normal. For n > ~30 the CLT usually bails you out even
   if the raw data isn't perfectly normal — but "usually" is doing real work in that sentence;
   heavily skewed data or extreme outliers can still distort results at moderate sample sizes.
3. **Independent, randomly sampled observations.** Non-random sampling (convenience samples,
   self-selection) breaks the "scope of inference" — your conclusion only generalizes to the
   population you actually sampled from, not the population you wish you'd sampled from.
4. **For the two-sample z-test:** the two samples are independent of each other. A paired design
   (before/after on the same subjects) needs a paired test instead, or your standard error formula
   is wrong.
5. **For the pooled two-proportion test:** the pooled proportion p̄ is only a sensible summary
   when it's reasonable to believe the two groups share a common proportion under H₀.

### Assumptions behind power analysis
1. **Power analysis requires normally distributed data** (as flagged directly in the lab
   notebook). If your data is heavily non-normal, a textbook power formula derived under
   normality can be misleading; Monte Carlo / simulation-based power analysis (drawing many
   simulated samples from a specified non-normal distribution and empirically counting how often
   you'd reject H₀) is the standard workaround, and is demonstrated in the extended lab.
2. **The effect size you plug in is an assumption, not a fact.** Power calculations tell you
   "IF the true effect is this large, here's your power / required n." If your assumed effect
   size is wrong, the whole calculation is answering a question you didn't actually ask. Pilot
   studies, prior literature, or a minimum practically-meaningful effect size are the usual
   sources for this number — never a number picked to make the required sample size look small.
3. **Homogeneity of variance** for pooled-variance tests (independent two-sample t/z-tests,
   ANOVA power). If the true variances are very different between groups, pooling them
   understates or overstates the true standard error.
4. **The `ratio` parameter** (relative sample sizes between groups) needs to reflect what you can
   realistically collect — an unbalanced design changes the required total sample size non-trivially.
5. **Power analyses answer "would this design detect an effect if it exists," not "does the
   effect exist."** A well-powered study that finds no significant result is still meaningful
   evidence; an underpowered study that does find significance should be treated with real
   skepticism (the effect may be exaggerated — "the winner's curse").

### What this type of lab simulates well
- The mechanics of moving from raw data → test statistic → p-value / critical value → decision.
- How sample size, effect size, and variability trade off against detection power — this is the
  single most transferable intuition from the whole lab, and it applies far beyond z-tests.
- Planning: "how many samples do I need before I collect data" is a real, common, and valuable
  question that these tools answer directly and correctly under their assumptions.
- Comparing textbook, closed-form calculations (`scipy.stats`, `statsmodels.stats.power`) against
  each other to build intuition for what changes each formula responds to.

### What this type of lab does *not* capture well, and where to be careful
- **Real-world messiness**: missing data, measurement error, non-random sampling, and confounding
  variables are entirely outside the scope of a z-test or a power calculation. A perfectly powered,
  perfectly executed z-test on badly collected data still produces a badly justified conclusion.
- **Multiple comparisons**: running many tests (e.g., testing dozens of proportions) inflates the
  overall Type I error rate. None of the tests here correct for that; you'd need a Bonferroni-style
  adjustment or a different framework entirely.
- **Non-normal, small-sample, or heavy-tailed data**: the z-test's normality reliance is at its
  weakest exactly when it matters most (small n). This is the textbook motivation for switching to
  a t-test, a non-parametric test, or a simulation/bootstrap-based approach.
- **Assuming known σ in practice**: treat every "σ is known" scenario in the lab as a simplification.
  Real projects should default to a t-test unless there is a genuinely strong reason to believe the
  population standard deviation is known independently of the sample (e.g., a long-established,
  extremely stable manufacturing process).
- **Effect size honesty**: power analysis is trivially easy to misuse by picking an inflated effect
  size to justify a small, cheap sample. Treat the effect size input as the most important — and
  most defensible — number in the whole calculation.
- **Causality**: none of these tests, by themselves, establish that a difference is *caused* by
  the thing you think caused it. That requires experimental design (randomization, control groups),
  not just a significant p-value.

---

## 7. How the pieces fit together across the lab notebooks

- `3_1` (Z-score/Z-statistic basics) → the building block: standardizing values and sample means.
- `3_2` (Z-test for means) → applying that standardization to test hypotheses about one or two
  population means.
- `3_3` (Z-test for proportions) → the same logic applied to proportions instead of means.
- `4_statistical_power` → stepping back and asking "was my test even capable of finding an effect,
  and how many samples would I need for it to be?" — across t-tests, z-tests, chi-square tests,
  and F-tests/ANOVA.
- The extended lab, skeleton, cheat sheet, template, and solutions notebooks in this bundle build
  directly on top of these four ideas, adding new datasets, visualizations, a Monte Carlo power
  simulation, and a small end-to-end "plan → collect → test → report" case study so you practice
  the entire workflow, not just isolated formulas.
