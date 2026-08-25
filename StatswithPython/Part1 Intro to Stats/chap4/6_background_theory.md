# Background Theory: Parametric Hypothesis Testing

*Companion reference for the Parametric Testing Lab (extended lab, skeleton, cheat sheet, template, and solutions notebooks). Based on Chapter 4, "Parametric Tests," of Huy Hoang Nguyen's* Building Statistical Models in Python*, extended with additional theoretical context, limitations, and simulation guidance.*

---

## 1. What makes a test "parametric"?

A parametric hypothesis test assumes the data come from a distribution belonging to a known family — described by a fixed, finite set of parameters (e.g., the normal distribution, defined entirely by its mean and standard deviation). Because the test statistic's sampling distribution is derived analytically from that assumed family, parametric tests are:

- **Powerful** — when their assumptions hold, they generally require fewer observations than non-parametric alternatives to detect a real effect of a given size.
- **Assumption-dependent** — their validity (correct Type I error rate, meaningful p-values and confidence intervals) rests on those assumptions actually holding, at least approximately.

This lab covers the z-test's natural extensions: the **t-test family**, **ANOVA**, and **Pearson's correlation coefficient**, plus the supporting machinery of **multiple-comparison correction** and **power analysis**.

## 2. The three core assumptions

### 2.1 Normality

Parametric tests built around the mean (t-tests, ANOVA) implicitly assume the population is normally distributed, because they treat the mean as the single best summary of central tendency. If the population is meaningfully skewed, the mean no longer represents "the middle" of the data well, and inferences about mean differences become unreliable.

**Why there's some robustness anyway.** Degrees-of-freedom corrections (used in the t-distribution, and via the Welch-Satterthwaite adjustment in ANOVA/Welch's t-test) inflate the assumed variance of the sampling distribution to account for uncertainty in small samples. A useful side effect is that the resulting central-tendency estimate becomes *more* representative of the true center of a mildly-skewed distribution than it would be without that correction. This is the theoretical basis for saying t-tests and ANOVA are "fairly robust to violations of normality" — particularly as sample size grows (by the Central Limit Theorem, the sampling distribution of the *mean* approaches normality even when the underlying population isn't normal).

**How to check it:**
- *Visual:* histograms, and especially Q-Q (quantile-quantile) plots, which compare your data's quantiles against a theoretical normal distribution's quantiles along a 45-degree reference line.
- *Formal tests:*
  - **Kolmogorov-Smirnov** — compares the empirical CDF to a reference normal CDF; most sensitive near the center of the distribution; requires the data to be centered (mean 0) and scaled (sd 1) first; performs best with large samples.
  - **Anderson-Darling** — a weighted variant of the KS idea that up-weights the tails, making it more sensitive to outliers and heavy-tailed departures from normality; also best with large samples.
  - **Shapiro-Wilk** — the most general-purpose test and the best choice for small samples (roughly n < 50); its main weakness is the opposite of KS/AD: with very large samples, it becomes so sensitive that it will flag trivially small, practically meaningless deviations from perfect normality as "significant."

### 2.2 Independence

Independence means no single observation's value influences, or is systematically related to, another observation's value beyond what's explained by the model. Two common violations:

1. **Subgroup contamination** — a sample nominally drawn from "the population" actually over- or under-represents a subgroup with systematically different responses (e.g., accidentally sampling mostly one manufacturer's cars when trying to characterize "cars from Origin X" broadly).
2. **Serial (auto-)correlation** — observations collected close together in a sequence (time, trial order) are not independent because one occurrence influences the likelihood or value of the next.

**Detecting serial correlation:** the **Durbin-Watson** statistic tests first-order (lag-one) autocorrelation. It ranges from 0 to 4: a value near 2 indicates no autocorrelation, near 0 indicates strong positive autocorrelation, and near 4 indicates strong negative autocorrelation. Crucially, Durbin-Watson is only meaningful when the order of your rows reflects a genuine sequence (time stamps, trial numbers) — running it on arbitrarily-ordered cross-sectional data (like rows of a car dataset sorted by manufacturer name) produces a statistic with no real interpretation.

**Detecting subgroup contamination:** this cannot be verified from the data alone — no statistical test looks at a dataset and tells you "this was sampled badly." It is a **study-design** issue, addressed through the sampling methodology itself (random sampling, stratification, documentation of the sampling frame), not through post-hoc statistical correction. Parametric tests have essentially no robustness to violations of this assumption — a few incidental departures may not matter much, but systematic violations invalidate the test's conclusions regardless of what the p-value says.

### 2.3 Equal variance (homogeneity of variance)

When comparing two or more group means, classic (pooled-variance) t-tests and ANOVA assume the groups share a common population variance. This matters because these tests use a pooled estimate of variance to compute the standard error of the mean difference; if the true variances differ substantially, that pooled estimate is biased and the test's Type I error rate can be distorted (usually inflated when the smaller group has the larger variance).

**Testing for it:**
- **Levene's test** — compares variances across two or more groups; robust to non-normality, which makes it the standard first choice.
- **Fisher's F-test** — the classic ratio-of-variances test for exactly two groups (larger variance in the numerator); more sensitive to non-normality than Levene's test.

**Robustness via the Welch-Satterthwaite adjustment.** Rather than assuming equal variances, **Welch's t-test** and **Welch's ANOVA** recompute the degrees of freedom based on each group's own variance and sample size, producing valid inference without the equal-variance assumption. For large, well-powered samples, Welch's versions give results very close to their classic counterparts — which is why many modern statistical tools (including `statsmodels`' `anova_oneway`, whose default is `use_var='unequal'`) now default to the Welch version rather than assuming equal variance.

## 3. The tests, one at a time

### 3.1 t-tests

| Variant | Compares | Key formula idea |
|---|---|---|
| One-sample | A sample mean to a fixed value | \\(t = (\\bar{x} - \\mu_0) / (s/\\sqrt{n})\\), df = n − 1 |
| Two-sample, pooled | Two independent group means, equal variance | Uses a pooled standard deviation \\(s_p\\) combining both groups' variances; df = n₁ + n₂ − 2 |
| Two-sample, Welch's | Two independent group means, unequal variance | Uses each group's own variance separately; df computed via Welch-Satterthwaite (usually non-integer) |
| Paired | Two measurements on the same subjects | Reduces to a one-sample t-test on the *differences* between paired observations |

The t-distribution itself has heavier tails than the standard normal distribution, reflecting the extra uncertainty introduced by estimating the population standard deviation from a (possibly small) sample rather than knowing it exactly. As sample size (and therefore degrees of freedom) grows, the t-distribution converges to the standard normal — which is why the t-test effectively subsumes the z-test as a special case with unknown population variance.

### 3.2 Multiple comparisons and ANOVA

Running k independent hypothesis tests at significance level α each gives a family-wise error rate (FWER) — the probability of at least one false positive across the family — that grows with k, roughly as \\(1-(1-\\alpha)^k\\). With three pairwise comparisons at α = 0.05, the FWER is already close to 0.14, nearly three times the nominal rate.

**Bonferroni correction** controls this by requiring each individual test to clear a stricter threshold, \\(p_i \\le \\alpha/m\\) for m tests. It is simple, general-purpose (works for any kind of hypothesis test, not just means), and conservative — meaning it trades away some statistical power (raising the Type II error rate) in exchange for tightly controlling the Type I error rate. Less conservative alternatives (Holm's step-down method, Benjamini-Hochberg's false discovery rate control) exist for situations where Bonferroni's conservatism costs too much power.

**ANOVA** takes a different approach for the specific case of comparing means across 3+ groups: rather than running multiple pairwise tests, it partitions total variance into between-group and within-group components and tests a single omnibus null hypothesis ("all group means are equal") using an F-statistic (the ratio of between-group to within-group variance). This sidesteps the multiple-comparisons problem for the *initial* question, but ANOVA alone cannot tell you *which* specific group(s) differ — that requires a follow-up **post-hoc test**.

**Post-hoc testing (Tukey's HSD).** After a significant ANOVA, Tukey's Honestly Significant Difference test compares all pairs of group means while directly controlling the family-wise error rate for that specific set of comparisons (rather than correcting p-values from separately-run tests after the fact). It also produces simultaneous confidence intervals for each pairwise mean difference, giving both a decision (significant or not) and an effect-size-like magnitude in the original units.

### 3.3 Pearson's correlation coefficient

Pearson's r measures the strength and direction of **linear** association between two continuous variables, ranging from −1 (perfect negative) through 0 (no linear association) to +1 (perfect positive). It is the standardized version of covariance:

\\[ r = \\frac{S_{xy}}{S_x S_y} \\]

**Assumptions are lighter than for t-tests/ANOVA**: Pearson's r does not require normality or equal variance. It does require independent, paired, continuous observations with finite variance (heavy-tailed distributions without finite variance can produce unstable, misleading correlation estimates).

**What it cannot tell you:**
- **Causation.** A strong, statistically significant correlation says nothing about which variable (if either) causes the other, or whether a third variable drives both.
- **Non-linear relationships.** A variable pair with a strong U-shaped or otherwise non-linear relationship can show an r near 0 despite an obviously strong (non-linear) association. Always inspect a scatterplot alongside the numeric r.
- **Practical importance from significance alone.** With large samples, even trivially small correlations (r = 0.05, say) can produce very small p-values. r² (the coefficient of determination) — the proportion of variance in one variable "explained" by the other — is a better guide to practical magnitude than the p-value.

### 3.4 Power analysis

Statistical power is the probability of correctly rejecting a false null hypothesis (1 − the Type II error rate, β). Power depends jointly on:

1. **Effect size** — the magnitude of the true difference/relationship, standardized (Cohen's d for mean differences).
2. **Sample size** — larger samples shrink standard errors, making smaller true effects detectable.
3. **Significance level (α)** — a stricter α (smaller, e.g., 0.01 instead of 0.05) reduces power for a fixed sample size and effect size.

Power analysis is used in two directions: (a) **prospectively**, to determine the sample size needed to reliably detect an effect of a hypothesized size before running a study, and (b) **retrospectively**, to understand how much confidence a completed "fail to reject" result actually deserves — a non-significant result from a severely underpowered study is weak evidence of "no effect," not proof of it.

## 4. Limitations of this lab and what it can (and cannot) simulate successfully

### 4.1 What this lab demonstrates well

- **Textbook-clean assumption violations.** The synthetic examples (skewed vs. normal distributions, sine-wave vs. random-noise sequences, deliberately mismatched variances) are constructed to make each assumption's violation and its detection unambiguous. This is ideal for building intuition about *what each test statistic is actually sensitive to*.
- **The mechanics of test selection.** Walking the same real dataset (Auto MPG) through the full "check assumptions → pick the right variant → run it → correct for multiple comparisons → check effect size/power" pipeline builds the habit of treating test selection as a *conditional decision process* rather than reaching for a single default test.
- **The gap between statistical and practical significance.** With n = 392 (Auto MPG) or n = 1,000 (synthetic examples), nearly every comparison in this lab is "statistically significant" — which is precisely why effect sizes (Cohen's d, r²) are emphasized throughout as the more informative number for real decisions.

### 4.2 Where this lab's simulations fall short of real-world complexity

- **Assumption checks in isolation vs. in combination.** Real projects often face *simultaneous* violations — non-normal **and** unequal-variance **and** modestly non-independent data all at once — where the individual remedies (Welch's adjustment, robust tests, transformations) may not compose cleanly. This lab mostly demonstrates one violation at a time for pedagogical clarity.
- **Small-sample edge cases.** Several formal normality tests (KS, Anderson-Darling) are explicitly *underpowered* on small samples — they can fail to detect real non-normality simply because there isn't enough data to distinguish it from sampling noise. A "fail to reject normality" result on n=10-20 observations is much weaker evidence of actual normality than the same result on n=1,000. This lab's larger datasets can create a false sense of how decisive these tests typically are.
- **Static, cross-sectional data vs. genuine time series.** The Durbin-Watson demonstrations use clean synthetic sequences (sine wave, white noise). Real time-ordered data — economic indicators, sensor streams, repeated-measures designs — often has richer autocorrelation structure (seasonal patterns, trends, higher-order lags) that a first-order Durbin-Watson statistic alone cannot fully characterize; specialized time-series methods (ACF/PACF plots, ARIMA diagnostics) are needed there.
- **The independence-of-sampling assumption cannot be validated computationally.** No amount of code run on a dataset can confirm that its sampling process avoided subgroup contamination; this always requires documented, external knowledge of how the data was collected. This lab can only illustrate the *concept* of this failure mode, not detect it algorithmically.
- **Bonferroni's conservatism compounds with many groups.** The lab's examples use small families (2-3 groups). With larger families (say, 20+ pairwise comparisons), Bonferroni's power cost becomes severe, and the choice between Bonferroni, Holm, and false-discovery-rate methods (only briefly mentioned here) becomes practically important — a topic this lab doesn't explore in depth.
- **Causal claims remain out of scope.** Every correlation, mean difference, and ANOVA result in this lab is purely associational. The Auto MPG "weight explains variance in MPG" observation, for instance, is suggestive but not a causal claim — a fuller treatment (and the natural next step, foreshadowed in the capstone) requires regression modeling with appropriate confounding-variable control, which is outside this chapter's scope.
- **Non-parametric alternatives are only referenced, not covered.** When normality truly cannot be defended (even after transformation), the appropriate next step is a non-parametric test (Mann-Whitney U, Kruskal-Wallis, Spearman's rank correlation) — the subject of the book's next chapter, not this lab.

### 4.3 Practical guidance for extending this lab to your own data

1. Treat assumption checks as informing a **judgment call**, not a strict gate — mild violations combined with reasonable sample sizes are often acceptable given these tests' documented robustness; severe violations (heavy skew, tiny samples, known non-independence) warrant either a transformation, a Welch/robust variant, or a switch to a non-parametric test.
2. Always report an effect size next to a p-value — a p-value alone cannot distinguish "large sample, trivial effect" from "small sample, large effect."
3. When comparing 3+ groups, prefer ANOVA + post-hoc testing over repeated pairwise t-tests when your question is really "which means differ" — it's a cleaner, more standard workflow, and its FWER control is more direct than manually correcting after the fact.
4. Before trusting any "fail to reject" result, check whether the study had adequate power to detect an effect size you'd actually care about; an underpowered null result is not evidence of "no effect."
5. Remember that Pearson's r only ever tells you about *linear* association — plot your data before trusting the number.

---

*This document accompanies `1_extended_lab.ipynb`, `2_skeleton_practice.ipynb`, `3_cheat_sheet.ipynb`, `4_reusable_template.ipynb`, and `5_solutions.ipynb`. The same theory is summarized more concisely as inline markdown within the extended lab notebook itself.*
