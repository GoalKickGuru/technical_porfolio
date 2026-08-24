# Background: Sampling & Generalization

*Companion reading for the Extended Lab, Skeleton, Cheat Sheet, Template, and Solutions notebooks. Based on Chapter 1, "Sampling and Generalization," of* Building Statistical Models in Python*, extended with additional theory needed to interpret the lab's Monte Carlo results.*

---

## 1. Why sampling exists

The goal of statistical modeling is to answer a question about a **population** — every machine in a factory, every voter in an election, every plant in a field — by studying a smaller, manageable **sample**. Populations are usually too large, too expensive, or too ethically fraught to measure in full. Sampling exists to make inference practical without collecting data on every unit.

A sample is only useful if it is **representative**: it must reproduce the variation that exists in the population. A sample pulled from one corner of a field, one segment of voters, or one type of customer will not generalize, no matter how large it is. Representativeness — not size — is what makes a sample trustworthy.

## 2. Random sampling vs. random assignment

The book identifies two independent ingredients of a fully randomized experiment:

- **Random sampling (random selection):** how units are chosen *out of the population* to be observed. This is the subject of this lab.
- **Random assignment of treatments:** how *already-selected* units are allocated to different conditions (e.g., treatment vs. placebo). This is what turns an experiment into a source of causal, not just correlational, conclusions.

A study can have one, both, or neither ingredient:

| Random sampling | Random assignment | Study type | What you can conclude |
|---|---|---|---|
| Yes | Yes | Randomized experiment | Correlation *and* causation |
| Yes | No | Observational study | Correlation, population-level description |
| No | Yes | Randomized experiment on a non-representative group | Causation *within that group only* |
| No | No | Convenience/quota study | Description of the sample only |

This lab is entirely about the **random sampling** half of the table — it does not touch treatment assignment or causal inference. Chapter 3 of the book (hypothesis testing) is where the causal machinery is built on top of this foundation.

## 3. Four probability sampling strategies

In **probability sampling**, every unit has a known, nonzero chance of being selected. This is what makes formal statistical inference (standard errors, confidence intervals, p-values) valid.

### 3.1 Simple random sampling (SRS)

Every unit has an equal chance of selection, drawn without replacement (`rng.choice(..., replace=False)`). It is the most direct and least assumption-laden method, and the baseline every other method is compared against. Its main drawback is practical: for large, dispersed, or hard-to-enumerate populations, SRS can be slow or expensive to execute, and no built-in mechanism protects against random unlucky draws that miss an important subgroup.

### 3.2 Systematic sampling

Choose a random starting point, then take every *k*-th unit, where `k = N / n` is the sampling interval. It is fast, requires only a single random number, and is easy to implement on a physical list. Its risk is **periodicity**: if the population's natural ordering has a cycle that lines up with `k`, every selected unit will share whatever property recurs at that cycle, silently biasing the sample without any obvious warning sign in the code.

### 3.3 Stratified sampling

Split the population into homogeneous, non-overlapping **strata** based on a variable believed to relate to the outcome (plot, region, department, age group), then draw an independent SRS from each stratum, typically proportional to stratum size. Because between-stratum variation is removed from the sampling error entirely, stratified sampling usually has **lower variance** than SRS at the same sample size — the lab's Monte Carlo section demonstrates this directly. Its limitation: it requires the stratifying variable to be known in advance for every unit in the population, which is not always available.

### 3.4 Cluster sampling

Divide the population into naturally occurring **clusters** (stores, classrooms, geographic zones), randomly select whole clusters, and use every unit within the selected clusters (or sub-sample within them in a **two-stage** design). Cluster sampling is attractive when travel or listing cost is the main constraint — you only need to physically reach a handful of clusters rather than scattered individuals across the whole population. The cost is **higher variance**: because units within a cluster tend to be more similar to each other than to the population at large, a single unlucky cluster draw can swing the whole estimate. This lab's simulation shows cluster sampling remaining unbiased on average but visibly noisier than SRS or stratified sampling.

## 4. Non-probability sampling

In **non-probability sampling**, some or all units do not have a known or equal chance of selection. These methods are cheaper and faster, but the book is explicit: *"any information obtained and modeled from self-selected samples — or any non-random samples — cannot be used for inference."*

- **Convenience sampling** selects whichever units are easiest to reach. It is common in exploratory or qualitative work but risks systematic bias whenever "easy to reach" correlates with the outcome (e.g., the plants nearest the road being exposed to dust).
- **Quota sampling** fixes the number of units drawn from each subgroup (mirroring stratified sampling's proportions) but selects units *within* each subgroup non-randomly. It removes the coarse subgroup-imbalance problem of convenience sampling but not the underlying lack of randomization.

Both methods can still produce a numeric "confidence interval" or "p-value" if you run the formulas — the arithmetic doesn't know where the data came from — but the guarantees behind those numbers (e.g., "95% of intervals built this way contain the true parameter") only hold under probability sampling. Applying inferential statistics to a non-probability sample is a common and serious analytical error.

## 5. Bias, variance, and MSE — how the lab scores each method

Because the lab's population is simulated, the true mean is known, which lets us decompose each method's error into two independent pieces:

- **Bias** = average of many sample estimates − true population value. A method is biased if it is systematically wrong in the same direction, no matter how many times you repeat it.
- **Variance** = how much the estimate jumps around from one sample to the next. A method can be unbiased on average yet still unreliable in any single application if its variance is high (this is exactly cluster sampling's profile in the lab).
- **Mean squared error (MSE)** = bias² + variance, a single number that penalizes both problems and is used in the lab to rank methods.

In a real study you cannot compute bias directly, because you don't know the true population value — that is precisely why simulation-based teaching tools like this lab are valuable: they let you see what "biased" and "noisy" actually look like before you have to reason about them abstractly on a real, unknown population.

## 6. From sampling to inference: critical values, test statistics, and p-values

The book previews the machinery of hypothesis testing that Chapter 3 develops in depth. Three ideas anchor it, and the lab's confidence-interval section gives a first taste of each:

- **Test statistic:** a single number computed from the sample that measures how far the observed result is from what the null hypothesis predicts.
- **Critical value:** a threshold on the test statistic's distribution beyond which a result is considered too extreme to be explained by chance alone, set by the desired significance level (commonly 0.01, 0.05, or 0.10).
- **p-value:** the probability, under the null hypothesis, of observing a result as extreme as — or more extreme than — the one actually obtained. A small p-value relative to the significance level supports rejecting the null hypothesis.

The lab's 95% confidence interval for the mean uses the t-distribution's critical value (`scipy.stats.t.ppf`) in exactly this role, and its coverage-check experiment (repeating the interval construction 500 times) is a direct, hands-on demonstration of what "95% confidence" actually means: **over repeated probability sampling**, about 95% of the intervals constructed this way contain the true population parameter. This guarantee is inherited entirely from the random sampling step — it is not available to convenience or quota samples.

## 7. What this type of lab can and cannot simulate

**What a simulated-population lab is good for:**

- Demonstrating bias and variance directly, since the ground truth is known — impossible with real data.
- Showing sampling distributions (via Monte Carlo repetition) that would otherwise require hundreds of real, costly studies.
- Making abstract warnings concrete — e.g., actually watching systematic sampling drift when the population order has structure, or watching convenience sampling land off-center every single time.
- Illustrating confidence interval coverage as a long-run frequency property, not an abstract definition.
- Safely exploring "what if" scenarios (different sample sizes, different numbers of clusters, different strata definitions) at zero cost or risk.

**What this type of lab cannot simulate, and why:**

- **Real non-response and self-selection.** In the simulation every "plant" that is chosen is instantly and perfectly measured. Real populations include units that refuse to respond, are unreachable, or drop out — introducing bias mechanisms no synthetic random-noise model captures on its own.
- **Unknown or mismeasured strata/clusters.** The lab assumes `plot` and `cluster_id` are perfectly known for every unit. Real projects often must estimate or guess at a sensible stratifying variable, and errors in that choice propagate into the sampling design itself.
- **True cost and logistics trade-offs.** Cluster sampling's real-world appeal is lower travel/listing cost, not lower statistical error — the simulation shows the statistical side clearly but has no notion of actual cost, so it cannot show you the *trade-off* a real budget would force.
- **Causal inference.** Nothing in this lab involves random assignment of treatments. All six methods here only address how units are selected for *observation* — they say nothing about cause and effect, which is the subject of randomized experiments and Chapter 3's hypothesis testing.
- **Ethical and consent constraints.** Real sampling frequently cannot reach every population member for ethical reasons (e.g., studies involving vulnerable groups). The simulation has no notion of consent, so it cannot illustrate how ethical constraints reshape what "probability sampling" is even achievable in practice.
- **Model misspecification of the population itself.** The simulated field's confounders (forest, irrigation, road) are simple, additive, and known by construction. Real populations have unknown numbers of confounders interacting in unknown ways — the lab's clean decomposition into bias/variance is a best-case scenario, not a guarantee of what real analyses will look like.

Used with these limitations in mind, the lab is a safe, fast way to build correct intuition about *why* each sampling method behaves the way it does — intuition you can then carry into real projects using the Template notebook, while remembering that real data will violate some of the lab's simplifying assumptions.

## 8. Where this leads next

Chapter 2 of the book (*Distributions of Data*) develops the theory behind the sampling distributions this lab generates empirically via Monte Carlo — the Central Limit Theorem, standard error formulas, and the shapes those histograms are approximating. Chapter 3 (*Hypothesis Testing*) formalizes the critical-value/test-statistic/p-value framework introduced in Section 6 above, and adds **power analysis** — using existing sample data to decide how large a future sample needs to be for a desired level of statistical confidence.
