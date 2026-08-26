# Extended Lab: Non-Parametric Hypothesis Testing

**Based on:** *Building Statistical Models in Python* (Huy Hoang Nguyen), Chapter 5 —
Non-Parametric Tests, plus the accompanying course notebooks on the permutation
test, rank-sum test, signed-rank test, Kruskal-Wallis test, chi-square tests,
and Spearman's correlation.

**Companion files:**
- `Skeleton_Practice_Notebook.ipynb` — do this lab from scratch
- `Solutions_Notebook.ipynb` — worked answers
- `Cheat_Sheet_Notebook.ipynb` — quick reference while you work
- `Reusable_Template_Notebook.ipynb` — generic starter for your own future projects
- `Background_Theory.md` — the theory, math, and limitations behind everything below

---

## 1. Learning Objectives

By the end of this lab you should be able to:

1. Recognize when a parametric test's assumptions (normality, equal variance,
   sufficient sample size, interval/ratio scale) are not credible, and justify
   that decision with plots and/or formal checks — not just assert it.
2. Choose the correct non-parametric test for a given experimental design
   (paired vs. independent groups; 2 groups vs. 3+; categorical vs. continuous;
   association vs. difference).
3. Run each of the seven tests below in Python, interpret the statistic and
   p-value correctly (including one-sided vs. two-sided), and state a
   plain-English conclusion.
4. Compute and interpret an **effect size** for each test family, not just a
   p-value.
5. Recognize the specific limitations of each test (what it does *not* tell
   you) and know what a reasonable next step or follow-up test would be.

## 2. Prerequisites

- Comfortable with `numpy`, `pandas`, basic `matplotlib`.
- Chapter 3–4 material: p-values, hypothesis testing logic (H0/Ha, Type I/II
  error, α), and the parametric tests these are alternatives to (one/two-sample
  t-tests, paired t-test, one-way ANOVA, Pearson correlation).
- Packages: `scipy`, `statsmodels`, `pyreadr` (for one dataset), `requests`.

## 3. The Seven Tests Covered

| # | Test | Non-parametric alternative to | Design |
|---|---|---|---|
| 1 | Permutation test | Any parametric test (esp. two-sample t-test) | 2 independent groups, any statistic |
| 2 | Mann-Whitney U (rank-sum) | Independent two-sample t-test | 2 independent groups |
| 3 | Wilcoxon signed-rank | Paired t-test | Paired/matched samples |
| 4 | Kruskal-Wallis | One-way ANOVA | 3+ independent groups |
| 5 | Chi-square goodness-of-fit | — | 1 categorical variable vs. expected counts |
| 6 | Chi-square test of independence | — | 2 categorical variables |
| 7 | Spearman's rank correlation | Pearson correlation | Association between 2 variables |

## 4. Lab Structure

Work through **Parts 1–7** in the skeleton notebook in order — each builds a
skill you'll reuse in the next part and in the Capstone. Datasets are drawn
directly from the source material so you can sanity-check your numbers against
the textbook's worked examples, then each part adds at least one **extension**
task that goes beyond the book.

### Part 1 — Permutation Test
Machine-failure counts at low vs. high temperature (n=7 vs. n=4). Build the
null distribution from scratch, then extend to a median-based statistic and
reason about why exhaustive permutation stops being feasible as n grows.

### Part 2 — Rank-Sum Test (Mann-Whitney U)
Real dataset (`gpa_iq.rda`, 78 students): compare IQ between students in the
top vs. bottom 10% of a self-concept test score. Check t-test assumptions
first, run `mannwhitneyu`, then reproduce the result by hand and compute a
rank-biserial effect size.

### Part 3 — Signed-Rank Test (Wilcoxon)
Nine patients' before/after treatment measurements. Run `wilcoxon`, reproduce
it by hand (ranks → S statistic → Z → p-value), compute an effect size, then
repeat the whole workflow on a second, larger simulated dataset of your own
design (call-center coaching scores).

### Part 4 — Kruskal-Wallis Test
Three independent groups (e.g., three production shifts). Run `kruskal`, then
extend into **post-hoc pairwise testing** (Mann-Whitney + Bonferroni
correction) — something the book doesn't cover but that you need in practice
whenever an omnibus test tells you groups differ but not which ones.

### Part 5 — Chi-Square Goodness-of-Fit
Phone-model sales vs. an equal-thirds expectation. Compute by hand, check
against `statsmodels`, compute Cohen's w, and run a power analysis to answer
"how many samples would I need to reliably detect this effect?"

### Part 6 — Chi-Square Test of Independence
Texas crash data: restraint use vs. fatality (a genuine 2x2 table with over a
million observations). Build the expected-frequency table by hand, compare
with/without Yates' correction, and compute Cramér's V — this dataset is a
great lesson in "statistically significant ≠ practically large."

### Part 7 — Spearman's Rank Correlation
Two judges scoring seven contestants. Compare Spearman vs. Pearson on the same
data, then build a bootstrap confidence interval for the correlation and
reflect on what a 7-point sample can and can't tell you.

### Capstone — Call Center Analytics
Four questions, one simulated dataset, **no hints about which test to use**.
You decide, justify, run it, and report an effect size — this is the part
that tests whether you actually learned the decision tree or just followed
along with worked examples.

## 5. Deliverables

Submit (or self-check against the Solutions notebook):

1. Completed `Skeleton_Practice_Notebook.ipynb` with every `# TODO` filled in.
2. For each of the 7 parts and 4 capstone questions: H0/Ha in words, test
   statistic + p-value, decision at α = 0.05, effect size, and a one-sentence
   plain-English conclusion.
3. A short (4-6 sentence) reflection (last cell of the notebook) on: which
   test was most/least intuitive, one case where you think you "paid a power
   cost" for using a non-parametric test, and one case where it was clearly
   the right tool.

## 6. Grading Rubric (suggested, 100 pts)

| Criterion | Points |
|---|---|
| Correct test chosen and justified for every scenario | 25 |
| Correct H0/Ha stated for every scenario | 10 |
| Correct computation (statistic, p-value) | 20 |
| Assumption checks shown (plots/tests) where relevant | 15 |
| Effect size computed and interpreted for every test | 15 |
| Capstone: correct test selection without scaffolding | 10 |
| Reflection quality | 5 |

## 7. Extension Ideas (optional, beyond this lab)

- Replace the normal-approximation p-values with **exact** methods
  (`method='exact'` in `mannwhitneyu`/`wilcoxon`) for the smallest datasets and
  compare.
- Implement a **Friedman test** (non-parametric repeated-measures ANOVA) on a
  3-condition paired design — natural next step after Wilcoxon/Kruskal-Wallis.
- Explore **Dunn's test** via the `scikit-posthocs` package as an alternative
  to the manual Bonferroni-corrected pairwise approach used in Part 4.
- Re-run Part 6 with a **much smaller** simulated 2x2 table (expected counts
  under 5) to see when Yates' correction actually changes the conclusion.

See `Background_Theory.md` for the full mathematical background, the
limitations of each test, and a discussion of what this class of methods can
and cannot simulate or answer reliably.
