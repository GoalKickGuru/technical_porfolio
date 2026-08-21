# General Strategy: How to Approach an R Data Analysis Project

A step-by-step guide for tackling this project — or any similar project that combines control flow, descriptive statistics, and hypothesis testing in R.

## Step 0 — Read the whole brief before writing code
Skim every part first. Note which parts *depend* on earlier parts (e.g., you can't run hypothesis tests before you have the data simulated). Identify the deliverables and the checkpoint questions — those tell you what "done" looks like.

## Step 1 — Set up your notebook
- First code cell: load every library you'll need (`library(...)`), and call `set.seed()` if anything random is involved. Doing this once, at the top, avoids "it worked yesterday but not today" bugs.
- Keep one **scratch/markdown cell** at the top as a running to-do list mirroring the project's parts.

## Step 2 — Get or simulate the data before anything else
- If a dataset is provided, load and `str()`/`head()` it immediately so you know column names and types before writing any function.
- If you're simulating data, write the simulation code first, run it, and sanity-check it (`summary()`, a quick histogram) before building anything on top of it. A bug in your simulated data will quietly break every later step.

## Step 3 — Write small, testable functions early
- Every reusable piece of logic (a classification rule, a conversion, a test-and-interpret step) should be its own function with clear parameters and a `return()`.
- Test each function on 2–3 hand-picked inputs where you already know the expected answer, *before* applying it to the full dataset. This is the fastest way to catch off-by-one and logic errors.
- Prefer `if`/`else if`/`else` chains over deeply nested `if`s when you have more than 2 mutually exclusive outcomes — it reads top-to-bottom like a decision table.

## Step 4 — Replace loops with `apply()` once the logic works
- It's fine to prototype a transformation with a `for` loop while you're figuring out the logic.
- Once it works, ask: "am I building up a result vector one element at a time?" If yes, rewrite it as `sapply()`/`lapply()`/`apply()`. This is both more idiomatic R and less error-prone (no manual index management).
- Reserve actual `for`/`while` loops for cases with genuine state that carries across iterations (simulations, accumulators with a stopping condition) — Part 5-style stretch goals, not routine per-row transforms.

## Step 5 — Describe before you infer
- Always compute descriptive statistics (mean, sd, quartiles, IQR) for each group *before* running a hypothesis test on it. This builds intuition for what the test result "should" look like, and helps you catch a botched test call (e.g., swapped arguments) if the p-value contradicts what the descriptive stats obviously show.
- Use quartiles/tiers to spot skew or outliers that could violate a t-test's normality assumptions, especially with small samples.

## Step 6 — Pick the right hypothesis test, and say why
Ask, in order:
1. Am I comparing a group to one known/expected value? → **one-sample t-test**
2. Am I comparing exactly two groups? → **two-sample t-test**
3. Am I comparing three or more groups at once? → **ANOVA** (not repeated two-sample tests — that inflates Type I error)
4. Is the data categorical/frequency counts rather than numeric? → **chi-square** (not covered here, but keep in mind for future projects)

Before running the test, write your **H0** and **H1** in a markdown cell — this forces clarity about what "significant" would even mean here, and makes your final interpretation easier to write.

## Step 7 — Interpret results, don't just print them
A p-value alone is not an answer for a non-technical audience. For every test:
- State the p-value and your chosen α.
- State the conclusion in plain English (reject / fail to reject H0).
- State what that means for the *business question*, not just the statistical one.
- Note any caveat (sample size, whether ANOVA tells you *which* group differs, etc.).

## Step 8 — Automate repeated analysis
If you find yourself copy-pasting the same test/summary code for group A, then group B, then group C — stop, and instead:
- Put the groups in a named `list()`.
- Write one function that does the analysis for a single group.
- Run it across all groups with `lapply()`/`sapply()`.
This also makes it trivial to re-run the whole analysis if a 4th group gets added later.

## Step 9 — Write the executive summary last
Once every checkpoint is answered, write a short (1–2 paragraph) plain-English summary aimed at a non-technical stakeholder:
- What did you test?
- What did you find (with the key numbers)?
- What do you recommend, and what's the confidence/caveat behind that recommendation?

## Step 10 — Sanity-check the whole notebook top to bottom
Restart and "Run All." A notebook that only works when cells are run out of order will fail review and, more importantly, hides bugs from you. Fix anything that breaks before calling it done.

---

### Reusable checklist for future projects
- [ ] Libraries loaded & seed set at the top
- [ ] Data loaded/simulated and sanity-checked
- [ ] Each custom function tested on known inputs
- [ ] Loops replaced with `apply()` family where appropriate
- [ ] Descriptive stats computed before any inferential test
- [ ] Correct test chosen and justified (H0/H1 stated)
- [ ] Every test result interpreted in plain English
- [ ] Repeated per-group analysis automated, not copy-pasted
- [ ] Executive summary written
- [ ] Notebook runs cleanly top to bottom on a fresh restart
