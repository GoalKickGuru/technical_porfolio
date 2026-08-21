# How to Use This Package

Three files, meant to be used **in this order**:

| # | File | Purpose |
|---|---|---|
| 1 | `01_practice_notebook.ipynb` | Skeleton with `# TODO` cells + `assert` checks — do this first, to learn the technique by writing the code yourself |
| 2 | `03_cheat_sheet.md` | Reference snippets, one-to-one with the practice notebook's tasks — check your work here, don't start here |
| 3 | `02_reusable_template.ipynb` | A finished, parameterized version — reuse this for a *new* project by editing a few input cells |

## What this package studies

A retirement plan broken into the three questions people actually ask, each
solved with a different NumPy technique from Chapter 3:

1. **"How much will I have by retirement?"** — a vectorized compound-growth
   calculation (no `for` loop), the same kind of technique the chapter uses
   to show NumPy arrays outperforming plain Python loops (Listing 3.2).
2. **"How do I fund my first few years of withdrawals with certainty?"** — a
   **bond ladder / cash-flow-matching** problem, which is a genuine linear
   system `A·x = b`, solved with `np.linalg.solve()` exactly like the
   mesh-current network and input-output examples in the chapter.
3. **"What's the risk I run out of money?"** — a **Monte Carlo simulation**
   of 1,000 possible 30-year retirements with random market returns, summarized
   with the statistical functions from Section 3.1.5 (`mean`, `median`,
   `percentile`).

## Step-by-step

1. **Open `01_practice_notebook.ipynb`** in Jupyter (`jupyter notebook` or
   `jupyter lab`, or upload to Google Colab/Jupyter within Claude). Run the
   Setup cell, then work through Tasks 1–7 top to bottom. Each task cell has
   an `assert` check below it — a passing `assert` (no error) means you're
   ready for the next task.
2. **Stuck for more than a few minutes on a task?** Open `03_cheat_sheet.md`,
   find the matching task heading, compare your code to the snippet — don't
   just copy it in without understanding the difference.
3. **Finish Section 8 (Limitations) in writing**, even though it has no code.
   This is the part of the exercise most people skip, and it's the part most
   likely to prevent a real mistake later.
4. **Starting a new, similar project** (a different saver, a different
   ladder, different withdrawal assumptions)? Don't start from a blank
   notebook — copy `02_reusable_template.ipynb`, edit only the
   `# >>> EDIT HERE <<<` cells in Sections 1, 2, and 4, then Kernel →
   Restart & Run All. Section 3's validity checks run automatically.
5. **Before showing template results to anyone else**, re-read Section 5 of
   the template notebook (Limitations) with *your* new numbers in mind — the
   generic warnings there don't automatically know whether they apply to
   your specific case. That judgment call is still yours to make.

## What "reusable" means here, concretely

The template notebook works for **any accumulation target, any n-bond
ladder, and any withdrawal/return assumptions**, not just the example numbers.
To reuse it for a different plan:

- Section 1: change `initial_balance`, `monthly_contribution`,
  `annual_return`, `years_to_retirement`
- Section 2: change `coupons`, `maturities`, and `liabilities` — as long as
  all three arrays stay the same length, you can ladder any number of years
- Section 4: change `annual_withdrawal`, `mu`, `sigma`, `n_years`,
  `n_scenarios`, `random_seed`

Everything else — the growth-factor math, the matrix construction, the
solvability checks, the Monte Carlo loop, the summary statistics — is written
generically against these inputs and works unchanged.

## When this whole approach is, and is not, the right tool

This applies to the *practice notebook and the technique itself*, not just
one instance of the model — it's worth repeating outside the notebooks' own
Limitations sections:

**Good fit for:**
- Teaching/practicing vectorized arrays, linear systems, and Monte Carlo
  simulation in NumPy on a concrete, motivating example (this notebook's
  primary purpose)
- A rough, order-of-magnitude check on whether a savings plan is broadly on
  track, or comparing the *relative* effect of one change (save more, retire
  later, spend less)
- The bond-ladder piece specifically, for its real, narrow purpose: locking
  in known near-term withdrawals with high-quality bonds — this part is
  genuinely low-risk arithmetic, not a statistical guess

**Poor fit / do not use for:**
- Any real, irreversible retirement decision (retiring, annuitizing, rolling
  over a pension) based on this notebook's numbers alone
- Long-run certainty about market returns — `annual_return`, `mu`, and
  `sigma` are typed-in assumptions, not measured facts about the future
- Anything involving taxes, inflation on the accumulation side, Social
  Security or pension income, healthcare costs, or required minimum
  distributions — none of these are modeled at all
- Treating the Monte Carlo "probability of ruin" as historically validated —
  it only reflects the normal-distribution assumption and the specific `mu`/
  `sigma` you chose. Real markets have fatter tails than a normal
  distribution predicts, so this model likely **understates** real risk
- Bonds that aren't high-quality/investment-grade — the ladder assumes zero
  default risk, which is not a safe assumption for lower-rated bonds

If your real situation falls into the second list, this template is a
reasonable *starting point* to build intuition and to know what questions to
ask, but the final plan you act on should come from a fuller tool or a
financial professional working with your actual numbers, tax situation, and
goals.
