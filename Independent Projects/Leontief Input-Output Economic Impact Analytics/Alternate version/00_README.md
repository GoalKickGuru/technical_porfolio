# How to Use This Package

Three files, meant to be used **in this order**:

| # | File | Purpose |
|---|---|---|
| 1 | `01_practice_notebook.ipynb` | Skeleton with `# TODO` cells + `assert` checks — do this first, to learn the technique by writing the code yourself |
| 2 | `03_cheat_sheet.md` | Reference snippets, one-to-one with the practice notebook's tasks — check your work here, don't start here |
| 3 | `02_reusable_template.ipynb` | A finished, parameterized version — reuse this for a *new* project by editing one input cell |

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
4. **Starting a new, similar project** (different sectors, different
   economy, different demand data)? Don't start from a blank notebook — copy
   `02_reusable_template.ipynb`, edit only the cell marked
   `# >>> EDIT HERE <<<` in Section 1, then Kernel → Restart & Run All.
   Sections 2–5 are generic and will validate your new inputs automatically.
5. **Before showing template results to anyone else**, re-read Section 5 of
   the template notebook (Limitations) with *your* new data in mind — the
   generic warnings there don't automatically know whether they apply to
   your specific case. That judgment call is still yours to make.

## What "reusable" means here, concretely

The template notebook works for **any Leontief-style, n-sector economy**, not
just the 3-sector agriculture/manufacturing/services example. To reuse it for,
say, a 5-sector city economy or a company's internal supply chain:

- Replace `SECTOR_NAMES` with your *n* sector/department names
- Replace `A` with your *n×n* technical coefficient matrix
- Replace `d` with your *n*-length final-demand vector
- Replace `DEMAND_STD` with your *n*-length uncertainty estimates (or zeros to
  skip the Monte Carlo step)

Everything past Section 1 — the validity checks, the solve, the transaction
matrix, the Monte Carlo simulation, the summary statistics — works unchanged
for any valid `n`, because it's written generically against `len(SECTOR_NAMES)`
rather than hardcoded to 3 sectors.

## When this whole approach is, and is not, the right tool

This bears repeating outside of just the template's own Limitations section,
because it applies to the *practice notebook and the technique itself*, not
just to one instance of the model:

**Good fit for:**
- Teaching/practicing linear algebra with NumPy on a concrete, motivating
  example (that's this notebook's primary purpose)
- A first-pass, short-run, directional estimate of how a demand shock ripples
  through a small number of sectors with roughly fixed input ratios
- Situations where you have (or can reasonably obtain) real technical
  coefficients, not placeholder numbers

**Poor fit / do not use for:**
- Real financial, investment, or policy decisions using the placeholder data
  in this notebook — that data is illustrative only, not measured from a real
  economy
- Long-run forecasts, or any period where technology, prices, or trade
  patterns are expected to shift materially
- Any question that depends on prices, substitution between inputs, or
  behavioral responses — this model has none of those
- Treating the Monte Carlo spread as a validated probability distribution —
  it only reflects the uncertainty widths you typed in, not measured risk
- Causal claims ("X caused sector Y to grow") — the model is mechanical and
  descriptive, with no error term and no way to rule out confounding

If your real project falls into the second list, this template is a
reasonable *starting point* to build intuition, but the final analysis you
present should use a richer model (e.g., an econometric model with estimated,
not assumed, uncertainty, or a full computable general equilibrium model) and
real input-output data from a statistical agency.
