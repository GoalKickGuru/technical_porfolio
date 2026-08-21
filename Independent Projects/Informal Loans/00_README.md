# How to Use This Package

Three files, meant to be used **in this order**:

| # | File | Purpose |
|---|---|---|
| 1 | `01_practice_notebook.ipynb` | Skeleton with `# TODO` cells + `assert` checks — do this first, to learn the technique by writing the code yourself |
| 2 | `03_cheat_sheet.md` | Reference snippets, one-to-one with the practice notebook's tasks — check your work here, don't start here |
| 3 | `02_reusable_template.ipynb` | A finished, parameterized version — reuse this for a *new* scenario by editing a few input cells |

## What this package studies, and what it's for

Informal lending (moneylenders, rotating savings groups, short-term cash
lenders operating outside the formal banking system) is a real, widely
studied part of the financial system in development and financial-inclusion
economics. This package is built for **analysis, financial literacy, and
research** — quantifying the true cost of informal credit and the risk it
carries — the same way an academic researcher, a regulator, or a
consumer-protection program would.

It is deliberately **not** built, and should not be extended, to help design
loan terms, pricing, or collection practices aimed at extracting more money
from borrowers. Each notebook's own Limitations section says this again in
more detail — read it, it's not boilerplate.

Three questions, each solved with a different NumPy technique from Chapter 3:

1. **"How expensive is a short-term informal loan, really?"** — a vectorized
   conversion from a quoted flat fee to an effective annual rate (EAR), the
   same kind of technique the chapter uses to show NumPy arrays replacing
   `for` loops (Listing 3.2).
2. **"How should a small lender mix loan products to meet its own funding
   obligations?"** — a cash-flow-matching linear system `A·x = b`, solved
   with `np.linalg.solve()`, structured exactly like the mesh-current
   network and retirement bond-ladder examples.
3. **"How much default risk does an informal loan portfolio carry?"** — a
   Monte Carlo simulation of loan defaults across a portfolio, summarized
   with the statistical functions from Section 3.1.5 (`mean`, `std`,
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
3. **Finish Section 8 (Limitations) in writing**, even though it has no
   code. In this notebook especially, this section carries real legal and
   ethical content, not just modeling caveats — don't skip it.
4. **Studying a new, similar scenario** (different fees, terms, products, or
   default assumptions)? Don't start from a blank notebook — copy
   `02_reusable_template.ipynb`, edit only the `# >>> EDIT HERE <<<` cells
   in Sections 1, 2, and 4, then Kernel → Restart & Run All. Section 3's
   validity checks run automatically.
5. **Before drawing any conclusion from template results**, re-read Section
   5 of the template notebook (Limitations) with *your* new numbers in
   mind — the generic warnings there don't automatically know whether they
   apply to your specific case. That judgment call is still yours to make.

## What "reusable" means here, concretely

The template notebook works for **any flat-fee/term combination, any
n-product loan mix, and any default/recovery assumptions**, not just the
example numbers. To reuse it for a different scenario:

- Section 1: change `fee_rate` and `terms_weeks`
- Section 2: change `terms`, `rates`, and `obligation` — as long as all
  three arrays stay the same length, you can model any number of products
  and months
- Section 4: change `n_loans`, `principal`, `flat_rate`, `default_prob`,
  `recovery_rate`, `n_scenarios`, `random_seed`, `backer_obligation`

Everything else — the annualization math, the matrix construction, the
solvability checks, the Monte Carlo simulation, the summary statistics — is
written generically against these inputs and works unchanged.

## When this whole approach is, and is not, the right tool

This applies to the *practice notebook and the technique itself*, not just
one instance of the model:

**Good fit for:**
- Teaching/practicing vectorized arrays, linear systems, and Monte Carlo
  simulation in NumPy on a concrete, motivating example (this notebook's
  primary purpose)
- Consumer-protection-style analysis: showing why a small quoted fee on a
  short-term loan translates into an enormous effective annual rate — a
  standard, well-established critique in financial-literacy and regulatory
  work, not a novel technique
- Academic or policy research into default risk and portfolio economics of
  informal credit, using clearly-labeled illustrative assumptions

**Poor fit / do not use for:**
- Designing real loan pricing, terms, or collection strategy — this package
  computes costs and risk; it should not be extended into a tool for
  extracting more from borrowers
- Any real lending or borrowing decision without local legal advice — usury
  limits, licensing rules, and consumer-protection law vary by jurisdiction
  and are not modeled here at all
- Justifying or normalizing real informal-lending harms — coercive
  collection, debt bondage, and over-indebtedness cycles are documented,
  serious problems in under-regulated informal credit markets, and no
  spreadsheet-level model can capture or excuse them
- Treating the fee rates, default probabilities, or recovery rates in this
  package as facts about any real market — they are illustrative
  placeholders; real analysis needs real data (household surveys,
  microfinance institution records, published research)

If your real interest is in this area professionally — as a researcher, a
regulator, or someone designing a responsible community lending program —
this template is a reasonable starting point for building the technical
intuition, but the analysis you act on or publish should be built on real
data and reviewed against the relevant law and ethical standards for your
context, not on this notebook's placeholder numbers.
