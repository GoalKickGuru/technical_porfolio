# Background & Theory: Automobile Fuel Efficiency Analysis Lab

This document explains the concepts underlying the lab, why the workflow is built the way it is, and —
just as importantly — what this style of analysis can and cannot tell you.

---

## 1. The data science pipeline this lab follows

The original book chapter frames the work around a standard **data science pipeline**:

1. **Obtain** — download/acquire raw data (here, EPA's `vehicles.csv`).
2. **Scrub** — handle missing values, mixed types, inconsistent categories.
3. **Explore** — compute descriptive statistics, look at distributions and counts.
4. **Model** — in this lab, "modeling" is lightweight: grouped averages and trend lines, not a fitted
   statistical or ML model. (See §5 for how you'd extend this into real modeling.)
5. **Interpret** — turn numbers and plots into a claim in plain language, along with its caveats.

Nearly every EDA (exploratory data analysis) notebook you'll ever write follows this shape. The lab
deliberately keeps step 4 simple so the emphasis stays on steps 2, 3, and 5 — which is where most
real-world analysis time and most real-world mistakes actually happen.

---

## 2. Split-apply-combine

The core computational pattern used throughout (`groupby(...).mean()`, `groupby(...).agg(...)`) is
called **split-apply-combine**, a term popularized by Hadley Wickham (creator of R's `plyr`/`dplyr`):

- **Split** the data into groups based on some key (e.g., model year, or year + make).
- **Apply** a function independently to each group (e.g., compute the mean MPG).
- **Combine** the per-group results back into a single table.

This pattern generalizes far beyond this lab: it's the same logic behind SQL's `GROUP BY`, Excel
pivot tables, and Spark's `groupBy().agg()`. Once you're comfortable with it in pandas, you can
recognize and use it in almost any data tool.

**Why it matters here:** a raw scatter of `year` vs `comb08` (one point per *car model*, not per year)
is dominated by the sheer variety of vehicles offered in a given year — it doesn't answer "did average
efficiency go up?" You have to *split* by year, *apply* `mean()`, and *combine* into one row per year
before a trend becomes visible.

---

## 3. The Grammar of Graphics (and why we didn't use `ggplot` for Python)

The original book uses a Python clone of R's `ggplot2`, itself an implementation of Leland Wilkinson's
**Grammar of Graphics** — the idea that any chart can be decomposed into composable layers:

- **data** (a data frame),
- **aesthetic mappings** (which columns map to x, y, color, size...),
- **geometries** (points, lines, bars...),
- **facets** (small multiples split by a categorical variable),
- **scales**, **coordinate systems**, and **themes**.

This is a genuinely powerful mental model for thinking about visualization, independent of language.
However, the Python `ggplot` package used in the 2017 book was abandoned years ago and no longer
installs cleanly on modern Python — which is exactly why this lab reimplements every chart with
**matplotlib** (imperative, layer-by-layer object control) and **seaborn** (a statistical-plotting
layer on top of matplotlib with some grammar-of-graphics-like conveniences, e.g. `hue=`, `col=` faceting).

Conceptually, the "facet by make" small-multiples grid in this lab's Part 3 *is* an application of
Grammar-of-Graphics thinking — we're just building the facets with an explicit `plt.subplots()` loop
instead of a one-line `facet_wrap()`.

---

## 4. Why the specific cleaning steps are necessary

A few data-cleaning moves appear repeatedly in this lab; each corresponds to a general EDA principle:

| Step | General principle |
|---|---|
| `low_memory=False` on `read_csv` | Wide CSVs with genuinely mixed types per column (numbers *and* strings) will make pandas guess dtypes chunk-by-chunk unless told to scan the whole file first. |
| Dropping `NaN` before `.astype(float)` | Casting a column containing `NaN` alongside strings will error or silently misbehave; always handle missingness *before* type coercion. |
| Filtering out hybrids / EVs (`fuelType1`, `atvType`, `fuelType2`) | A single numeric column (`comb08`) can represent **physically different quantities** (MPG vs. MPGe) depending on other columns. Aggregating across an unfiltered mix produces numbers that are technically computable but semantically meaningless. |
| Reducing `trany` to a first-letter flag | High-cardinality string categories are often better analyzed as a coarser derived category; you trade detail for interpretability. |
| `set.intersection` across per-year makes | To make a fair year-over-year comparison of a specific make, you need it to be a "balanced panel" — present in every year — otherwise entry/exit of manufacturers confounds the trend. |

---

## 5. What this lab *can* simulate successfully

This lab is a good, low-friction way to practice and internalize:

- Reading messy real-world CSVs and diagnosing dtype/missing-value issues.
- The split-apply-combine pattern for descriptive aggregation.
- Multi-condition boolean filtering.
- Reshaping data (wide ↔ long) for multi-series comparison plots.
- Building small-multiples/faceted visualizations by hand.
- Translating a plain-language question ("did fuel economy improve?") into a specific, falsifiable
  sequence of pandas operations.
- Recognizing and defusing a **confounding-variable hypothesis** (hybrids inflating the average) by
  filtering and re-checking, which is a miniature version of real causal-inference reasoning.
- Comparing the R and Python approaches to the *same* analysis (if you also work through the book's
  R chapter), which builds transferable data-manipulation intuition independent of any one language.

It is a solid template for **any similarly-shaped project**: a wide CSV, a time dimension, a few
categorical dimensions, and a metric you want to track over time and segment by category (e.g., retail
sales by region and quarter, sports statistics by season and team, survey results by year and
demographic group). The `03_reusable_template.ipynb` notebook is built explicitly for that reuse.

---

## 6. Limitations — what this lab does *not* do, and shouldn't be mistaken for

**1. It is descriptive, not causal.** Every finding in this lab is of the form "X and Y moved together"
or "X differs across groups of Y." None of it establishes that engine downsizing, turbocharging, or
regulation *caused* the MPG increase — it only rules some explanations in or out as *consistent* or
*inconsistent* with the aggregate pattern. Real causal claims would need either a designed experiment
(not possible here — you can't randomly assign regulations to some car years and not others) or a
quasi-experimental method (e.g., difference-in-differences around the CAFE standard changes, instrumental
variables, or a regression that controls for confounders like vehicle class, weight, and turbocharging
directly).

**2. No statistical modeling or uncertainty quantification.** We compute means and plot them; we never
fit a regression, compute a confidence interval, or run a hypothesis test. A `mean()` computed from a
group of 40 cars and a `mean()` computed from a group of 4,000 cars are plotted identically here, even
though the two-car-model average is far noisier. A natural extension: fit
`comb08 ~ year + displ + cylinders + drive` with `statsmodels` or `scikit-learn`, and look at
coefficients, residuals, and confidence intervals rather than raw group means.

**3. Selection bias in "what's in the dataset."** `vehicles.csv` only contains models that were
**certified for U.S. sale** with an EPA fuel-economy rating. It excludes vehicles never sold in the
U.S., older vehicles before EPA testing began, and non-passenger vehicles like heavy trucks. Any trend
you find describes "U.S.-certified passenger vehicles as EPA-tested," not "all cars" or "real-world
driving," full stop.

**4. Test-cycle vs. real-world driving.** EPA `city08`/`highway08`/`comb08` figures come from
standardized dynamometer test cycles, not from telemetry of real drivers. Real-world fuel economy is
consistently a bit lower and varies more with driving style, climate, and terrain than the label
suggests — a well-known and openly acknowledged gap in the EPA's own methodology.

**5. Survivorship / reporting changes over time.** The `make` field, model taxonomy, and even which
columns exist (`atvType`, `phevBlended`, `startStop`) have expanded over the almost 40 years this file
covers, since EPA added categories as automotive technology changed (hybrids, plug-in hybrids, EVs).
Comparisons across the full 1984–present span implicitly compare a rapidly changing menu of
technologies, not a fixed population measured consistently over time.

**6. Confounded categorical comparisons.** The "MPG by drivetrain" exercise in Part 4 is a good example
of a comparison that *looks* like a clean categorical effect but is heavily confounded: 4WD/AWD vehicles
in the dataset skew toward larger, heavier vehicle classes (trucks, SUVs) that would have lower MPG
regardless of drivetrain. Concluding "front-wheel drive causes better MPG" from the boxplot alone would
be a classic omitted-variable-bias mistake. This is intentionally left as a challenge exercise so
students practice *noticing* the confound rather than just computing the number.

**7. No handling of duplicate/near-duplicate rows.** EPA's dataset lists trim-level variants of the same
underlying model as separate rows (e.g., 2WD vs 4WD versions, different transmissions). Depending on
your question, you may want to deduplicate by `(make, model, year)` before averaging, or you may want to
keep every certified configuration — this lab doesn't make that choice for you, and you should be
deliberate about it in your own projects.

**8. Single-dataset case study, not a general benchmark.** Everything above is scoped to this one CSV.
The general *techniques* (split-apply-combine, boolean filtering, wide/long reshaping, small multiples)
transfer directly to other tabular datasets, but the specific *findings* (efficiency trends, drivetrain
comparisons) are about the U.S. car market only and shouldn't be generalized elsewhere without new data.

---

## 7. Suggested next steps beyond this lab

- Fit `statsmodels.OLS` or `sklearn.linear_model.LinearRegression` for `comb08` on
  `year, displ, cylinders, drive, VClass` to get a *controlled* estimate of the year effect, holding
  engine size and vehicle class constant.
- Bring in an external CAFE-standard timeline dataset and overlay policy-change dates on the trend
  plots to visually connect regulation with the inflection points you found.
- Try a difference-in-differences design comparing manufacturers who entered/exited the 4-cylinder
  segment at different times.
- Repeat the whole workflow on a different domain dataset using `03_reusable_template.ipynb` to test how
  well the pattern really does generalize.
