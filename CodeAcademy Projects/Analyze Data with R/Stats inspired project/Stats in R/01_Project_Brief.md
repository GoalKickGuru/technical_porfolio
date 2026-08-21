# Capstone Project: BrewTime Coffee — Loyalty App Rollout Analysis

**Language:** R
**Estimated time:** 4–6 hours
**Builds on:** Control Flow in R (if/else, loops, functions, `apply()` family), Quartiles & Percentiles, Hypothesis Testing with R (t-tests, ANOVA)

---

## 1. Business Scenario

You are the data analyst for **BrewTime Coffee**, a coffee chain with stores across three test regions. Corporate wants to know whether a new **loyalty app** increases daily sales, and whether the effect is worth a full rollout.

Three groups of stores took part in a 90-day trial:

| Group | Description |
|---|---|
| **A — Control** | No app. Business as usual. |
| **B — App Basic** | Rolled out a basic loyalty app (points only). |
| **C — App Premium** | Rolled out a premium loyalty app (points + personalized offers). |

Corporate has also set a **daily sales target of $5,000/store** that every store is expected to hit on average, independent of the trial.

Your job: simulate the trial data, describe it, and use statistical inference to tell corporate (1) whether stores are meeting target, (2) whether the app helped at all, and (3) whether the premium version is worth the extra cost over the basic version.

---

## 2. Learning Objectives

By completing this project you will practice:

- **Control flow** — `if` / `else if` / `else`, nested conditionals, `for` and `while` loops, and writing your own functions.
- **The `apply()` family** — `sapply()`, `lapply()`, `apply()` — to avoid repetitive loops when processing groups of stores.
- **Quartiles & percentiles** — `quantile()`, `IQR()`, building a data-driven tiering system, and outlier detection.
- **Hypothesis testing** — sample vs. population mean, p-values, significance level, Type I/II errors, one-sample t-test, two-sample t-test, and ANOVA (with the multiple-comparisons problem it solves).

---

## 3. Dataset

You will **simulate** the dataset yourself (no external file needed) — this mirrors how the original course exercises used `rnorm()` to build practice datasets.

```r
set.seed(42)
n_days <- 90
corporate_target <- 5000

sales_A <- rnorm(n_days, mean = 5000, sd = 600)   # Control
sales_B <- rnorm(n_days, mean = 5150, sd = 650)   # App Basic
sales_C <- rnorm(n_days, mean = 5400, sd = 700)   # App Premium
```

You will reshape these three vectors into one tidy data frame with columns: `day`, `weekday`, `is_weekend`, `group`, `sales`.

---

## 4. Project Parts

### Part 1 — Data Simulation & Control Flow
1.1. Generate `sales_A`, `sales_B`, `sales_C` as shown above.
1.2. Write a function `day_of_week(day_index)` that uses `%%` (modulo) and `if`/`else if`/`else` to return the weekday name ("Monday" … "Sunday") for a day index starting at day 1 = Monday.
1.3. Write a function `is_weekend(day_index)` that returns `TRUE`/`FALSE` using an `if`/`else`.
1.4. Use `sapply()` (not a hand-written loop) to build a `weekday` vector and an `is_weekend` vector for all 90 days.
1.5. Combine everything into one data frame called `daily_sales` with columns `day`, `weekday`, `is_weekend`, `group`, `sales` (90 rows per group, 270 rows total). Hint: build one data frame per group and `rbind()` them, or use `rep()` for the `group` column.

### Part 2 — Descriptive Statistics & Quartiles
2.1. For each group, compute Q1, Q2 (median), and Q3 of `sales` using `quantile()`.
2.2. Compute the `IQR()` for each group.
2.3. Write a function `classify_tier(value, q1, q2, q3)` that uses **nested if/else** to return one of `"Low"`, `"Below Average"`, `"Above Average"`, `"High"` depending on which quartile bucket `value` falls into.
2.4. Use `sapply()` to apply `classify_tier()` to every row of `daily_sales$sales` (using that row's own group's quartiles) and add the result as a new column `tier`.
2.5. Write a function `flag_outliers(values)` that returns a logical vector flagging any value below `Q1 - 1.5*IQR` or above `Q3 + 1.5*IQR`. Apply it per group.
2.6. **Checkpoint question:** which group has the widest IQR, and what does that tell you about consistency vs. Group A?

### Part 3 — Hypothesis Testing
3.1. **One-sample t-test.** Test whether Group A's mean daily sales differs significantly from `corporate_target` (mu = 5000). State your null and alternative hypotheses first, then run `t.test()`, then interpret the p-value at α = 0.05.
3.2. **Two-sample t-test.** Test whether Group B's mean sales are significantly different from Group A's. Repeat for Group C vs. Group A, and Group C vs. Group B.
3.3. Write a function `interpret_test(test_result, alpha = 0.05)` that takes the object returned by `t.test()` and uses `if`/`else` on `test_result$p.value` to `print()` a plain-English interpretation ("Reject the null hypothesis…" / "Fail to reject the null hypothesis…").
3.4. **The multiple-comparisons problem.** You just ran three separate two-sample t-tests. Compute the approximate probability of at least one Type I error across those three tests (`1 - (1 - alpha)^n`). This is why we don't stop here.
3.5. **ANOVA.** Run `aov(sales ~ group, data = daily_sales)` and pull the p-value with `summary()`. State the conclusion at α = 0.05.
3.6. **Checkpoint question:** if the ANOVA is significant, what can and can't you conclude from it alone? (Hint: revisit the "Key Concept" section of the original Hypothesis Testing material.)

### Part 4 — Automating the Analysis with `apply()`
4.1. Build a **named list** of the three sales vectors: `list(A = sales_A, B = sales_B, C = sales_C)`.
4.2. Use `lapply()` to compute a summary (mean, sd, Q1, Q2, Q3) for every group in one call — write a helper function `summarize_group(x)` that returns a named vector, and pass it to `sapply()` so the result comes back as a clean matrix/data frame instead of a list.
4.3. Use `sapply()` with a custom function to compute the p-value of "Group X vs. corporate target" for every group in one line (no copy-pasted `t.test()` calls).

### Part 5 — Stretch Goals (optional)
5.1. **While loop simulation:** simulate flipping a biased "coin" where landing on Group C's app increases the chance of a day's sales beating target. Use a `while` loop to count how many simulated days it takes to reach 10 "wins."
5.2. **Nested loop + matrix:** build a 3×90 matrix of the three sales vectors (stores as rows, days as columns) and use `apply()` with `MARGIN = 1` to get each store's total sales, and `MARGIN = 2` to get each day's combined sales across stores.
5.3. Write a two-paragraph **executive summary** in a markdown cell, written for a non-technical VP: what did you find, and what do you recommend?

---

## 5. Deliverables

- A completed `.ipynb` notebook (start from the skeleton) with all code cells filled in and all checkpoint questions answered in markdown cells.
- Every custom function documented with a one-line comment describing what it does.
- A final "Executive Summary" markdown cell.

## 6. Grading Rubric (self-check)

| Criteria | Weight |
|---|---|
| Control flow used correctly (if/else/else if, no `for` loops used where `sapply`/`lapply` was requested) | 20% |
| Custom functions are reusable (take parameters, `return()` a value) | 15% |
| Quartile/tiering logic correct and applied via `apply()` family | 20% |
| Hypothesis tests correctly chosen, run, and interpreted | 30% |
| Clear write-up / checkpoint answers / executive summary | 15% |
