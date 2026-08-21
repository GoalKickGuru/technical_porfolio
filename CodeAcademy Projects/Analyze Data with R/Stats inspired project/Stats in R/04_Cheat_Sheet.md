# R Cheat Sheet: Control Flow, Quartiles, and Hypothesis Testing

Quick reference for the BrewTime Coffee project (and similar R projects).

---

## 1. Control Flow

### if / else / else if
```r
if (condition) {
  # runs if condition is TRUE
} else if (other_condition) {
  # checked only if the first condition was FALSE
} else {
  # runs if nothing above was TRUE
}
```
- Conditions must evaluate to a single `TRUE`/`FALSE`.
- Comparison operators: `<  >  <=  >=  ==  !=`
- Logical operators: `&` (AND), `|` (OR), `!` (NOT)
- Once one branch fires, the rest are **never evaluated** ("short-circuit").
- You can **nest** if/else blocks inside each other — the inner block only runs if the outer condition was TRUE.

### for loops
```r
for (i in 1:n) { ... }                 # counts a fixed number of times
for (item in some_vector) { ... }      # iterates elements of a vector/list
for (row in 1:nrow(m)) {               # nested loop over a matrix
  for (col in 1:ncol(m)) { ... }
}
```
- Number of iterations = length of the sequence/vector, not a guess.
- The loop variable only exists **inside** the loop.
- Outer-loop-first order determines traversal order (row-major vs. column-major).

### while loops
```r
while (condition) {
  # body MUST eventually make condition FALSE, or this never stops
}
```
- Always double check: does something inside the loop move the condition toward FALSE?
- Classic bug: forgetting to update the loop variable → infinite loop.

### Functions
```r
my_function <- function(param_1, param_2 = default_value) {
  # body
  return(some_value)     # optional if it's the last evaluated expression
}
```
- **Parameter** = name in the definition. **Argument** = value passed at call time.
- Give default values for optional parameters: `function(x, alpha = 0.05)`.
- Keep functions small and single-purpose so they're reusable across projects.

---

## 2. The `apply()` Family (use instead of loops when possible)

| Function | Input | Output | When to use |
|---|---|---|---|
| `apply(X, MARGIN, FUN)` | matrix/data frame | vector/matrix | `MARGIN = 1` → by row, `MARGIN = 2` → by column |
| `sapply(X, FUN)` | vector/list | simplified vector/matrix | you want a clean vector back |
| `lapply(X, FUN)` | vector/list | list | you want to preserve list structure |
| `vapply(X, FUN, FUN.VALUE)` | vector/list | typed vector | you want type safety/speed |
| `mapply(FUN, x, y)` | multiple vectors | vector/list | function needs more than one argument at a time |

```r
apply(my_matrix, 1, sum)     # row sums
apply(my_matrix, 2, sum)     # column sums
sapply(my_vector, my_fun)    # vector out
lapply(my_vector, my_fun)    # list out
```

**Rule of thumb:** if you catch yourself writing `for (x in vec) { result <- c(result, f(x)) }`, that's almost always better as `sapply(vec, f)`.

---

## 3. Quartiles & Percentiles

```r
quantile(x, 0.25)                       # Q1
quantile(x, 0.5)                        # Q2 = median
quantile(x, 0.75)                       # Q3
quantile(x, c(0.25, 0.5, 0.75))         # all three at once
quantile(x, c(0.1, 0.2, ..., 0.9))      # deciles
IQR(x)                                  # Q3 - Q1
```

- **By hand:** sort the data. Q2 is the median of the whole set. Q1 is the median of the lower half; Q3 is the median of the upper half. If the "half" has an even count, average the two middle values.
- **Outlier rule of thumb:** flag any value `< Q1 - 1.5*IQR` or `> Q3 + 1.5*IQR`.
- Quartiles split data into 4 equal-size groups; percentiles/deciles generalize this to 100 or 10 groups.

---

## 4. Hypothesis Testing

### Core vocabulary
| Term | Meaning |
|---|---|
| **Population mean** | The true average across the entire group of interest (usually unknown). |
| **Sample mean** | The average of a subset (sample) — used to *estimate* the population mean. |
| **Null hypothesis (H0)** | The "nothing is different / no effect" statement. |
| **Alternative hypothesis (H1)** | The "there is a difference / an effect" statement. |
| **p-value** | Probability of seeing data this extreme (or more) *if H0 were true*. Small p-value = evidence against H0. |
| **Significance level (α)** | The threshold you decide *before* testing (commonly 0.05). If p-value < α, reject H0. |
| **Type I error** | False positive — rejecting H0 when it was actually true. Its probability is α. |
| **Type II error** | False negative — failing to reject H0 when it was actually false. |

### Choosing a test

| Test | Use when… | R code |
|---|---|---|
| **One-sample t-test** | Comparing one group's mean to a known/expected value | `t.test(x, mu = expected_value)` |
| **Two-sample t-test** | Comparing the means of exactly two groups | `t.test(group1, group2)` |
| **ANOVA** | Comparing means of three or more groups at once | `aov(y ~ group, data = df)` then `summary(result)` |
| **Chi-square** | Categorical / frequency data (not covered in this project) | `chisq.test()` |

### Reading a `t.test()` result
```r
results <- t.test(sample_a, sample_b)
results$p.value       # the number you compare to alpha
results$estimate      # the sample mean(s)
results$conf.int      # confidence interval for the difference
```

### Reading an `aov()` result
```r
results <- aov(sales ~ group, data = daily_sales)
summary(results)                     # look at Pr(>F) — that's your p-value
summary(results)[[1]][["Pr(>F)"]][1] # pull the p-value programmatically
```

### Why not just run lots of t-tests?
Every individual test carries a Type I error risk of α. Run enough of them and the *combined* risk of at least one false positive climbs fast:
```r
1 - (1 - alpha)^n_tests
```
ANOVA tests all groups **at once**, keeping the overall Type I error rate at α. But a significant ANOVA only tells you *some* group differs — it doesn't say which one (that needs a follow-up test, e.g. pairwise t-tests with a correction, which is beyond this project's scope).

### Interpreting a p-value in plain English
```r
interpret_test <- function(test_result, alpha = 0.05) {
  if (test_result$p.value < alpha) {
    print("Reject the null hypothesis: there is a statistically significant difference.")
  } else {
    print("Fail to reject the null hypothesis: no statistically significant difference found.")
  }
}
```

---

## 5. Common Gotchas

- R indexing starts at **1**, not 0.
- `vec[i]` returns a length-1 vector/sublist; `list[[i]]` returns the actual element.
- Mixing types in `c()` coerces everything to the most flexible type (numbers become strings if any string is present).
- Matrices must be **homogeneous** (one data type); data frames can mix types per column.
- `matrix()` fills **column-by-column** by default — use `byrow = TRUE` to fill row-by-row.
- `set.seed(n)` before any `rnorm()`/`sample()` call makes "random" simulations reproducible — always set it once at the top of a script/notebook.
