# Cheat Sheet — Informal Lending Market Model in NumPy

Matches `01_practice_notebook.ipynb` task-by-task. Try each task yourself first —
use this only to check your work or get unstuck.

---

## Setup

```python
import numpy as np
from numpy.linalg import solve
np.set_printoptions(precision=2, suppress=True)
```

## Part 1 — Task 1: Effective annual rate, vectorized

```python
fee_rate = 0.10  # 10% flat fee per loan cycle

terms_weeks = np.arange(1, 9)
cycles_per_year = 52 / terms_weeks
EAR = (1 + fee_rate) ** cycles_per_year - 1
```

## Part 2 — Task 2: Build the repayment matrix `A`

```python
terms = np.array([1, 2, 3])
rates = np.array([0.10, 0.08, 0.06])
n = len(terms)

A = np.zeros((n, n))
for i in range(1, n + 1):
    for j in range(n):
        if i <= terms[j]:
            A[i-1, j] = 1/terms[j] + rates[j]
```

## Part 2 — Task 3: Obligation vector `b`

```python
b = np.array([5000, 4000, 3000])
```

## Part 2 — Task 4: Solve for the lending amount per product

```python
x = solve(A, b)
check = A @ x   # should equal b
```

## Part 3 — Task 5: Simulate defaults across scenarios

```python
np.random.seed(1)
n_loans = 200
principal = 100
flat_rate = 0.10
default_prob = 0.15
recovery_rate = 0.30
n_scenarios = 2000

performing_payoff = principal * (1 + flat_rate)
defaulted_payoff = principal * recovery_rate

defaults = np.random.random((n_scenarios, n_loans)) < default_prob
collections_per_loan = np.where(defaults, defaulted_payoff, performing_payoff)
```

## Part 3 — Task 6: Total collections per scenario

```python
total_collections = np.sum(collections_per_loan, axis=1)
```

## Part 3 — Task 7: Summarize the risk

```python
total_principal_lent = n_loans * principal
backer_obligation = 21000

mean_collections = np.mean(total_collections)
std_collections = np.std(total_collections)
p10 = np.percentile(total_collections, 10)
p90 = np.percentile(total_collections, 90)

prob_loss = np.mean(total_collections < total_principal_lent)
prob_cant_repay_backer = np.mean(total_collections < backer_obligation)
```

---

## General-purpose NumPy reference (Chapter 3 techniques used in this project)

| Task | Function | Notes |
|---|---|---|
| Term/period index array | `np.arange(start, stop)` | one entry per loan term, e.g. 1–8 weeks |
| Vectorized rate conversion | `(1 + rate) ** periods` | replaces a `for` loop entirely — same idea as Listing 3.2's runtime comparison |
| Build a matrix from a word problem | `np.zeros((n, n))` + fill by index | same approach as Table 3.2 (mesh analysis) and the retirement bond ladder |
| Solve `A x = b` | `np.linalg.solve(A, b)` | preferred over computing `A⁻¹` by hand |
| Check a solution | `A @ x` | should reproduce `b` (up to floating-point rounding) |
| Matrix rank / solvability check | `np.linalg.matrix_rank(A)` | rank `< n` ⇒ singular, `solve()` will raise an error |
| Random uniform sample | `np.random.random(size)` | compare against a probability to build a boolean "did this happen" array |
| 2D random array | `np.random.random((n_scenarios, n_loans))` | one row per scenario, one column per loan — enables per-scenario, per-loan simulation without nested loops |
| Elementwise choice | `np.where(condition, if_true, if_false)` | applies a different value to each element depending on a boolean array |
| Sum across a chosen axis | `np.sum(x, axis=1)` | `axis=1` sums each row (per-scenario total across all loans) |
| Mean of a boolean array | `np.mean(bool_array)` | gives the *fraction* of `True` values — handy for probabilities like "P(loss)" |
| Percentile | `np.percentile(x, q)` | `q=10` / `q=90` give a rough "bad case / good case" range |

---

## Common mistakes

- **Confusing the flat fee with the effective annual rate.** A "10% fee" and
  a "10% EAR" are very different numbers once the term is short — that gap
  is the entire point of Part 1.
- **Off-by-one errors in the repayment matrix.** `A[i-1, j]` because arrays
  are 0-indexed but months are 1-indexed. Getting this backwards silently
  shifts every cash flow by one period.
- **Using `*` instead of `@` to check `A @ x == b`.** `*` multiplies
  element-by-element and will not reproduce `b` even when `x` is correct.
- **Building the default simulation with a Python `for` loop over loans.**
  `np.random.random((n_scenarios, n_loans))` draws the entire simulation at
  once — no loop needed, and it's what makes 2,000 scenarios fast.
- **Treating `prob_loss` or `EAR` values from this notebook as facts about a
  real market.** They only reflect the assumptions typed into the code —
  see the Limitations section, especially the legal and ethical notes.
