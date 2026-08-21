# Cheat Sheet — Retirement Plan Model in NumPy

Matches `01_practice_notebook.ipynb` task-by-task. Try each task yourself first —
use this only to check your work or get unstuck.

---

## Setup

```python
import numpy as np
from numpy.linalg import solve
np.set_printoptions(precision=2, suppress=True)
```

## Part 1 — Task 1: Vectorized accumulation phase

```python
initial_balance = 20000
monthly_contribution = 500
annual_return = 0.07
years_to_retirement = 30

monthly_rate = (1 + annual_return)**(1/12) - 1
months = np.arange(0, years_to_retirement * 12 + 1)

growth_factors = (1 + monthly_rate) ** months
balance_from_initial = initial_balance * growth_factors
balance_from_contrib = monthly_contribution * ((growth_factors - 1) / monthly_rate)
balance = balance_from_initial + balance_from_contrib
```

## Part 2 — Task 2: Build the cash-flow matrix `A`

```python
coupons = np.array([0.03, 0.035, 0.04])
maturities = np.array([1, 2, 3])
n_bonds = len(coupons)

A = np.zeros((3, 3))
for i in range(1, 4):
    for j in range(n_bonds):
        m = maturities[j]
        c = coupons[j]
        if i < m:
            A[i-1, j] = c
        elif i == m:
            A[i-1, j] = c + 1
```

## Part 2 — Task 3: Liability vector `b`

```python
b = np.array([40000, 42000, 44000])
```

## Part 2 — Task 4: Solve for face values

```python
x = solve(A, b)
check = A @ x   # should equal b
```

## Part 3 — Task 5: Set up the Monte Carlo simulation

```python
np.random.seed(1)
n_scenarios = 1000
n_years = 30
start_balance = balance[-1]
annual_withdrawal = 40000
mu, sigma = 0.06, 0.12

bal = np.full(n_scenarios, start_balance)
depletion_year = np.full(n_scenarios, -1)
```

## Part 3 — Task 6: Run the year-by-year loop

```python
for year in range(1, n_years + 1):
    returns = np.random.normal(mu, sigma, n_scenarios)
    bal = bal * (1 + returns) - annual_withdrawal
    newly_ruined = (bal <= 0) & (depletion_year == -1)
    depletion_year[newly_ruined] = year
    bal = np.maximum(bal, 0)
```

## Part 3 — Task 7: Summarize results

```python
prob_ruin = np.mean(depletion_year != -1)
ending_balance = bal

median_bal = np.median(ending_balance)
mean_bal = np.mean(ending_balance)
p10 = np.percentile(ending_balance, 10)
p90 = np.percentile(ending_balance, 90)

ruined = depletion_year[depletion_year != -1]
avg_ruin_year = np.mean(ruined) if len(ruined) else None
```

---

## General-purpose NumPy reference (Chapter 3 techniques used in this project)

| Task | Function | Notes |
|---|---|---|
| Month/year index array | `np.arange(start, stop)` | one entry per period, e.g. `np.arange(0, 361)` for 360 months |
| Vectorized compound growth | `(1 + rate) ** periods` | replaces a `for` loop entirely — see Listing 3.2's runtime comparison |
| Future value of an annuity (vectorized) | `contribution * ((growth_factors - 1) / rate)` | growth_factors is an array, so this returns the whole balance history at once |
| Build a matrix from a word problem | `np.zeros((n, n))` + fill by index | same approach as Table 3.2 (mesh analysis) |
| Solve `A x = b` | `np.linalg.solve(A, b)` | preferred over computing `A⁻¹` by hand |
| Check a solution | `A @ x` | should reproduce `b` (up to floating-point rounding) |
| Matrix rank / solvability check | `np.linalg.matrix_rank(A)` | rank `< n` ⇒ singular, `solve()` will raise an error |
| Random normal sample | `np.random.normal(mean, std, size)` | set `np.random.seed()` first for reproducibility |
| Elementwise max (a floor) | `np.maximum(array, 0)` | clips every element at 0, keeps ruined scenarios from going further negative |
| Boolean mask + assignment | `arr[mask] = value` | only updates the elements where `mask` is `True` |
| Mean of a boolean array | `np.mean(bool_array)` | gives the *fraction* of `True` values — handy for probabilities |
| Median | `np.median(x)` | |
| Percentile | `np.percentile(x, q)` | `q=10` and `q=90` give a rough "bad case / good case" range |
| Filter an array by condition | `x[x != -1]` | keep only elements matching a condition, e.g. only ruined scenarios |

---

## Common mistakes

- **Using the annual rate directly on monthly data.** `annual_return` must be
  converted to `monthly_rate` with `(1 + annual_return)**(1/12) - 1` before
  it's used in a monthly loop or array — using it directly overstates growth.
- **Off-by-one errors in the bond ladder matrix.** `A[i-1, j]` because arrays
  are 0-indexed but years are 1-indexed (year 1 is row 0). Mixing this up
  silently shifts every cash flow by one year.
- **Drawing all years of Monte Carlo returns in one call outside the loop
  without a `(n_years, n_scenarios)` shape.** If you flatten years and
  scenarios together you lose the per-year, per-scenario structure. Keep the
  loop over years and vectorize only across scenarios each year, or use an
  explicit 2D array of shape `(n_years, n_scenarios)` if you vectorize fully.
- **Forgetting `np.maximum(bal, 0)` after subtracting the withdrawal.**
  Without it, a "ruined" scenario can go deeply negative and then randomly
  recover in a later year, which doesn't happen in real life (you can't earn
  interest on debt you don't have in a retirement account).
- **Treating `prob_ruin` as a scientifically validated number.** It only
  reflects the `mu`/`sigma` you chose — see the Limitations section.
