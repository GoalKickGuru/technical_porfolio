# Cheat Sheet — Leontief Input-Output Model in NumPy

Matches `01_practice_notebook.ipynb` task-by-task. Try each task yourself first —
use this only to check your work or get unstuck.

---

## Setup

```python
import numpy as np
from numpy.linalg import solve, inv, matrix_rank
np.set_printoptions(precision=2, suppress=True)
```

## Task 1 — Sectors and technical coefficient matrix

```python
sectors = ["Agriculture", "Manufacturing", "Services"]

A = np.array([[0.10, 0.20, 0.05],
              [0.30, 0.10, 0.15],
              [0.15, 0.25, 0.10]])
```

## Task 2 — Final demand vector

```python
d = np.array([100, 150, 120])
```

## Task 3 — Solve `(I - A)x = d`

```python
I3 = np.eye(3)
system_matrix = I3 - A
x = solve(system_matrix, d)
```

## Task 4 — Inter-industry transaction matrix

```python
Z = A @ np.diag(x)
```

## Task 5 — Simulate demand uncertainty

```python
np.random.seed(1)
n_scenarios = 1000

d_sim = np.column_stack([
    np.random.normal(100, 15, n_scenarios),
    np.random.normal(150, 25, n_scenarios),
    np.random.normal(120, 20, n_scenarios),
])
```

## Task 6 — Solve every scenario at once

```python
inv_system = inv(system_matrix)
x_sim = d_sim @ inv_system.T
```

## Task 7 — Summarize the simulation

```python
for i, s in enumerate(sectors):
    col = x_sim[:, i]
    mean = np.mean(col)
    std = np.std(col)
    mn = np.amin(col)
    mx = np.amax(col)
    print(f"{s:14s} mean={mean:8.2f}  std={std:6.2f}  min={mn:8.2f}  max={mx:8.2f}")

total_output = np.sum(x_sim, axis=1)
max_index = np.where(total_output == np.amax(total_output))

cv = np.std(x_sim, axis=0) / np.mean(x_sim, axis=0)
most_volatile = sectors[np.argmax(cv)]
```

---

## General-purpose NumPy reference (Chapter 3 techniques used in this project)

| Task | Function | Notes |
|---|---|---|
| Build a matrix from a table | `np.array([[...], [...]])` | rows first, nested lists |
| Identity matrix | `np.eye(n)` | needed for `I - A` |
| Matrix multiply | `A @ B` | **not** `A * B` (that's element-wise) |
| Vector → diagonal matrix | `np.diag(x)` | turns output vector into a diagonal matrix |
| Solve `A x = b` | `np.linalg.solve(A, b)` | preferred over computing `A⁻¹` by hand |
| Explicit inverse | `np.linalg.inv(A)` | only when you need the inverse itself, e.g. to apply to many `b` vectors at once |
| Matrix rank / invertibility check | `np.linalg.matrix_rank(A)` | rank `< n` ⇒ singular, `solve()` will fail |
| Random normal sample | `np.random.normal(mean, std, size)` | set `np.random.seed()` first for reproducibility |
| Stack columns into a 2D array | `np.column_stack([...])` | one array per sector/variable |
| Mean / median | `np.mean(x)` / `np.median(x)` | |
| Std dev / variance | `np.std(x)` / `np.var(x)` | |
| Min / max | `np.amin(x)` / `np.amax(x)` | |
| Find index of a value | `np.where(x == np.amax(x))` | returns a tuple of arrays |
| Index of max directly | `np.argmax(x)` | simpler than `np.where` when you just need one index |
| Sum across rows/scenarios | `np.sum(x, axis=1)` | `axis=1` sums each row (per-scenario total) |
| Sum down columns/sectors | `np.sum(x, axis=0)` | `axis=0` sums each column (per-sector total) |

---

## Common mistakes

- **Using `*` instead of `@`** for matrix multiplication — `*` multiplies
  element-by-element and silently gives the wrong (but shaped-correctly)
  answer.
- **Forgetting `np.random.seed()`** before a Monte Carlo block — your results
  won't be reproducible, which makes debugging and grading harder.
- **Column vs. row confusion** — `A[i, j]` here is *input from sector i into
  sector j*. Mixing up rows/columns silently transposes the whole model.
- **Treating `x_sim`'s spread as a real probability distribution** — it only
  reflects the standard deviations you chose in `DEMAND_STD`/Task 5, not
  measured real-world uncertainty. See the Limitations section.
