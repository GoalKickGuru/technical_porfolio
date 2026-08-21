# Computational Economics with SciPy
### A Data Analytics Project Applying Numerical Methods to Economic Modeling

**Based on:** Steinkamp, V. — *Python for Engineering and Scientific Computing* (2024), Chapter 6, "Numerical Computations and Simulations Using SciPy"

---

## 1. Motivation

Chapter 6 of the source text builds a toolkit of numerical methods — root-finding,
optimization, interpolation, differentiation, integration, differential equations,
and Fourier analysis — using engineering and physics examples (damped oscillations,
free fall, filters, bearing vibration). **Every one of these methods has a direct,
widely-used analogue in economics and quantitative finance.** This project keeps the
exact SciPy machinery from the book and re-targets each example at an economic
question, so you practice both the numerical method and the economic reasoning
behind it.

| SciPy technique (book §) | Economic application (this project) |
|---|---|
| `scipy.optimize.root` (§6.1) | Finding the market-clearing (equilibrium) price |
| `scipy.optimize.minimize` (§6.2) | Cost-minimizing input mix for a firm (Cobb–Douglas) |
| `scipy.interpolate` (§6.3) | Building a continuous yield curve from bond quotes |
| `numdifftools` / finite differences (§6.4) | Marginal cost, marginal revenue, elasticity |
| `scipy.integrate.quad` (§6.5) | Consumer surplus & producer surplus |
| `scipy.integrate.solve_ivp` (§6.6) | Solow–Swan economic growth model |
| `scipy.fft` (§6.7) | Detecting the business cycle in GDP data |
| Growth/epidemic ODEs (§6.11) | Bass diffusion model of technology adoption |

## 2. Learning Objectives

By the end of this project you will be able to:

1. Translate an economic problem (equilibrium, optimization, growth, cycles,
   diffusion) into a mathematical model suitable for SciPy.
2. Use `scipy.optimize` for root-finding and constrained optimization.
3. Build interpolants with `scipy.interpolate` and judge when linear vs. cubic
   interpolation is appropriate for financial data.
4. Compute derivatives and integrals numerically and connect them to marginal
   analysis and surplus measures from microeconomics.
5. Solve ordinary differential equations describing economic dynamics with
   `solve_ivp` and interpret steady states.
6. Use the Fast Fourier Transform to extract cyclical structure from a noisy
   economic time series.
7. Present all results as a small, reproducible data-analytics notebook with
   plots, interpretation text, and a short written conclusion.

## 3. Deliverables

1. **`econ_scipy_project_SKELETON.ipynb`** — your starting point. Contains
   background theory, a full narrative, synthetic datasets already generated,
   and empty `# TODO` code cells for you to complete.
2. A completed notebook (rename to `econ_scipy_project_<yourname>.ipynb`) with
   all TODOs filled in, all plots produced, and the short-answer questions
   answered in markdown cells.
3. (Optional/instructor use) `econ_scipy_project_SOLUTION.ipynb` — a full
   worked solution / cheat sheet. Use it to check your work or get unstuck,
   not as a copy-paste source.

## 4. Data

All datasets are **synthetically generated with a fixed random seed** directly
inside the notebook (no external files or internet access required), so your
results will exactly match the solution notebook if you use the same seed.
This mirrors how the textbook's examples are self-contained and reproducible.

## 5. Grading Rubric (suggested, 100 pts)

| Section | Points |
|---|---|
| 1. Market equilibrium (root-finding) | 12 |
| 2. Cost minimization (constrained optimization) | 14 |
| 3. Yield curve (interpolation) | 12 |
| 4. Marginal analysis & surplus (differentiation/integration) | 18 |
| 5. Solow growth model (ODE) | 18 |
| 6. Business-cycle detection (FFT) | 16 |
| 7. Bonus: Bass diffusion model | +10 |
| Code quality, plots labeled, written interpretation | 10 |

## 6. Necessary Python Packages

```bash
pip install numpy scipy matplotlib pandas numdifftools --break-system-packages
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import root, minimize
from scipy.interpolate import interp1d, CubicSpline
from scipy.integrate import quad, solve_ivp
from scipy.fft import fft, fftfreq
import numdifftools as nd
```

## 7. Key Theory Cheat Sheet (quick reference)

- **Equilibrium condition:** `Qd(p) = Qs(p)` → root of `f(p) = Qd(p) - Qs(p)`.
- **Cobb–Douglas production:** `Q = K^α L^β`; minimize cost `C = rK + wL` s.t. `Q = Q0`.
- **Interpolation:** linear (`interp1d`) connects points with straight lines;
  cubic splines (`CubicSpline`) give a smooth curve with continuous 2nd derivative
  — standard for yield curves.
- **Marginal cost/revenue:** `MC(Q) = dTC/dQ`, `MR(Q) = dTR/dQ`; profit-maximizing
  output solves `MC(Q) = MR(Q)`.
- **Price elasticity of demand:** `ε = (dQ/dP) · (P/Q)`.
- **Consumer surplus:** `CS = ∫₀^Q* [P_d(q) − P*] dq`.
- **Producer surplus:** `PS = ∫₀^Q* [P* − P_s(q)] dq`.
- **Solow–Swan model:** `dk/dt = s·k^α − (n+δ)·k`; steady state
  `k* = (s / (n+δ))^(1/(1-α))`.
- **Fourier transform:** `X(f) = FFT(x(t))`; dominant non-zero frequency peak
  in `|X(f)|` reveals the periodicity (business-cycle length) hidden in noisy
  data.
- **Bass diffusion model:** `dF/dt = (p + qF)(1−F)`, `F` = fraction of the
  population who has adopted; `p` = innovation coefficient, `q` = imitation
  coefficient.
