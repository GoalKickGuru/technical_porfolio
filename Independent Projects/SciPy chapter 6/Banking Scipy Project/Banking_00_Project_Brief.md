# Computational Banking with SciPy
### A Data Analytics Project Applying Numerical Methods to Banking & Treasury Problems

**Based on:** Steinkamp, V. — *Python for Engineering and Scientific Computing* (2024), Chapter 6, "Numerical Computations and Simulations Using SciPy"

---

## 1. Motivation

Chapter 6 of the source text builds a numerical toolkit — root-finding,
constrained optimization, interpolation, differentiation, integration,
differential equations, and Fourier analysis — around engineering and physics
examples. **Banks use the exact same numerical machinery every day**, in bond
pricing, treasury (ALM), credit risk, and market risk functions. This project
keeps the SciPy tools from the book and re-targets each example at a banking
question.

| SciPy technique (book §) | Banking application (this project) |
|---|---|
| `scipy.optimize.root` (§6.1) | Solving for a bond's Yield to Maturity (YTM) |
| `scipy.optimize.minimize` (§6.2) | Minimum-variance investment portfolio (Markowitz) |
| `scipy.interpolate` (§6.3) | Building a SOFR discount curve to value a loan book |
| `numdifftools` (§6.4) | Bond duration & convexity (interest-rate risk) |
| `scipy.integrate.quad` (§6.5) | Expected Credit Loss (ECL) under CECL/IFRS 9 |
| `scipy.integrate.solve_ivp` (§6.6) | Vasicek mean-reverting short-rate model (ALM) |
| `scipy.fft` (§6.7 / §6.9) | Detecting cyclicality in loan charge-off rates |
| Epidemic-type ODE system (§6.11) | Interbank contagion / systemic-risk model |

## 2. Learning Objectives

By the end of this project you will be able to:

1. Solve for a bond's yield to maturity using numerical root-finding.
2. Build a minimum-variance portfolio subject to a return target and budget
   constraint using constrained optimization.
3. Construct a smooth discount curve from money-market and swap quotes and use
   it to present-value a loan portfolio.
4. Compute duration and convexity numerically and use them to approximate
   bond price changes for interest-rate risk management.
5. Compute Expected Credit Loss by integrating a hazard (default-probability)
   function over the life of an exposure — the core CECL/IFRS 9 calculation.
6. Solve the Vasicek ODE for the mean-reverting short rate used in
   asset-liability management (ALM) and interest-rate stress testing.
7. Use the FFT to detect cyclical and seasonal structure in charge-off /
   default-rate time series.
8. (Bonus) Simulate interbank contagion with a compartmental (SIR-style) model
   of systemic risk.

## 3. Deliverables

1. **`banking_scipy_project_SKELETON.ipynb`** — starting point, with theory,
   synthetic data already generated, and `# TODO` code cells to complete.
2. A completed notebook with all TODOs filled in and short-answer questions
   answered.
3. (Optional/instructor use) **`banking_scipy_project_SOLUTION.ipynb`** — full
   worked solution / cheat sheet with executed outputs and plots.

## 4. Data

All data is **synthetically generated with a fixed random seed** inside the
notebook — no external files, market-data subscriptions, or internet access
required, and your results will match the solution notebook exactly.

## 5. Grading Rubric (suggested, 100 pts)

| Section | Points |
|---|---|
| 1. Yield to maturity (root-finding) | 12 |
| 2. Minimum-variance portfolio (constrained optimization) | 14 |
| 3. SOFR discount curve (interpolation) | 12 |
| 4. Duration, convexity & Expected Credit Loss (differentiation/integration) | 20 |
| 5. Vasicek short-rate model (ODE) | 16 |
| 6. Charge-off rate cyclicality (FFT) | 16 |
| 7. Bonus: interbank contagion model | +10 |
| Code quality, labeled plots, written interpretation | 10 |

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

- **Bond pricing:** `P(y) = Σ CFᵢ / (1+y/m)^i` where `y` = yield to maturity,
  `m` = compounding frequency. YTM solves `P(y) = P_market` — a root-finding
  problem since it can't generally be inverted algebraically.
- **Markowitz minimum-variance portfolio:** minimize `wᵀΣw` subject to
  `Σwᵢ = 1`, `wᵀμ ≥ target return`, `wᵢ ≥ 0`.
- **Discount factor:** `DF(t) = exp(-r(t)·t)` where `r(t)` is the
  zero/spot rate at maturity `t`, typically built by interpolating a small
  set of market quotes (deposits, swaps).
- **Modified duration:** `D = -(1/P)·(dP/dy)` — the % price change per unit
  change in yield. **Convexity:** `C = (1/P)·(d²P/dy²)`. Price approximation:
  `ΔP/P ≈ -D·Δy + 0.5·C·Δy²`.
- **Expected Credit Loss (ECL):**
  `ECL = ∫₀^T EAD(t) · LGD · h(t) · S(t) dt`, where `h(t)` is the default
  hazard rate, `S(t) = exp(-∫₀^t h(u)du)` is the survival probability,
  `EAD` = exposure at default, `LGD` = loss given default.
- **Vasicek short-rate model (deterministic drift):**
  `dr/dt = a·(b - r)`, mean-reverts to long-run level `b` at speed `a`.
  Solution: `r(t) = b + (r₀-b)e^{-at}`.
- **Fourier transform:** peaks in `|FFT(x)|` reveal hidden periodicities
  (business cycle length, seasonal effects) in a noisy loss/default series.
- **SIR-style contagion model:** `dS/dt=-βSI`, `dI/dt=βSI-γI`, `dR/dt=γI`,
  where `S` = healthy banks, `I` = distressed/illiquid banks, `R` =
  resolved/recapitalized banks; `β` = interconnectedness, `γ` = speed of
  resolution (bailout/liquidity support).
