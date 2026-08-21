# Cheat Sheet — Statistical Data Analytics for Economics
*(companion to `Economics_DataAnalytics_SKELETON.ipynb`)*

Source material: Steinkamp, *Python for Engineering and Scientific Computing*, Ch. 9 (Statistical Computations).

---

## 0. Imports you'll need

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.integrate import quad
```

| Library | Role |
|---|---|
| `numpy` | random-number generation, arrays, mean/std/histogram |
| `pandas` | tabular data (DataFrame), CSV I/O |
| `matplotlib.pyplot` | histograms, scatter plots, control charts |
| `scipy.stats` | mode, hmean, gmean, skew, linregress |
| `scipy.integrate.quad` | numerical integration under the normal curve |

---

## 1. Generating simulated data

```python
values = np.random.normal(setpoint, std_dev, size=n)   # 1-D array of n normal values
values = np.around(values, decimals=2)                 # round for realism
```
* `setpoint` = the mean (e.g., inflation target)
* `std_dev` = the standard deviation (volatility)
* `size=n` = number of observations

**Correlated series (e.g., Phillips curve):** build `Y` as a linear function of `X` plus noise:
```python
Y = intercept + slope * X + np.random.normal(0, noise_std, size=n)
```

---

## 2. Saving / loading data

```python
df.to_csv("file.csv", index=False)       # save
df2 = pd.read_csv("file.csv")            # reload
np.savetxt("file.txt", values, fmt="%4.2f")   # NumPy-only alternative
values = np.loadtxt("file.txt")
```

---

## 3. Frequency distribution & histograms

Number of classes (Sturges-like rule of thumb used in the book):
$$k = \lceil \sqrt{n} \rceil \qquad R = x_{max}-x_{min} \qquad w = R/k$$

```python
k = int(np.sqrt(n) + 0.5)
minimum, maximum = np.amin(values), np.amax(values)
R = round(maximum - minimum, 2)
w = round(R / k, 2)

H, I = np.histogram(values, bins=k)     # H = counts, I = bin edges
h = 100 * H / n                         # relative frequency (%)

fig, ax = plt.subplots()
ax.hist(values, bins=k, edgecolor="navy", color="skyblue")
ax.set(xlabel="...", ylabel="Absolute frequency")
plt.show()
```

---

## 4. Location parameters

| Statistic | Formula | Code |
|---|---|---|
| Arithmetic mean | $\bar{x}=\frac{1}{n}\sum x_i$ | `np.mean(values)` |
| Median | middle value of sorted data | `np.median(values)` |
| Mode | most frequent value | `stats.mode(values, keepdims=True)` |
| Harmonic mean | $n\left(\sum 1/x_i\right)^{-1}$ | `stats.hmean(values)` |
| Geometric mean | $\sqrt[n]{x_1 x_2 \cdots x_n}$ | `stats.gmean(values)` |

> Harmonic and geometric means require **strictly positive** values — safe for rates like inflation
> or unemployment, **not safe** for values that can be negative (e.g., returns).

---

## 5. Dispersion parameters & capability indices

$$s=\sqrt{\frac{1}{n-1}\sum_{i=1}^n (x_i-\bar x)^2} \qquad \text{(sample std. dev., } n-1 \text{ denominator)}$$

```python
s = np.std(values, ddof=1)     # ddof=1 => divide by (n-1), NOT n
R = np.amax(values) - np.amin(values)   # span
```

**Capability index** (analogous to machine capability $C_m$, $C_{mk}$ in the book):
$$C_m=\frac{T}{6s}\ \ge 1.67 \qquad C_{mk}=\frac{\Delta_{krit}}{3s}\ \ge 1.67$$

* $T$ = width of the target/tolerance band (upper limit − lower limit)
* $\Delta_{krit}$ = distance from the mean to the **nearer** limit = `min(upper-mean, mean-lower)`
* $C_m \ge 1.67$ → the process (or policy) can, in principle, stay inside the band
* $C_{mk} \ge 1.67$ → the process is also **centered**, not drifting toward one edge

```python
T = upper_limit - lower_limit
Cm = T / (6 * s)
delta_o, delta_u = upper_limit - mean, mean - lower_limit
delta_k = min(delta_o, delta_u)
Cmk = delta_k / (3 * s)
```

---

## 6. Normal distribution & probability

Density function:
$$g(x)=\frac{1}{\sigma\sqrt{2\pi}}\,e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

```python
def g(x, sigma, mu):
    return np.exp(-0.5 * (x - mu) ** 2 / sigma ** 2) / (sigma * np.sqrt(2 * np.pi))

x = np.arange(mu - 4*sigma, mu + 4*sigma, 0.01)
y = g(x, sigma, mu)
```

**Probability that a value falls in `[a, b]`** = area under the curve = numerical integral:

```python
prob, error = quad(g, a, b, args=(sigma, mu))
```

Classic reference values (memorize!):

| Range | Probability |
|---|---|
| $\mu \pm 1\sigma$ | 68.27 % |
| $\mu \pm 2\sigma$ | 95.45 % |
| $\mu \pm 3\sigma$ | 99.73 % |

*(Equivalent shortcut using `scipy.stats.norm`: `stats.norm.cdf(b, mu, sigma) - stats.norm.cdf(a, mu, sigma)`.)*

---

## 7. Skewness

$$S_{\text{Pearson}}=\frac{\bar x-\tilde x}{s}$$

```python
S1 = (mean - median) / std        # quick approximation
S2 = stats.skew(values)           # exact
```

| Sign of skew | Shape | Economics interpretation (returns) |
|---|---|---|
| $S<0$ | right-skewed (long left tail) | rare but large **losses** — crash risk |
| $S=0$ | symmetric (≈ normal) | no tail asymmetry |
| $S>0$ | left-skewed (long right tail) | rare but large **gains** |

---

## 8. Regression analysis

$$y = mx + a \qquad r = \frac{s_{xy}}{s_x s_y}$$

```python
m, a, r, p, e = stats.linregress(X, Y)
# m = slope, a = intercept, r = correlation coefficient
# p = p-value for slope != 0, e = standard error of slope
```

```python
fig, ax = plt.subplots()
ax.plot(X, Y, "rx")            # scatter of observed data
ax.plot(X, m * X + a, "b-")    # fitted regression line
```

| \|r\| range | Strength |
|---|---|
| 0.0 – 0.3 | weak |
| 0.3 – 0.7 | moderate |
| 0.7 – 1.0 | strong |

Sign of `r` tells you the **direction** (e.g., negative r between inflation and unemployment supports
a Phillips-curve trade-off).

---

## 9. Control charts (two-lane: mean + dispersion)

For samples of size **n = 5** (5 regions, 5 workpieces, etc.), standard SPC factors:
$$A_3 = 1.152 \qquad B_4 = 1.669$$

```python
UCLm = grand_mean + A3 * mean_of_sample_stds
LCLm = grand_mean - A3 * mean_of_sample_stds
UCLs = B4 * mean_of_sample_stds
```

**Reshaping a flat series into a sample table** (rows = items per sample, columns = number of samples):
```python
table = np.reshape(values, (rows, columns), order="F")   # Fortran order fills columns first
```

**Per-column (per-sample) statistics via slicing** — no inner loop needed:
```python
for i in range(columns):
    sample_mean = np.mean(table[0:rows, i])
    sample_std  = np.std(table[0:rows, i], ddof=1)
```

**Two-panel plot:**
```python
fig, ax = plt.subplots(2, 1)
ax[0].plot(x, [UCLm, UCLm], "r-")
ax[0].plot(x, [grand_mean, grand_mean], "g-")
ax[0].plot(x, [LCLm, LCLm], "r-")
ax[0].plot(sample_index, sample_means, "bx-")

ax[1].plot(x, [UCLs, UCLs], "r-")
ax[1].plot(sample_index, sample_stds, "gx-")
fig.tight_layout()
plt.show()
```

**Out-of-control signal:** a value above `UCL` or below `LCL` is a red flag — investigate the cause
(in economics: a policy shock, a data error, a genuine regime change).

---

## Quick reference: which NumPy/SciPy function do I need?

| Task | Function |
|---|---|
| Random normal sample | `np.random.normal(mean, std, size=n)` |
| Round | `np.around(arr, decimals=2)` |
| Min / Max | `np.amin(arr)` / `np.amax(arr)` |
| Sort | `np.sort(arr)` |
| Reshape flat → table | `np.reshape(arr, (rows, cols), order='F')` |
| Save / load text | `np.savetxt()` / `np.loadtxt()` |
| Mean | `np.mean(arr)` |
| Median | `np.median(arr)` |
| Mode | `scipy.stats.mode(arr, keepdims=True)` |
| Harmonic mean | `scipy.stats.hmean(arr)` |
| Geometric mean | `scipy.stats.gmean(arr)` |
| Std dev (sample) | `np.std(arr, ddof=1)` |
| Histogram / frequency table | `np.histogram(arr, bins=k)` |
| Skewness | `scipy.stats.skew(arr)` |
| Linear regression | `scipy.stats.linregress(X, Y)` |
| Numerical integration | `scipy.integrate.quad(f, a, b, args=(...))` |
| Correlation matrix | `np.corrcoef(X, Y)` |
| Covariance matrix | `np.cov(X, Y)` |

---

## Common pitfalls

* **`ddof=1`** — always use it for a *sample* standard deviation (`n-1` in the denominator); NumPy's
  default (`ddof=0`) divides by `n`, which understates variability for small samples.
- **Harmonic/geometric mean fail on non-positive data** — don't apply them to return series that can
  be negative or zero.
- **`order='F'`** in `np.reshape()` fills columns first — needed to replicate the book's sample-table
  layout (each column = one time period's sample).
- **`quad()` returns a tuple** `(value, error_estimate)` — only take element `[0]` (or unpack both).
- Always **label your axes** in economics plots (units matter — % vs. index points vs. currency).
