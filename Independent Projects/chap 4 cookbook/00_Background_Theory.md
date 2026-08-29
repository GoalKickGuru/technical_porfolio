# Background & Theory: Modeling Stock Market Data

This document explains the concepts behind the lab, why each step exists, the
statistical/financial theory involved, and — importantly — what this kind of
project can and cannot actually tell you about investing.

---

## 1. What problem is this lab solving?

The lab walks through a **relative valuation + screening pipeline**, a very
common first pass that equity analysts and retail investors use to shrink a
universe of thousands of stocks down to a short list worth a closer look. The
pipeline has five stages:

1. **Acquire** a cross-sectional snapshot of many stocks (a "screener" export)
   plus a time series of historical prices for a shortlist.
2. **Summarize** the data to understand its shape, types, and quality.
3. **Clean** it (numbers stored as text, missing values, extreme outliers).
4. **Value** each stock relative to its peers (sector/industry averages).
5. **Screen** using hard cutoffs, then **visualize** price history and
   volatility for the survivors.

None of these steps require anything exotic — this is a textbook example of
the general data-science pipeline (acquire → clean → explore → model →
communicate) applied to financial fundamentals data.

---

## 2. Core financial concepts

### 2.1 Valuation multiples ("relative valuation")
A valuation multiple relates a stock's price to some fundamental (earnings,
sales, book value, cash). The idea of *relative* valuation is: a stock isn't
"expensive" or "cheap" in isolation — it's expensive or cheap **relative to
similar companies**. A $200 stock can be cheap and a $2 stock can be expensive;
what matters is the multiple relative to peers with similar growth,
profitability, and risk.

| Multiple | Formula | Interpretation |
|---|---|---|
| P/E | Price ÷ Earnings per share | How much you pay per $1 of current earnings |
| Forward P/E | Price ÷ *expected* next-year EPS | Same, but forward-looking |
| PEG | P/E ÷ expected EPS growth rate | Normalizes P/E for growth — a high P/E can be fine if growth is high |
| P/S | Price ÷ Sales per share | Useful when earnings are negative/volatile |
| P/B | Price ÷ Book value per share | Price relative to net assets |
| P/Cash | Price ÷ Cash per share | Price relative to liquid cushion |

**Why compare to sector *and* industry averages?** Sector (e.g. "Financial")
is coarse; industry (e.g. "Property & Casualty Insurance") is fine-grained.
Comparing at both levels catches cases where a stock looks cheap vs. its broad
sector but expensive vs. its narrow, more comparable peer group (or vice
versa).

### 2.2 Risk and financial-health screens
- **Beta** measures a stock's historical volatility relative to the overall
  market (from a linear regression of stock returns on market returns). Beta
  = 1 → moves with the market; > 1 → more volatile; < 1 → less volatile.
- **Total Debt/Equity** measures leverage — how much of the company's capital
  structure is debt vs. equity. Higher leverage amplifies both gains and
  losses and increases bankruptcy risk in downturns.
- **Institutional ownership** is the percent of shares held by institutions
  (funds, pensions, etc.). Very low institutional ownership can mean a stock
  is under-followed (possible inefficiency) or that "smart money" doesn't
  find it interesting — it cuts both ways and isn't a reliable signal by
  itself.
- **EPS growth (trailing, next year, next 5 years)** is used both as a
  quality filter (a company screened as "cheap" but with *shrinking*
  earnings is a value trap, not a bargain) and inside the PEG ratio.

### 2.3 The relative-valuation index
The lab builds a simple composite score: for each of several multiples, flag
1 if the stock's multiple is below its sector average, and 1 again if it's
below its industry average, then sum all flags. This is a **breadth score**,
not a valuation *estimate* — it just counts "how many peer comparisons look
cheap." It deliberately treats all metrics as equally important (naive equal
weighting), which is the main modeling simplification worth remembering.

### 2.4 Moving averages
A **simple moving average (SMA)** smooths a noisy daily price series by
averaging the last *N* closing prices. Two common windows:
- **50-day MA**: a medium-term trend line.
- **200-day MA**: a long-term trend line.

They're used descriptively (to see trend direction through the noise) and as
inputs to simple technical heuristics — e.g., a "golden cross" (50-day moving
above the 200-day) or "death cross" (50-day moving below) are popular but
*not* proven predictors; they're pattern-recognition heuristics, not causal
models.

---

## 3. Statistical / data-science concepts used

- **Cross-sectional vs. time-series data**: the screener snapshot is
  cross-sectional (many stocks, one point in time); the price history is
  time-series (one stock, many points in time). The pipeline deliberately
  handles them with different tools (`groupby` aggregation vs. `rolling`
  windows).
- **Data cleaning of "dirty" numeric strings**: real financial exports store
  numbers as text with `%`, `$`, `,`, `-` (for missing), and suffixes like
  `B`/`M` for billions/millions. Converting these to proper numeric dtypes is
  a prerequisite for *any* aggregation or filtering — this is the single
  most common real-world data-cleaning task in finance data.
- **Outlier sensitivity of the mean**: a single extreme value (in the book,
  Berkshire Hathaway's ~$172,000/share price; the same trick is reproduced
  synthetically here) can distort a sector average enough to make an entire
  sector look mispriced. This motivates either removing known extreme
  outliers or using a robust statistic (median) alongside the mean.
- **Long ("melted") vs. wide data**: computing group averages across many
  metrics at once is often easiest by reshaping wide data (`Sector, P/E, P/S,
  ...`) into long/tidy form (`Sector, metric, value`), aggregating, then
  pivoting back to wide. This is the pandas equivalent of R's
  `melt`/`dcast` (`pandas.melt` / `pivot_table`).
- **Multi-criteria filtering ("screening")**: combining several boolean
  conditions (`Price.between(...)`, `Beta < 1.5`, etc.) with `&` is a simple
  but very general pattern for narrowing any dataset by business rules.

---

## 4. What this lab *can* successfully simulate / teach

- The **full mechanics** of building a stock screener and relative-valuation
  pipeline in pandas: cleaning messy numeric text, grouped aggregation at two
  levels of granularity, reshaping data, building a composite score, and
  multi-condition filtering.
- **Realistic data quality problems**: missing values, mixed units, and a
  single dominating outlier — and the standard techniques to handle them.
- **Time-series basics**: rolling windows/moving averages, and comparing
  multiple securities' price paths and volatility on one chart.
- A **template workflow** that generalizes to *any* "acquire → clean →
  score/rank → filter → visualize" project (e.g. screening real estate
  listings, comparing product SKUs, ranking job candidates by criteria) —
  see the reusable template notebook.

## 5. Limitations — what this lab does *not* do

- **It is not investment advice, and the synthetic data is fake.** Prices,
  fundamentals, and price histories in these notebooks are randomly generated
  (log-normal fundamentals; geometric Brownian motion for prices) purely so
  the pipeline runs deterministically and offline. Do not draw any real
  investment conclusions from the numbers.
- **finviz's free CSV export and Yahoo's legacy `ichart` endpoint (used in
  the original book) are both gone.** finviz's bulk CSV export now requires
  an Elite subscription, and `ichart.finance.yahoo.com` was shut down years
  ago. The included "how to get real data" cells show current alternatives
  (`yfinance` for prices/fundamentals, or a manual finviz CSV download).
- **Equal-weighted flag counting is a toy scoring model.** It ignores
  correlation between metrics (P/E and Forward P/E are highly correlated, so
  they're nearly double-counted), ignores statistical significance, and
  treats "below average" as good regardless of *why* a multiple is low (a
  low P/E can mean "cheap" or it can mean "the market correctly expects
  earnings to collapse").
- **No fundamental/quality analysis.** The pipeline never reads a balance
  sheet, cash-flow statement, or looks at management quality, competitive
  moat, accounting red flags, or macro conditions — all of which matter far
  more than multiples alone.
- **Sector/industry averages are themselves noisy** with few members per
  group and are sensitive to outliers (exactly the point of the outlier-
  removal step) — a small peer group's "average" may not be a meaningful
  benchmark.
- **Moving averages and beta are backward-looking.** They describe what
  *has* happened, not what will. Technical patterns like moving-average
  crossovers have weak, inconsistent predictive power in the academic
  literature.
- **Survivorship / look-ahead issues in real deployments.** A real screener
  run today only contains companies that still exist today — companies that
  went bankrupt or were delisted are invisible, which biases historical
  backtests of "would this strategy have worked."
- **No transaction costs, taxes, slippage, or position sizing** — anything
  resembling a real trading or portfolio strategy needs all of these before
  it means anything economically.

## 6. Suggested extensions (see "Further ideas" in the template notebook)
- Weight metrics by predictive power instead of equally.
- Add quality filters (return on equity, profit margin trend, insider buying).
- Use the median (not just mean) for sector/industry benchmarks.
- Backtest a screening rule historically with point-in-time data (avoiding
  survivorship bias) before trusting it.
- Swap the synthetic price generator for real `yfinance` data when you have
  outbound internet access, and real finviz/quandl/Alpha Vantage fundamentals
  data with a paid or free-tier API key.
