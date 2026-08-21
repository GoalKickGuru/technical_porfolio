# Coding Strategy Guide: Analyzing Central Tendency in R

## Executive Overview
When working on exploratory data analysis (EDA) for continuous variables like real estate prices, understanding **central tendency** (mean, median, mode) provides the baseline for data-driven decisions. This strategy guide outlines a repeatable workflow for building robust R scripts.

---

## 5-Phase Analysis Strategy

### Phase 1: Problem Definition & Data Hygiene
1. **Identify Variable Types**: Real estate prices are continuous, strictly positive numeric variables.
2. **Anticipate Distribution Shape**: Housing markets almost always exhibit **right-skewness** due to high-end luxury listings pulling up averages.
3. **Handle Missing Values**: Always audit data for `NA` values. Unhandled `NA` values cause functions like `mean()` and `median()` to evaluate to `NA`.

### Phase 2: Choosing the Appropriate Central Tendency Metric
* **Arithmetic Mean (\(\bar{x}\))**: Best for total revenue calculation; highly sensitive to extreme luxury outliers.
* **Median (\(Q_2\))**: The most robust metric for standard tenant rent expectations. It divides the distribution into two equal halves.
* **Mode (\(Mo\))**: Represents the most common price point (e.g., standard floor plan or popular price tier like $2,500/month).

### Phase 3: Navigating R Language Peculiarities
* **The `mode()` Pitfall in Base R**: Calling `mode(x)` in Base R returns the *storage mode* (`"numeric"`, `"character"`), NOT the statistical mode.
* **Package Selection**: Use `DescTools::Mode()` or write a custom frequency function to calculate true modal values.
* **Multi-Modal Data Handling**: Real estate prices can be bimodal (e.g., walk-up vs. doorman luxury). Be prepared to handle vector outputs from `Mode()`.

### Phase 4: Structured Scripting Best Practices
1. **Modular Variable Naming**: Standardize variable naming (`<borough>_<metric>`), making script debugging straightforward.
2. **Defensive Logic**: Use existence and `null` checks before printing or transforming values.
3. **Tabular Summaries**: Wrap individual scalar variables into `tibble` or `data.frame` objects for clean reporting.

### Phase 5: Verification & Quality Assurance
* Verify that \(\text{Mean} > \text{Median}\) for right-skewed data.
* Validate that `Mode` corresponds to actual observation values within the dataset.
* Ensure code executes cleanly without unresolved warnings or runtime syntax errors.
