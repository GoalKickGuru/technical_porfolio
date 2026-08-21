# R Central Tendency Technical Cheat Sheet

## Quick Reference Summary

| Metric | Base R Function | Package Alternative | Outlier Sensitivity | Key Usage Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Mean** | `mean(x, na.rm = TRUE)` | `dplyr::summarise(mean = mean(x))` | **High** | Sensitive to extreme upper tail |
| **Median** | `median(x, na.rm = TRUE)` | `dplyr::summarise(med = median(x))` | **None (Robust)** | Represents 50th percentile |
| **Mode** | *Custom function* | `DescTools::Mode(x, na.rm = TRUE)` | **Low** | Base R `mode()` returns data type! |

---

## Detailed Code Patterns

### 1. Arithmetic Mean
```R
# Basic usage
mean_val <- mean(prices, na.rm = TRUE)

# Trimmed mean (removes top/bottom 5% outliers)
trimmed_mean <- mean(prices, trim = 0.05, na.rm = TRUE)
```

### 2. Median
```R
# Basic usage
med_val <- median(prices, na.rm = TRUE)
```

### 3. Statistical Mode

#### Option A: Using `DescTools` (Recommended)
```R
library(DescTools)

# DescTools::Mode returns a vector of modes with frequency attributes
raw_mode <- Mode(prices, na.rm = TRUE)

# Extract primary mode value
primary_mode <- raw_mode[1]
```

#### Option B: Base R Custom Function Fallback
```R
get_mode <- function(v) {
  v <- v[!is.na(v)]
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

primary_mode <- get_mode(prices)
```

---

## Critical Traps to Avoid

1. **Base R `mode()` Misconception**:
   ```R
   x <- c(1000, 2000, 2000, 3000)
   mode(x) 
   # [1] "numeric"  <-- THIS IS NOT THE STATISTICAL MODE!
   ```

2. **Missing Value Propagation**:
   ```R
   x <- c(2500, 3000, NA)
   mean(x) # Output: NA
   mean(x, na.rm = TRUE) # Output: 2750
   ```

3. **Multi-Modal Data Vectors**:
   If a dataset has two modes (e.g., $2,500 and $3,200), `DescTools::Mode()` returns `c(2500, 3200)`. Always select `Mode(x)[1]` if assigning to a single scalar variable.
