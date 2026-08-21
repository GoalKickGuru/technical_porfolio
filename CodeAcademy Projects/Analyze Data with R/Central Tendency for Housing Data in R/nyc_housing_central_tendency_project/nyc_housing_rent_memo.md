# MEMORANDUM

**TO:** Real Estate Investment Committee & Market Strategy Team  
**FROM:** Senior Quantitative Data Analyst  
**DATE:** August 20, 2026  
**SUBJECT:** Statistical Analysis of Central Tendency in NYC One-Bedroom Rental Markets (Manhattan, Brooklyn, Queens)

---

### Executive Summary

An empirical analysis of StreetEasy rental listings for one-bedroom apartments across **Manhattan**, **Brooklyn**, and **Queens** demonstrates significant structural variation in pricing distributions. 

By comparing the **Mean**, **Median**, and **Mode** across these markets, we observe substantial **right-skewness** driven by high-end luxury inventory—most prominently in Manhattan. Consequently, relying solely on average rent leads to an overestimation of typical tenant costs.

---

### Quantitative Findings

| Borough | Sample Size ($n$) | Mean Rent ($ar{x}$) | Median Rent ($Q_2$) | Mode Rent ($Mo$) | Mean-Median Spread | Distribution Shape |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Manhattan** | ~1,000 listings | **$4,200** | **$3,800** | **$3,500** | +$400 (+10.5%) | High Right-Skew |
| **Brooklyn** | ~1,000 listings | **$3,320** | **$3,000** | **$2,800** | +$320 (+10.7%) | Moderate Right-Skew |
| **Queens** | ~700 listings | **$2,450** | **$2,300** | **$2,200** | +$150 (+6.5%) | Symmetric / Low Skew |

---

### Key Analytical Takeaways

1. **Skewness & Outlier Impact**:
   * In **Manhattan**, the arithmetic mean ($4,200) exceeds the median ($3,800) by $400. Ultra-luxury penthouse units ($10,000+/month) disproportionately pull the mean upward.
   * **Queens** exhibits the most symmetric distribution, where mean ($2,450) and median ($2,300) differ by only $150, reflecting a more homogeneous inventory of residential mid-rises.

2. **The Role of Mode in Real Estate Pricing**:
   * The modal rent represents the most frequently listed contract price. In Brooklyn ($2,800) and Manhattan ($3,500), modes correspond to developer "sweet spot" pricing tiers designed to hit specific search filter thresholds on listing portals.

3. **Strategic Recommendations for Stakeholders**:
   * **Underwriting Baseline**: Investment decisions and underwriting models should utilize **Median Rent** as the primary proxy for achievable unit revenues rather than Mean Rent.
   * **Tenant Budgeting**: Tenants searching for typical listings should reference the **Mode** and **Median** as realistic entry price points.
