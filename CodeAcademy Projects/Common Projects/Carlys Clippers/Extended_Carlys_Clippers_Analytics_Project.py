# ===== DATA SETUP =====
hairstyles = ["bouffant", "pixie", "dreadlocks", "crew", "bowl", "bob", "mohawk", "flattop"]
prices = [30, 25, 40, 20, 20, 35, 50, 35]
last_week = [2, 3, 5, 8, 4, 4, 6, 2]

print("=" * 50)
print("CARLY'S CLIPPERS - EXTENDED ANALYTICS REPORT")
print("=" * 50)

# ===== PART 1: BASIC PRICE ANALYSIS =====
print("\n--- PART 1: Price Analysis ---")

# Task 1: Calculate average price
total_price = sum(prices)
average_price = total_price / len(prices)
print(f"Average Haircut Price: ${average_price:.2f}")

# Task 2: Apply discount (reduce all prices by $5)
new_prices = [price - 5 for price in prices]
print(f"\nOriginal Prices: {prices}")
print(f"New Prices (after $5 discount): {new_prices}")

# Task 3: Find most expensive and cheapest hairstyles
max_price_index = prices.index(max(prices))
min_price_index = prices.index(min(prices))
print(f"\nMost Expensive: {hairstyles[max_price_index]} at ${prices[max_price_index]}")
print(f"Cheapest: {hairstyles[min_price_index]} at ${prices[min_price_index]}")

# ===== PART 2: REVENUE ANALYSIS =====
print("\n--- PART 2: Revenue Analysis ---")

# Task 4: Calculate total weekly revenue
total_revenue = 0
for i in range(len(hairstyles)):
    revenue_for_style = prices[i] * last_week[i]
    total_revenue += revenue_for_style
    
print(f"Total Revenue: ${total_revenue}")

# Task 5: Calculate average daily revenue
average_daily_revenue = total_revenue / 7
print(f"Average Daily Revenue: ${average_daily_revenue:.2f}")

# Task 6: Project monthly revenue (assuming 4 weeks)
projected_monthly_revenue = total_revenue * 4
print(f"Projected Monthly Revenue: ${projected_monthly_revenue}")

# ===== PART 3: PRODUCT PERFORMANCE =====
print("\n--- PART 3: Product Performance ---")

# Task 7: Create a list of (hairstyle, revenue) tuples
revenue_by_hairstyle = [(hairstyles[i], prices[i] * last_week[i]) for i in range(len(hairstyles))]
print("\nRevenue by Hairstyle:")
for style, revenue in revenue_by_hairstyle:
    print(f"  {style}: ${revenue} ({last_week[hairstyles.index(style)]} customers)")

# Task 8: Find best-selling hairstyle (by customer count)
most_popular_index = last_week.index(max(last_week))
print(f"\nBest-Selling Hairstyle: {hairstyles[most_popular_index]} ({max(last_week)} customers)")

# Task 9: Find highest-revenue generating hairstyle
revenue_values = [prices[i] * last_week[i] for i in range(len(hairstyles))]
highest_revenue_index = revenue_values.index(max(revenue_values))
print(f"Highest Revenue Generator: {hairstyles[highest_revenue_index]} (${max(revenue_values)})")

# Task 10: Calculate customer distribution
total_customers = sum(last_week)
print(f"\nTotal Customers This Week: {total_customers}")
for i in range(len(hairstyles)):
    percentage = (last_week[i] / total_customers) * 100
    print(f"  {hairstyles[i]:12} | {last_week[i]:2} customers ({percentage:.1f}%)")

# ===== PART 4: PROMOTIONAL STRATEGY =====
print("\n--- PART 4: Promotional Strategy ---")

# Task 11: Find cuts under $30 (using new discounted prices)
cuts_under_30 = [hairstyles[i] for i in range(len(new_prices)) if new_prices[i] < 30]
print(f"\nHaircuts Under $30 (after discount): {cuts_under_30}")

# Task 12: Create premium package recommendations
premium_cuts = [hairstyles[i] for i in range(len(prices)) if prices[i] >= 35]
print(f"\nPremium Haircuts ($35+): {premium_cuts}")

# Task 13: Value deals (high popularity + low price)
value_deals = [hairstyles[i] for i in range(len(prices)) 
               if prices[i] <= 25 and last_week[i] >= 5]
print(f"\nValue Deals (≤$25 & ≥5 customers): {value_deals}")

# ===== PART 5: ADVANCED INSIGHTS =====
print("\n--- PART 5: Advanced Insights ---")

# Task 14: Price elasticity indicators (popularity relative to price tier)
print("\nPrice Tier Performance:")
budget_tier = [i for i, p in enumerate(prices) if p < 25]
mid_tier = [i for i, p in enumerate(prices) if 25 <= p < 35]
premium_tier = [i for i, p in enumerate(prices) if p >= 35]

def tier_stats(indices):
    if not indices:
        return None, None
    avg_customers = sum(last_week[i] for i in indices) / len(indices)
    avg_price = sum(prices[i] for i in indices) / len(indices)
    return round(avg_customers, 2), round(avg_price, 2)

budget_stats = tier_stats(budget_tier)
mid_stats = tier_stats(mid_tier)
premium_stats = tier_stats(premium_tier)

print(f"  Budget (<$25):   Avg Customers: {budget_stats[0]}, Avg Price: ${budget_stats[1]}")
print(f"  Mid-Tier ($25-35): Avg Customers: {mid_stats[0]}, Avg Price: ${mid_stats[1]}")
print(f"  Premium (≥$35):  Avg Customers: {premium_stats[0]}, Avg Price: ${premium_stats[1]}")

# Task 15: Recommendation engine - what to promote
print("\n--- CARLY'S ACTIONABLE RECOMMENDATIONS ---")
print("""
1. POPULARITY BOOST: Push '{0}' - most popular with {1} customers
2. REVENUE FOCUS: Highlight '{1}' - highest revenue generator at ${2}
3. DISCOUNT WINNERS: The following styles remain affordable post-discount: {3}
4. VALUE OPPORTUNITY: Consider promoting '{4}' - good volume at reasonable price
5. PREMIUM UPSOLD: Target customers with '{5}', '{6}' - premium options with strong demand

ESTIMATED IMPACT:
- If budget-tier sales increase 20%: +${0} extra monthly revenue
- If premium-tier gets 10% more customers: +${1} extra monthly revenue
""".format(
    hairstyles[most_popular_index], max(last_week),
    hairstyles[highest_revenue_index], max(revenue_values),
    cuts_under_30,
    value_deals[0] if value_deals else "none",
    premium_cuts[0] if premium_cuts else "none",
    premium_cuts[1] if len(premium_cuts) > 1 else "none",
    # Revenue estimates
    int(total_revenue * 0.2 * 4),
    int((prices[highest_revenue_index] * last_week[highest_revenue_index]) * 0.1 * 4)
))

# ===== SUMMARY CARD =====
print("\n" + "=" * 50)
print("EXECUTIVE SUMMARY")
print("=" * 50)

summary_data = {
    "total_customers": total_customers,
    "weekly_revenue": total_revenue,
    "avg_ticket": total_revenue / total_customers,
    "best_performer": f"{hairstyles[most_popular_index]} ({max(last_week)} customers)",
    "revenue_champion": f"{hairstyles[highest_revenue_index]} (${max(revenue_values)})",
    "discount_styles": len(cuts_under_30)
}

print(f"""
Total Customers This Week:     ${summary_data['total_customers']}
Weekly Revenue:                ${summary_data['weekly_revenue']:.2f}
Average Ticket per Customer:   ${summary_data['avg_ticket']:.2f}
Best-Selling Style:            {summary_data['best_performer']}
Top Revenue Generator:         {summary_data['revenue_champion']}
Affordable Options (Post-Disc.): {summary_data['discount_styles']} styles
""")

print("=" * 50)
print("END OF REPORT")
print("=" * 50)