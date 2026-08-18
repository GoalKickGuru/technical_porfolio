"""Mock noshmishmosh library for the Nosh Mish Mosh A/B sample size exercise.
Data is crafted so calculations produce the classic Codecademy numbers:
baseline ≈ 18.6 %, average payment ≈ 26.38, new customers needed = 47,
percentage point increase ≈ 9.4, MDE ≈ 50.54 %, and the calculator yields ~490.
"""
import numpy as np

np.random.seed(42)

# 500 visitors in a typical week
customer_visits = list(range(500))

# 93 of them purchase → baseline 18.6 %
purchasing_customers = list(range(93))

# Money spent by those 93 customers. Mean ≈ 26.383 so that
# np.ceil(1240 / mean) == 47
_raw = np.random.normal(loc=26.383, scale=8.0, size=93)
_raw = np.clip(_raw, 5.0, 80.0)  # realistic bounds
# Force exact mean for reproducibility of the classic numbers
money_spent = (_raw - _raw.mean() + 26.382978723404257).tolist()

# Convenience aliases used by the original project
# (some solutions also import these)
all_visitors = customer_visits
paying_visitors = purchasing_customers
payment_history = money_spent

if __name__ == "__main__":
    print("customer_visits length:", len(customer_visits))
    print("purchasing_customers length:", len(purchasing_customers))
    print("baseline %:", len(purchasing_customers)/len(customer_visits)*100)
    print("mean payment:", np.mean(money_spent))
    print("new customers needed:", np.ceil(1240 / np.mean(money_spent)))
