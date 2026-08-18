"""Macroeconomic Policy Cycles Simulation.

Simulates the Inflation-Unemployment-Debt Trilemma across 100, 1,000, or 10,000 cycles.
"""

import random
import statistics

# Policy States
EXPANSIONARY = "EXPANSIONARY"
HAWKISH = "HAWKISH"
AUSTERITY = "AUSTERITY"


class Economy:

    def __init__(self):
        self.gdp_growth = 2.5  # Annual GDP Growth (%)
        self.inflation = 2.0  # Inflation Rate (%)
        self.unemployment = 4.5  # Unemployment Rate (%)
        self.debt_to_gdp = 60.0  # Debt-to-GDP Ratio (%)
        self.current_policy = EXPANSIONARY

    def step(self) -> str:
        """Simulates one economic cycle/quarter and transitions policy based on economic indicators."""
        # 1. Update economic variables based on current policy regime
        if self.current_policy == EXPANSIONARY:
            self.gdp_growth += random.uniform(0.2, 0.8)
            self.unemployment = max(2.0,
                                    self.unemployment - random.uniform(0.1, 0.4))
            self.inflation += random.uniform(0.3, 0.9)  # Inflation builds up
            self.debt_to_gdp += random.uniform(0.5, 1.5)

        elif self.current_policy == HAWKISH:
            self.inflation = max(1.0, self.inflation - random.uniform(0.4, 0.8))
            self.gdp_growth -= random.uniform(0.2, 0.6)
            self.unemployment += random.uniform(0.2, 0.5)  # Unemployment rises
            self.debt_to_gdp += random.uniform(
                0.2, 0.8)  # Higher interest burden

        elif self.current_policy == AUSTERITY:
            self.debt_to_gdp = max(
                40.0, self.debt_to_gdp - random.uniform(0.8, 1.8))
            self.gdp_growth -= random.uniform(0.1, 0.4)
            self.unemployment += random.uniform(0.1, 0.3)
            self.inflation -= random.uniform(0.1, 0.3)

        # Apply stochastic market noise
        self.gdp_growth += random.uniform(-0.2, 0.2)
        self.inflation += random.uniform(-0.1, 0.1)

        # 2. Rule-based Policy Transition Logic (The Game Loop)
        if self.inflation > 5.5:
            # Shift to Hawkish if inflation breaks upper target
            self.current_policy = HAWKISH
        elif self.unemployment > 7.0 or self.gdp_growth < 0.5:
            # Shift to Expansionary if recession or high unemployment strikes
            self.current_policy = EXPANSIONARY
        elif self.debt_to_gdp > 90.0:
            # Shift to Austerity if debt burden becomes critical
            self.current_policy = AUSTERITY

        return self.current_policy


def run_macro_simulation(num_cycles: int = 1000):
    """Runs the macro simulation for a specified number of cycles (e.g., 100, 1000, 10000)."""
    economy = Economy()

    history = {
        "policy": [],
        "gdp_growth": [],
        "inflation": [],
        "unemployment": [],
        "debt_to_gdp": [],
    }

    policy_counts = {EXPANSIONARY: 0, HAWKISH: 0, AUSTERITY: 0}

    for _ in range(num_cycles):
        policy = economy.step()
        policy_counts[policy] += 1

        history["policy"].append(policy)
        history["gdp_growth"].append(economy.gdp_growth)
        history["inflation"].append(economy.inflation)
        history["unemployment"].append(economy.unemployment)
        history["debt_to_gdp"].append(economy.debt_to_gdp)

    print("==========================================================")
    print(f"       MACROECONOMIC SIMULATION REPORT ({num_cycles:,} CYCLES)")
    print("==========================================================")
    print("\n--- Policy Regime Distribution ---")
    for pol, count in policy_counts.items():
        pct = (count / num_cycles) * 100
        print(f"  • {pol:<15}: {count:>6,} periods ({pct:>5.1f}%)")

    print("\n--- Aggregate Economic Indicators ---")
    print(
        f"  • Avg GDP Growth Rate : {statistics.mean(history['gdp_growth']):.2f}%"
    )
    print(
        f"  • Avg Inflation Rate  : {statistics.mean(history['inflation']):.2f}%"
    )
    print(
        f"  • Avg Unemployment    : {statistics.mean(history['unemployment']):.2f}%"
    )
    print(
        f"  • Avg Debt-to-GDP     : {statistics.mean(history['debt_to_gdp']):.2f}%"
    )
    print("==========================================================\n")

    return history


# Select your simulation length: 100, 1000, or 10000
if __name__ == "__main__":
    random.seed(42)

    # Change parameter to 100, 1000, or 10000
    SIMULATION_CYCLES = 1000
    run_macro_simulation(num_cycles=SIMULATION_CYCLES)