"""EV Charging Infrastructure vs. Local Grid Capacity Simulation.

Simulates neighborhood grid load under Uncontrolled, Smart (V1G), and V2G charging strategies.
"""

import random
import statistics


class NeighborhoodGrid:

    def __init__(self,
                 num_homes: int = 100,
                 transformer_limit_kw: float = 250.0):
        self.num_homes = num_homes
        self.transformer_limit_kw = transformer_limit_kw

    def get_base_load(self, hour: int) -> float:
        """Returns baseline residential load (non-EV) in kW depending on the hour (0-23)."""
        # Peak residential demand occurs between 17:00 and 21:00
        if 17 <= hour <= 21:
            base_per_home = random.uniform(1.8, 2.5)
        elif 0 <= hour <= 5:
            base_per_home = random.uniform(0.4, 0.7)
        else:
            base_per_home = random.uniform(0.8, 1.4)

        return self.num_homes * base_per_home


def simulate_grid_day(
    strategy: str,
    ev_penetration: float = 0.50,
    transformer_limit: float = 250.0,
    num_days: int = 30,
) -> dict:
    """Simulates daily 24-hour grid operation over a period of days.
    
    EV Specs:
    - Charger: 7.2 kW (Level 2)
    - Battery demand per day: ~15-30 kWh
    """
    grid = NeighborhoodGrid(
        num_homes=100, transformer_limit_kw=transformer_limit)
    num_evs = int(100 * ev_penetration)

    transformer_overloads = 0
    peak_loads = []
    unmet_ev_demand_kwh = 0

    for _ in range(num_days):
        # Assign daily energy requirements to each EV (15 to 30 kWh)
        ev_demand = [random.uniform(15.0, 30.0) for _ in range(num_evs)]

        for hour in range(24):
            total_load = grid.get_base_load(hour)
            ev_load = 0.0

            for i in range(num_evs):
                if ev_demand[i] <= 0:
                    continue

                # Strategy 1: Uncontrolled (Charge upon arrival at 17:00-22:00)
                if strategy == "UNCONTROLLED":
                    if 17 <= hour <= 23:
                        ev_load += 7.2
                        ev_demand[i] -= 7.2

                # Strategy 2: Smart Charging V1G (Shift to Off-Peak 23:00-06:00)
                elif strategy == "SMART_V1G":
                    if hour >= 23 or hour <= 6:
                        ev_load += 7.2
                        ev_demand[i] -= 7.2

                # Strategy 3: V2G + Storage (Discharge during Peak, Charge Off-Peak)
                elif strategy == "V2G":
                    if 17 <= hour <= 20:
                        # Feed 3.0 kW back into grid to support peak base load
                        ev_load -= 3.0
                    elif hour >= 23 or hour <= 5:
                        ev_load += 7.2
                        ev_demand[i] -= 7.2

            net_grid_load = total_load + ev_load
            peak_loads.append(net_grid_load)

            # Check for local transformer overload
            if net_grid_load > transformer_limit:
                transformer_overloads += 1

        # Track any uncharged range
        unmet_ev_demand_kwh += sum(max(0, d) for d in ev_demand)

    return {
        "strategy": strategy,
        "max_peak_load_kw": max(peak_loads),
        "avg_peak_load_kw": statistics.mean(peak_loads),
        "total_overload_hours": transformer_overloads,
        "unmet_ev_demand_kwh": unmet_ev_demand_kwh,
    }


def run_ev_grid_simulation(ev_adoption_rate: float = 0.60):
    """Executes comparison across 100, 1,000, or 10,000 simulated days."""
    random.seed(42)
    strategies = ["UNCONTROLLED", "SMART_V1G", "V2G"]
    results = {}

    print("==============================================================")
    print(" EV CHARGING INFRASTRUCTURE VS. LOCAL GRID CAPACITY SIMULATION")
    print("==============================================================")
    print(f"EV Penetration: {int(ev_adoption_rate * 100)}% of households")
    print("Transformer Capacity Limit: 250 kW\n")

    for strat in strategies:
        res = simulate_grid_day(
            strategy=strat,
            ev_penetration=ev_adoption_rate,
            transformer_limit=250.0,
            num_days=30,  # Change to 100, 1000, or 10000 for extended runs
        )
        results[strat] = res

        print(f"--- Strategy: {strat} ---")
        print(f"  • Max Peak Load Recorded : {res['max_peak_load_kw']:.1f} kW")
        print(
            f"  • Overload Instances     : {res['total_overload_hours']} hours (over 30 days)"
        )
        print(
            f"  • Unmet EV Charge Demand : {res['unmet_ev_demand_kwh']:.1f} kWh\n"
        )


if __name__ == "__main__":
    # Test at 60% EV adoption rate
    run_ev_grid_simulation(ev_adoption_rate=0.60)