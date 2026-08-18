"""Predictive Maintenance vs. Dynamic Scheduling Simulation.

Simulates and compares three manufacturing maintenance strategies:
1. Run to Failure (RTF)
2. Strict Preventive Maintenance (PM)
3. Predictive Maintenance (PdM) based on sensor thresholds
"""

import random
import statistics


class Machine:
    def __init__(self, failure_threshold: float = 100.0):
        self.wear = 0.0
        self.failure_threshold = failure_threshold
        self.is_broken = False

    def operate(self) -> float:
        """Simulates one production cycle, accumulating wear and sensor noise."""
        if self.is_broken:
            return self.wear

        # Base wear increment with process variability
        wear_increment = random.uniform(2.0, 5.0)
        self.wear += wear_increment

        if self.wear >= self.failure_threshold:
            self.is_broken = True

        return self.wear

    def read_sensor(self) -> float:
        """Simulates a telemetry sensor reading with slight noise."""
        sensor_noise = random.uniform(-1.5, 1.5)
        return max(0.0, self.wear + sensor_noise)

    def service(self) -> None:
        """Resets machine state back to nominal conditions."""
        self.wear = 0.0
        self.is_broken = False


def simulate_strategy(
    strategy: str,
    total_cycles: int = 1000,
    pm_interval: int = 25,
    pdm_sensor_threshold: float = 80.0
) -> dict:
    """Runs a production simulation for a specific maintenance strategy.
    
    Costs:
    - Production profit: +$100 per successful cycle
    - Planned maintenance cost: -$300 (downtime: 2 cycles)
    - Unplanned breakdown repair cost: -$1500 (downtime: 8 cycles)
    """
    machine = Machine(failure_threshold=100.0)
    
    total_profit = 0
    successful_cycles = 0
    planned_maintenances = 0
    unplanned_breakdowns = 0
    downtime_remaining = 0

    for cycle in range(1, total_cycles + 1):
        # Handle downtime when machine is being repaired or serviced
        if downtime_remaining > 0:
            downtime_remaining -= 1
            if downtime_remaining == 0:
                machine.service()
            continue

        # Strategy-specific decision logic
        should_maintenance = False

        if strategy == "PM":
            # Scheduled maintenance every X cycles
            if cycle % pm_interval == 0:
                should_maintenance = True

        elif strategy == "PdM":
            # Sensor-driven predictive trigger
            sensor_reading = machine.read_sensor()
            if sensor_reading >= pdm_sensor_threshold:
                should_maintenance = True

        # Perform planned maintenance if triggered
        if should_maintenance:
            planned_maintenances += 1
            total_profit -= 300  # Cost of planned maintenance
            downtime_remaining = 2  # 2 cycles lost
            continue

        # Run machine cycle
        current_wear = machine.operate()

        # Check for catastrophic breakdown
        if machine.is_broken:
            unplanned_breakdowns += 1
            total_profit -= 1500  # High breakdown & emergency repair cost
            downtime_remaining = 8  # 8 cycles lost due to emergency repair
        else:
            successful_cycles += 1
            total_profit += 100  # Revenue earned

    return {
        "strategy": strategy,
        "total_profit": total_profit,
        "successful_cycles": successful_cycles,
        "planned_maintenances": planned_maintenances,
        "unplanned_breakdowns": unplanned_breakdowns,
    }


def run_comparison(num_runs: int = 50, cycles_per_run: int = 1000) -> None:
    """Runs multiple simulation passes to account for stochastic variation."""
    results = {"RTF": [], "PM": [], "PdM": []}

    for _ in range(num_runs):
        for strat in ["RTF", "PM", "PdM"]:
            res = simulate_strategy(strat, total_cycles=cycles_per_run)
            results[strat].append(res["total_profit"])

    print("==========================================================")
    print(" PREDICTIVE MAINTENANCE VS DYNAMIC SCHEDULING SIMULATION  ")
    print("==========================================================")
    print(f"Simulation Runs: {num_runs} | Cycles per Run: {cycles_per_run}\n")
    print(f"{'Strategy':<25} | {'Avg Profit ($)':<15} | {'Std Dev ($)':<12}")
    print("-" * 60)

    for strat, profits in results.items():
        avg_p = statistics.mean(profits)
        std_p = statistics.stdev(profits)
        strategy_names = {
            "RTF": "1. Run to Failure (RTF)",
            "PM": "2. Preventive (PM)",
            "PdM": "3. Predictive (PdM)"
        }
        print(f"{strategy_names[strat]:<25} | ${avg_p:>13,.2f} | ${std_p:>10,.2f}")
    print("-" * 60)


if __name__ == "__main__":
    random.seed(42)  # For reproducible simulation results
    run_comparison()