"""Health Insurance Adverse Selection Simulator — The Death Spiral
A Monte Carlo simulation showing how insurance markets destabilize from
asymmetric information and voluntary enrollment.

Based on econ research by Akerlof (1970) on "lemons markets" and
contemporary health economics (Finkelstein et al., 2019).

Each period:
  • Agents have PRIVATE knowledge of their own health risk
  • The insurer sees only the average risk of ENROLLED agents
  • Premium = average claimed cost + loading fee
  • Agents decide: buy or stay uninsured based on premium vs. expected cost

Result: Without mandates or subsidies, the pool spirals toward collapse.

Tags: intermediate, economics, health, simulation, policy, cli
"""

import argparse
import csv
import random
import sys
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple

# ──────────────────────────────────────────────
#   Data Structures
# ──────────────────────────────────────────────

@dataclass
class Agent:
    id: int
    health_risk: float          # 0.0 (healthy) to 1.0 (very sick)
    age_group: int             # 1=young, 2=middle, 3=senior (optional)
    enrolled: bool = False     # Currently insured?
    
    def expected_cost(self, base_cost: float = 5000) -> float:
        """Expected annual medical cost for this agent."""
        return self.health_risk * base_cost
    
    def utility_buy_insurance(self, premium: float, base_cost: float = 5000) -> float:
        """Utility of buying insurance (negative = disutility)."""
        # Risk-averse: pay slightly more than expected cost for certainty
        risk_aversion = 0.15  # Willing to pay 15% premium over expected cost
        max_willing_to_pay = self.expected_cost(base_cost) * (1 + risk_aversion)
        return -premium if premium <= max_willing_to_pay else float('-inf')
    
    def utility_stay_uninsured(self, base_cost: float = 5000) -> float:
        """Utility of staying uninsured (expected cost of random illness)."""
        # Uninsured agents face expected cost but with variance
        expected_loss = self.expected_cost(base_cost) * 0.7  # Some costs self-pay
        return -expected_loss


# ──────────────────────────────────────────────
#   Distribution Generation
# ──────────────────────────────────────────────

def generate_population(
    num_agents: int,
    skew: str = "uniform"
) -> List[Agent]:
    """Generate a population with varying health risk distribution.
    
    skews:
      - "uniform": flat distribution (unrealistic but instructive)
      - "beta_low": most people healthy, few very sick (realistic US pattern)
      - "bimodal": two peaks (young healthy + old sick)
    """
    agents = []
    for i in range(num_agents):
        if skew == "uniform":
            risk = random.uniform(0.05, 0.95)
        elif skew == "beta_low":
            # Beta(2,5) gives most mass at low values, tail at high
            risk = random.betavariate(2, 5)
        elif skew == "bimodal":
            # Half young/healthy, half older/sicker
            if i < num_agents // 2:
                risk = random.betavariate(3, 4)  # Lower end
            else:
                risk = random.betavariate(4, 2)  # Higher end
        else:
            risk = random.uniform(0.05, 0.95)
        
        # Age correlation (older = slightly higher risk on average)
        age_probs = [0.4, 0.4, 0.2]  # 40% young, 40% middle, 20% senior
        age_group = random.choices([1, 2, 3], weights=age_probs)[0]
        
        agents.append(Agent(id=i, health_risk=risk, age_group=age_group))
    
    return agents


# ──────────────────────────────────────────────
#   Simulation Engine
# ──────────────────────────────────────────────

def run_single_period(
    population: List[Agent],
    premium: float,
    base_cost: float = 5000,
    mandate: bool = False,
    subsidy_rate: float = 0.0,
    employer_coverage_pct: float = 0.0,
) -> Tuple[int, float]:
    """Run one enrollment period. Returns (enrolled_count, avg_claim_cost)."""
    
    if mandate:
        # Everyone enrolled regardless of decision
        for agent in population:
            agent.enrolled = True
    else:
        # Employer coverage removes some healthy people before individual market
        num_employer = int(len(population) * employer_coverage_pct)
        employer_ids = set(random.sample(range(len(population)), num_employer))
        
        for agent in population:
            if agent.id in employer_ids:
                agent.enrolled = False  # Covered through employer, opt out of individual
                continue
            
            # Decision: buy insurance or not?
            utility_buy = agent.utility_buy_insurance(premium, base_cost)
            utility_uninsured = agent.utility_stay_uninsured(base_cost)
            
            # Apply subsidy to premium
            effective_premium = premium * (1 - subsidy_rate)
            utility_buy_subsidized = -effective_premium if effective_premium <= agent.expected_cost(base_cost) * 1.15 else float('-inf')
            
            agent.enrolled = utility_buy_subsidized > utility_uninsured

    # Calculate claims from enrolled pool
    enrolled = [a for a in population if a.enrolled]
    if len(enrolled) == 0:
        return 0, 0.0
    
    total_claims = sum(a.expected_cost(base_cost) for a in enrolled)
    avg_claim_cost = total_claims / len(enrolled)
    
    return len(enrolled), avg_claim_cost


def run_death_spiral_simulation(
    population: List[Agent],
    num_periods: int = 50,
    base_cost: float = 5000,
    loading_fee_pct: float = 0.10,
    mandate: bool = False,
    subsidy_rate: float = 0.0,
    employer_coverage_pct: float = 0.0,
    progress: bool = True,
) -> Dict:
    """Simulate the insurance market over multiple periods.
    
    Returns historical data for plotting and analysis.
    """
    history = {
        "period": [],
        "enrolled": [],
        "enrolled_pct": [],
        "avg_risk_in_pool": [],
        "premium": [],
        "total_claims": [],
        "healthy_enrolled": [],
        "sick_enrolled": [],
    }
    
    # Initial premium guess (based on full population risk)
    total_risk = sum(a.health_risk for a in population)
    initial_avg_risk = total_risk / len(population)
    premium = base_cost * initial_avg_risk * (1 + loading_fee_pct)
    
    print(f"\n┌{'─' * 50}┐")
    print(f"│  INSURANCE MARKET SIMULATION — {len(population)} agents            │")
    print(f"├{'─' * 50}┤")
    print(f"│  Base medical cost:        ${base_cost:,.0f}                       │")
    print(f"│  Loading fee:              {loading_fee_pct:.0%}                   │")
    if mandate:
        print(f"│  MANDATE active            (everyone must enroll)                  │")
    if subsidy_rate > 0:
        print(f"│  Subsidy rate:             {subsidy_rate:.0%}                      │")
    if employer_coverage_pct > 0:
        print(f"│  Employer coverage:        {employer_coverage_pct:.0%}                 │")
    print(f"└{'─' * 50}┘\n")
    
    start_time = time.time()
    
    for period in range(num_periods):
        enrolled_count, avg_claim_cost = run_single_period(
            population, premium, base_cost, mandate, subsidy_rate, employer_coverage_pct
        )
        
        if enrolled_count == 0:
            # Market collapsed
            history["period"].append(period)
            history["enrolled"].append(0)
            history["enrolled_pct"].append(0.0)
            history["avg_risk_in_pool"].append(0.0)
            history["premium"].append(float('inf'))
            history["total_claims"].append(0.0)
            history["healthy_enrolled"].append(0)
            history["sick_enrolled"].append(0)
            if progress:
                print(f"  Period {period}: MARKET COLLAPSED — no enrollees!")
            break
        
        # Update premium for next period based on actual pool risk
        total_risk_in_pool = sum(a.health_risk for a in population if a.enrolled)
        avg_risk_in_pool = total_risk_in_pool / enrolled_count
        new_premium = base_cost * avg_risk_in_pool * (1 + loading_fee_pct)
        
        # Track healthy vs. sick (threshold = 0.5 risk)
        healthy_threshold = 0.3
        healthy_enrolled = sum(1 for a in population if a.enrolled and a.health_risk < healthy_threshold)
        sick_enrolled = sum(1 for a in population if a.enrolled and a.health_risk >= healthy_threshold)
        
        # Record history
        history["period"].append(period)
        history["enrolled"].append(enrolled_count)
        history["enrolled_pct"].append(enrolled_count / len(population) * 100)
        history["avg_risk_in_pool"].append(avg_risk_in_pool)
        history["premium"].append(premium)
        history["total_claims"].append(avg_claim_cost * enrolled_count)
        history["healthy_enrolled"].append(healthy_enrolled)
        history["sick_enrolled"].append(sick_enrolled)
        
        if progress:
            elapsed = time.time() - start_time
            pct_change = ((new_premium - premium) / premium * 100) if premium > 0 else 0
            print(f"  Period {period:2d}: {enrolled_count:4d} enrolled ({enrolled_count/len(population)*100:5.1f}%), "
                  f"premium ${premium:,.0f} → ${new_premium:,.0f} ({pct_change:+.1f}%), "
                  f"avg_risk={avg_risk_in_pool:.2f}")
        
        premium = new_premium
    
    elapsed = time.time() - start_time
    print(f"\n  Completed {len(history['period'])} periods in {elapsed:.2f}s\n")
    
    return history


# ──────────────────────────────────────────────
#   Analysis & Metrics
# ──────────────────────────────────────────────

def calculate_gini(values: List[float]) -> float:
    """Calculate Gini coefficient for a list of values."""
    n = len(values)
    if n == 0:
        return 0.0
    sorted_v = sorted(values)
    cumulative = sum((2 * i - n - 1) * v for i, v in enumerate(sorted_v, 1))
    total = sum(sorted_v)
    return cumulative / (n * total) if total > 0 else 0.0


def analyze_history(history: Dict, scenario_name: str) -> Dict:
    """Extract summary metrics from simulation history."""
    if len(history["premium"]) == 0:
        return {"scenario": scenario_name, "collapsed": True}
    
    final_period = history["period"][-1]
    final_enrolled = history["enrolled"][-1]
    final_premium = history["premium"][-1]
    final_avg_risk = history["avg_risk_in_pool"][-1]
    
    # Did the market collapse?
    collapsed = final_enrolled == 0 or len(history["premium"]) > 1 and \
                abs(history["premium"][-1] - history["premium"][-2]) > history["premium"][-2] * 0.5
    
    # Premium trajectory
    if len(history["premium"]) > 1:
        premium_increase_pct = (history["premium"][-1] - history["premium"][0]) / history["premium"][0] * 100
    else:
        premium_increase_pct = 0
    
    # Enrollee composition stability
    if len(history["healthy_enrolled"]) > 1:
        healthy_decline = history["healthy_enrolled"][0] - history["healthy_enrolled"][-1]
    else:
        healthy_decline = 0
    
    return {
        "scenario": scenario_name,
        "collapsed": collapsed,
        "final_enrolled": final_enrolled,
        "final_premium": final_premium,
        "final_avg_risk": final_avg_risk,
        "premium_increase_pct": premium_increase_pct,
        "healthy_decline": healthy_decline,
        "total_periods": len(history["period"]),
    }


def print_scenario_analysis(analyses: List[Dict]):
    """Print a formatted comparison table of multiple scenarios."""
    if not analyses:
        return
    
    print(f"\n{'═' * 80}")
    print(f"  SCENARIO COMPARISON")
    print(f"{'═' * 80}")
    
    header = (
        f"  {'Scenario':<25} {'Status':>10} {'Final Pm':>10} "
        f"{'Δ Premium':>12} {'Enrolled':>10} {'Risk':>6}"
    )
    print(header)
    print(f"  {'─' * 77}")
    
    for a in analyses:
        status = "COLLAPSED" if a["collapsed"] else "STABLE"
        pm_str = f"${a['final_premium']:,.0f}" if not a["collapsed"] else "---"
        delta_str = f"{a['premium_increase_pct']:+.0f}%" if not a["collapsed"] else "---"
        enrolled_str = f"{a['final_enrolled']:,}" if not a["collapsed"] else "---"
        risk_str = f"{a['final_avg_risk']:.2f}" if not a["collapsed"] else "---"
        
        print(f"  {a['scenario']:<25} {status:>10} {pm_str:>10} "
              f"{delta_str:>12} {enrolled_str:>10} {risk_str:>6}")
    
    print(f"  {'═' * 80}\n")


def print_market_state(history: Dict):
    """Print a detailed breakdown of the final market state."""
    if len(history["premium"]) == 0:
        print("  No data to display (market had no initial enrollment)")
        return
    
    final_idx = len(history["premium"]) - 1
    print(f"\n┌{'─' * 50}┐")
    print(f"│  FINAL MARKET STATE (Period {history['period'][final_idx]})         │")
    print(f"├{'─' * 50}┤")
    print(f"│  Enrolled:                 {history['enrolled'][final_idx]:,}                         │")
    print(f"│  Enrollment Rate:          {history['enrolled_pct'][final_idx]:.1f}%                       │")
    print(f"│  Final Premium:            ${history['premium'][final_idx]:,.0f}                       │")
    print(f"│  Avg Risk in Pool:         {history['avg_risk_in_pool'][final_idx]:.2f}                    │")
    print(f"│  Healthy Enrolled (<30%):  {history['healthy_enrolled'][final_idx]:,}                         │")
    print(f"│  Sick Enrolled (≥30%):     {history['sick_enrolled'][final_idx]:,}                         │")
    print(f"└{'─' * 50}┘\n")


def print_advice(scenario_name: str, analysis: Dict):
    """Print policy advice based on scenario outcome."""
    print(f"\n{'─' * 50}")
    print(f"  POLICY IMPLICATIONS: {scenario_name}")
    print(f"{'─' * 50}")
    
    if analysis.get("collapsed"):
        print("  ❌ Market FAILURE — insurance unavailable")
        print("     Without intervention, voluntary markets can collapse entirely.")
        print("     Policy solutions: mandate enrollment, subsidies, or risk adjustment.")
    elif analysis.get("premium_increase_pct", 0) > 100:
        print("  ⚠️  SEVERE stress — unsustainable premium spiral")
        print(f"     Premiums increased {analysis['premium_increase_pct']:.0f}% over simulation.")
        print("     Consider subsidies or risk adjustment mechanisms.")
    else:
        print("  ✓  Market functioning relatively well")
        print("     Stable enrollment and reasonable premium growth.")
        print("     Continue monitoring for adverse selection pressures.")
    
    print()


# ──────────────────────────────────────────────
#   Visualization
# ──────────────────────────────────────────────

def print_ascii_chart(history: Dict, title: str = "Premium Trajectory"):
    """Print a text-based line chart of premium over time."""
    if len(history["premium"]) < 2:
        print("  Not enough data to chart")
        return
    
    periods = history["period"]
    values = history["premium"]
    
    width = 60
    height = 12
    max_val = max(values)
    min_val = min(values)
    
    if max_val == min_val:
        print("  Flat line — no variation in premiums")
        return
    
    # Build grid
    grid = [[" " for _ in range(width + 1)] for _ in range(height + 1)]
    
    # Plot points
    for i, (p, v) in enumerate(zip(periods, values)):
        col = min(int(i / (len(periods) - 1) * width) if len(periods) > 1 else 0, width)
        row = height - 1 - int((v - min_val) / (max_val - min_val) * (height - 1))
        if 0 <= row < height and 0 <= col <= width:
            grid[row][col] = "█"
    
    # Add Y-axis labels
    print(f"\n  {title}")
    print(f"  ${max_val:,.0f} │" + "".join(grid[0]))
    for r in range(1, height - 1):
        pct = (height - 1 - r) / (height - 1) * 100
        print(f"        │" + "".join(grid[r]))
    print(f"  ${min_val:,.0f} │" + "".join(grid[height - 1]))
    print(f"        └{'─' * width}")
    print(f"         Period 0 {'─' * (width - 15)} Period {len(periods) - 1}")
    print()


def print_enrollment_funnel(history: Dict):
    """Show enrollment trajectory as an ASCII funnel."""
    if len(history["enrolled_pct"]) < 2:
        return
    
    print(f"\n── Enrollment Funnel ────────────────────────────────────────────")
    for i, pct in enumerate(history["enrolled_pct"][::5][:6]):  # Every 5th period, max 6
        bar_len = int(pct / 100 * 40)
        bar = "█" * bar_len
        period = history["period"][i * 5]
        print(f"  Period {period:2d}: [{bar:<40}] {pct:.1f}%")
    print()


# ──────────────────────────────────────────────
#   CSV Export
# ──────────────────────────────────────────────

def export_csv(history: Dict, scenario_name: str, filename: str):
    """Export simulation history to CSV."""
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Period", "Enrolled", "Enrolled_Pct", "Avg_Risk", 
                        "Premium", "Total_Claims", "Healthy_Enrolled", "Sick_Enrolled"])
        for i in range(len(history["period"])):
            writer.writerow([
                history["period"][i],
                history["enrolled"][i],
                f"{history['enrolled_pct'][i]:.2f}",
                f"{history['avg_risk_in_pool'][i]:.4f}",
                f"{history['premium'][i]:.2f}",
                f"{history['total_claims'][i]:.2f}",
                history["healthy_enrolled"][i],
                history["sick_enrolled"][i],
            ])
    print(f"  Exported '{scenario_name}' to '{filename}'\n")


# ──────────────────────────────────────────────
#   CLI
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Health Insurance Adverse Selection Simulator — The Death Spiral",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python insurance_death_spiral.py                              # Interactive\n"
            "  python insurance_death_spiral.py -n 2000 -p 50 --compare      # Full comparison\n"
            "  python insurance_death_spiral.py -n 1000 -p 30 -s 0.15        # With 15% subsidy\n"
            "\n"
            "Key insight: Voluntary insurance markets tend toward collapse\n"
            "because healthy people opt out when premiums exceed their\n"
            "expected costs, leaving only the sick who drive premiums\n"
            "higher still—a 'death spiral.'\n"
        ),
    )
    parser.add_argument("-n", "--agents", type=int, default=None,
                        help="Number of agents (default: 1000)")
    parser.add_argument("-p", "--periods", type=int, default=None,
                        help="Number of simulation periods (default: 50)")
    parser.add_argument("--compare", action="store_true",
                        help="Run comparison: voluntary, mandate, 15% subsidy")
    parser.add_argument("-s", "--subsidy", type=float, default=None,
                        help="Subsidy rate (e.g., 0.15 for 15%%)")
    parser.add_argument("--mandate", action="store_true",
                        help="Enable individual mandate (all must enroll)")
    parser.add_argument("--employer", type=float, default=None,
                        help="Employer coverage percentage (0.0-1.0)")
    parser.add_argument("--csv", metavar="FILE", default=None,
                        help="Export results to CSV")
    parser.add_argument("--distribution", type=str, default="beta_low",
                        choices=["uniform", "beta_low", "bimodal"],
                        help="Initial health risk distribution (default: beta_low)")
    parser.add_argument("--no-progress", action="store_true",
                        help="Suppress progress messages")
    return parser.parse_args()


def interactive_mode():
    print("=" * 60)
    print("  HEALTH INSURANCE DEATH SPIRAL SIMULATOR")
    print("  Based on Akerlof (1970) 'Market for Lemons'")
    print("=" * 60)
    print()
    print("  Each period:")
    print("    • Agents know their OWN health risk")
    print("    • Insurers only see AVERAGE risk of ENROLLEES")
    print("    • Premium adjusts to cover actual pool claims")
    print("    • Healthy agents may drop out if premium > expected cost")
    print()
    print("  Watch what happens WITHOUT mandates or subsidies...")
    print()
    
    while True:
        try:
            n = int(input("  Number of agents [1000]: ").strip() or "1000")
            if n < 50:
                raise ValueError
            break
        except ValueError:
            print("  ⚠  Please enter an integer ≥ 50.")
    
    while True:
        try:
            p = int(input("  Number of periods [50]: ").strip() or "50")
            if p < 5:
                raise ValueError
            break
        except ValueError:
            print("  ⚠  Please enter an integer ≥ 5.")
    
    choice = input("\n  Run comparison mode (voluntary vs. mandate vs. subsidy)? [y/N]: ").strip().lower()
    compare = choice in ("y", "yes")
    
    return n, p, compare


# ──────────────────────────────────────────────
#   Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()
    show_progress = not args.no_progress
    
    if args.agents is not None and args.periods is not None:
        num_agents = args.agents
        num_periods = args.periods
        compare = args.compare
        if num_agents < 50:
            print("Error: --agents must be ≥ 50")
            sys.exit(1)
        if num_periods < 5:
            print("Error: --periods must be ≥ 5")
            sys.exit(1)
    else:
        num_agents, num_periods, compare = interactive_mode()
    
    base_cost = 5000
    loading_fee = 0.10
    
    analyses = []
    
    if compare:
        print("\n" + "╔" + "═" * 58 + "╗")
        print("║  COMPARISON MODE: Testing Three Policy Regimes             ║")
        print("╚" + "═" * 58 + "╝\n")
        
        # ── Scenario 1: Voluntary enrollment (classic death spiral) ──
        population = generate_population(num_agents, args.distribution)
        history1 = run_death_spiral_simulation(
            population, num_periods, base_cost, loading_fee,
            mandate=False, subsidy_rate=0.0, employer_coverage_pct=0.0,
            progress=show_progress
        )
        analysis1 = analyze_history(history1, "Voluntary (No Intervention)")
        analyses.append(analysis1)
        print_market_state(history1)
        print_advice("Voluntary (No Intervention)", analysis1)
        print_ascii_chart(history1, "Premium Over Time (Voluntary)")
        print_enrollment_funnel(history1)
        if args.csv:
            export_csv(history1, "Voluntary", args.csv.replace(".csv", "_voluntary.csv"))
        
        # ── Scenario 2: Individual mandate ──
        population = generate_population(num_agents, args.distribution)
        history2 = run_death_spiral_simulation(
            population, num_periods, base_cost, loading_fee,
            mandate=True, subsidy_rate=0.0, employer_coverage_pct=0.0,
            progress=show_progress
        )
        analysis2 = analyze_history(history2, "Individual Mandate")
        analyses.append(analysis2)
        print_market_state(history2)
        print_advice("Individual Mandate", analysis2)
        print_ascii_chart(history2, "Premium Over Time (Mandate)")
        print_enrollment_funnel(history2)
        if args.csv:
            export_csv(history2, "Mandate", args.csv.replace(".csv", "_mandate.csv"))
        
        # ── Scenario 3: Subsidy ──
        subsidy = args.subsidy if args.subsidy is not None else 0.15
        population = generate_population(num_agents, args.distribution)
        history3 = run_death_spiral_simulation(
            population, num_periods, base_cost, loading_fee,
            mandate=False, subsidy_rate=subsidy, employer_coverage_pct=0.0,
            progress=show_progress
        )
        analysis3 = analyze_history(history3, f"{int(subsidy*100)}% Subsidy")
        analyses.append(analysis3)
        print_market_state(history3)
        print_advice(f"{int(subsidy*100)}% Subsidy", analysis3)
        print_ascii_chart(history3, f"Premium Over Time ({int(subsidy*100)}% Subsidy)")
        print_enrollment_funnel(history3)
        if args.csv:
            export_csv(history3, "Subsidy", args.csv.replace(".csv", "_subsidy.csv"))
        
        # ── Comparison table ──
        print_scenario_analysis(analyses)
        
        # ── Cross-scenario chart ──
        print("\n" + "═" * 60)
        print("  KEY FINDING")
        print("═" * 60)
        print("""
        Without mandates or subsidies, voluntary insurance markets tend to:
        
          1. Lose healthy enrollees first (they perceive premium > value)
          2. Leave only high-risk individuals in the pool
          3. Force insurers to raise premiums to cover costs
          4. Trigger more healthy exits — a self-reinforcing spiral
        
        This explains why real-world reforms (ACA, etc.) combine:
          • Individual mandates OR automatic enrollment
          • Subsidies for lower-income buyers  
          • Risk-adjustment payments between insurers
          • Employer coverage as a stabilizing anchor
        
        See also: Akerlof (1970) "The Market for Lemons"
        """)
    
    else:
        # Single scenario
        mandate = args.mandate
        subsidy = args.subsidy if args.subsidy is not None else 0.0
        employer = args.employer if args.employer is not None else 0.0
        
        scenario_name = "Voluntary"
        if mandate:
            scenario_name = "Mandate"
        elif subsidy > 0:
            scenario_name = f"Subsidy ({int(subsidy*100)}%)"
        elif employer > 0:
            scenario_name = f"Employer ({int(employer*100)}%)"
        
        population = generate_population(num_agents, args.distribution)
        history = run_death_spiral_simulation(
            population, num_periods, base_cost, loading_fee,
            mandate=mandate, subsidy_rate=subsidy, 
            employer_coverage_pct=employer,
            progress=show_progress
        )
        
        analysis = analyze_history(history, scenario_name)
        analyses.append(analysis)
        
        print_market_state(history)
        print_ascii_chart(history, "Premium Trajectory")
        print_enrollment_funnel(history)
        print_advice(scenario_name, analysis)
        
        if args.csv:
            export_csv(history, scenario_name, args.csv)
        
        print(f"\n💡 Tip: Run with --compare to see how policy choices change outcomes")
    
    # Save aggregate analysis
    if args.csv and len(analyses) > 1:
        with open(args.csv.replace(".csv", "_summary.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Scenario", "Status", "Final_Premium", "Delta_Premium", 
                           "Final_Enrolled", "Final_Risk", "Collaposed"])
            for a in analyses:
                writer.writerow([
                    a["scenario"],
                    "COLLAPSED" if a["collapsed"] else "STABLE",
                    f"{a.get('final_premium', 0):.2f}",
                    f"{a.get('premium_increase_pct', 0):.2f}",
                    a.get("final_enrolled", 0),
                    f"{a.get('final_avg_risk', 0):.4f}",
                    a["collapsed"],
                ])
        print(f"  Summary exported to '{args.csv.replace('.csv', '_summary.csv')}'\n")


if __name__ == "__main__":
    main()