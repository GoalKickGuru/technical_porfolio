"""Wealth Inequality Simulator — The Yard-Sale Model
A Monte Carlo simulation showing how inequality emerges from random transactions.

Inspired by Al Sweigart's "Million Dice Roll Statistics Simulator" and
the econophysics yard-sale model (Bouchaud & Mézard, 2000; Boghosian et al., 2017).

Two agents meet at random. They exchange a fraction γ of the *poorer* agent's
wealth, decided by a fair coin flip. Total wealth is conserved, yet repeated
fair trades spontaneously produce extreme inequality — an "oligarch" who
holds nearly everything. Adding a flat wealth tax-and-redistribute mechanism
produces a Pareto-tailed distribution that closely matches real-world data.

Run it to see how policy choices shape inequality — no assumptions about
greed, talent, or effort required.

Tags: intermediate, economics, simulation, statistics, policy, cli
"""

import argparse
import csv
import random
import sys
import time

# ──────────────────────────────────────────────
#   Inequality Metrics
# ──────────────────────────────────────────────

def gini_coefficient(wealths):
    """Compute the Gini coefficient (0 = perfect equality, 1 = perfect inequality)."""
    n = len(wealths)
    if n == 0:
        return 0.0
    sorted_w = sorted(wealths)
    cumulative = sum((2 * i - n - 1) * w for i, w in enumerate(sorted_w, 1))
    total = sum(sorted_w)
    if total == 0:
        return 0.0
    return cumulative / (n * total)


def lorenz_curve(wealths, num_points=20):
    """Compute Lorenz curve data points as (population_pct, wealth_pct) tuples."""
    n = len(wealths)
    if n == 0:
        return [(0, 0), (1, 1)]
    sorted_w = sorted(wealths)
    total = sum(sorted_w)
    if total == 0:
        return [(0, 0), (1, 1)]
    points = [(0.0, 0.0)]
    for i in range(1, num_points + 1):
        cutoff = int(n * i / num_points)
        share = sum(sorted_w[:cutoff]) / total
        points.append((i / num_points, share))
    return points


def wealth_shares(wealths, percentiles=(10, 20, 50, 90, 99)):
    """What share of total wealth is held by the bottom X% and top X%."""
    n = len(wealths)
    sorted_w = sorted(wealths)
    total = sum(sorted_w)
    if total == 0:
        return {}
    shares = {}
    for p in percentiles:
        cutoff = int(n * p / 100)
        share = sum(sorted_w[:cutoff]) / total * 100
        shares[f"Bottom {p}%"] = share
        shares[f"Top {100 - p}%"] = 100 - share
    return shares

# ──────────────────────────────────────────────
#   Simulation Engine
# ──────────────────────────────────────────────

def simulate_yardsale(
    num_agents,
    num_transactions,
    gamma=0.20,
    tax_rate=0.0,
    redistribution_freq=0,
    progress=True,
):
    """Run the yard-sale simulation.

    Parameters:
        num_agents:          Number of economic agents
        num_transactions:    Total random transactions to simulate
        gamma:               Fraction of poorer agent's wealth at stake per trade
        tax_rate:            Flat wealth tax applied periodically (0.0 = none)
        redistribution_freq: Apply tax every N transactions (0 = never)
        progress:            Show progress messages

    Returns:
        List of final wealth values, one per agent.
    """
    initial_wealth = 100.0
    wealths = [initial_wealth] * num_agents

    print(f"Simulating {num_transactions:,} transactions among {num_agents} agents...")
    print(f"  Trade size γ = {gamma:.0%} of poorer agent's wealth")
    if redistribution_freq > 0 and tax_rate > 0:
        print(f"  Wealth tax = {tax_rate:.1%} every {redistribution_freq:,} transactions")
    else:
        print("  No redistribution (pure yard-sale)")
    print()

    start_time = time.time()
    last_print = start_time
    interval = max(num_transactions // 100, 1)

    for t in range(num_transactions):
        # Pick two distinct random agents
        i = random.randrange(num_agents)
        j = random.randrange(num_agents - 1)
        if j >= i:
            j += 1

        wi = wealths[i]
        wj = wealths[j]

        # The amount at stake is γ × min(wi, wj)
        delta = gamma * min(wi, wj)

        # Fair coin flip: who wins?
        if random.random() < 0.5:
            wealths[i] += delta
            wealths[j] -= delta
        else:
            wealths[i] -= delta
            wealths[j] += delta

        # Prevent negative wealth (numerical safeguard)
        if wealths[i] < 0:
            wealths[i] = 0.0
        if wealths[j] < 0:
            wealths[j] = 0.0

        # Periodic wealth tax and redistribution
        if redistribution_freq > 0 and tax_rate > 0 and (t + 1) % redistribution_freq == 0:
            total_tax = 0.0
            for k in range(num_agents):
                tax = wealths[k] * tax_rate
                wealths[k] -= tax
                total_tax += tax
            # Redistribute evenly
            rebate = total_tax / num_agents
            for k in range(num_agents):
                wealths[k] += rebate

        # Progress report
        if progress and (t + 1) % interval == 0:
            now = time.time()
            if now - last_print >= 1.0:
                pct = (t + 1) / num_transactions * 100
                elapsed = now - start_time
                rate = (t + 1) / elapsed if elapsed > 0 else 0
                print(f"  {pct:.1f}% done... ({rate:,.0f} tx/sec)")
                last_print = now

    elapsed = time.time() - start_time
    print(f"\nDone! {num_transactions:,} transactions in {elapsed:.2f}s "
          f"({num_transactions / elapsed:,.0f} tx/sec)\n")

    return wealths

# ──────────────────────────────────────────────
#   Output & Visualisation
# ──────────────────────────────────────────────

def print_wealth_histogram(wealths, num_bins=25):
    """Print an ASCII histogram of the wealth distribution."""
    max_w = max(wealths)
    min_w = min(wealths)
    if max_w == min_w:
        print("  All agents have equal wealth — no histogram to show.")
        return

    bin_width = (max_w - min_w) / num_bins
    bins = [0] * num_bins
    for w in wealths:
        idx = min(int((w - min_w) / bin_width), num_bins - 1)
        bins[idx] += 1

    max_count = max(bins)
    print("── Wealth Distribution Histogram ──────────────────────────────────")
    print(f"  Range: ${min_w:.2f} – ${max_w:.2f}  ({num_bins} bins)")
    print()

    for b in range(num_bins):
        lo = min_w + b * bin_width
        hi = lo + bin_width
        count = bins[b]
        bar_len = int(count / max_count * 45) if max_count > 0 else 0
        bar = "█" * bar_len
        label = f"${lo:>8.1f}-${hi:>8.1f}"
        print(f"  {label} │ {count:>5} │ {bar}")

    print()


def print_lorenz_ascii(wealths):
    """Print a text-based Lorenz curve and the Gini coefficient."""
    points = lorenz_curve(wealths, num_points=20)
    gini = gini_coefficient(wealths)

    print("── Lorenz Curve (ASCII) ──────────────────────────────────────────")
    print(f"  Gini coefficient: {gini:.4f}")
    print()

    width = 40
    height = 15
    grid = [[" " for _ in range(width + 1)] for _ in range(height + 1)]

    # Draw equality line
    for i in range(width + 1):
        row = height - int(i / width * height)
        if 0 <= row <= height:
            grid[row][i] = "·"

    # Draw Lorenz curve
    for pop_pct, wealth_pct in points:
        col = int(pop_pct * width)
        row = height - int(wealth_pct * height)
        if 0 <= row <= height and 0 <= col <= width:
            grid[row][col] = "■"

    print("  Wealth %")
    for r in range(height + 1):
        pct_label = f"{(height - r) / height * 100:>5.0f}"
        print(f"  {pct_label} │{''.join(grid[r])}")
    print(f"        └{'─' * width}")
    print(f"         {''.join(f'{i * 100 // width:>3}' for i in range(width + 1))} Population %")
    print()
    print("  ■ = Lorenz curve    · = Perfect equality line")
    print()


def print_results_table(wealths, scenario_name, num_agents, num_transactions):
    """Print a detailed summary table for one scenario."""
    sorted_w = sorted(wealths, reverse=True)
    total = sum(wealths)
    gini = gini_coefficient(wealths)
    shares = wealth_shares(wealths)

    print(f"\n{'═' * 65}")
    print(f"  RESULTS: {scenario_name}")
    print(f"{'═' * 65}")
    print(f"  Agents:              {num_agents:,}")
    print(f"  Transactions:        {num_transactions:,}")
    print(f"  Total wealth:        ${total:,.2f}")
    print(f"  Mean wealth:         ${total / num_agents:,.2f}")
    print(f"  Median wealth:       ${sorted_w[len(sorted_w) // 2]:,.2f}")
    print(f"  Wealthiest agent:    ${sorted_w[0]:,.2f}")
    print(f"  Poorest agent:       ${sorted_w[-1]:,.2f}")
    print(f"  Gini coefficient:    {gini:.4f}")
    print(f"  (0 = perfect equality, 1 = one person owns everything)")
    print()
    print("── Wealth Concentration ──────────────────────────────────────────")
    for key, val in shares.items():
        bar_len = int(val / 100 * 40)
        bar = "█" * bar_len
        print(f"  {key:>12}: {val:>6.2f}% │ {bar}")
    print()

    # Top 10 wealth holders
    print("── Top 10 Wealthiest Agents ──────────────────────────────────────")
    for rank, w in enumerate(sorted_w[:10], 1):
        share = w / total * 100
        print(f"  #{rank:>2}: ${w:>10.2f}  ({share:>5.2f}% of total wealth)")
    print()

    # Bottom 10
    print("── Bottom 10 Poorest Agents ──────────────────────────────────────")
    for rank, w in enumerate(reversed(sorted_w[-10:]), 1):
        share = w / total * 100 if total > 0 else 0
        print(f"  #{rank:>2}: ${w:>10.2f}  ({share:>5.2f}% of total wealth)")
    print()


def print_comparison(scenarios):
    """Print a side-by-side comparison of multiple scenarios."""
    print(f"\n{'═' * 75}")
    print("  SCENARIO COMPARISON")
    print(f"{'═' * 75}")
    header = f"  {'Scenario':<35} {'Gini':>8} {'Top 10%':>10} {'Bottom 50%':>12}"
    print(header)
    print(f"  {'─' * 68}")
    for name, wealths in scenarios:
        gini = gini_coefficient(wealths)
        shares = wealth_shares(wealths)
        top10 = shares.get("Top 90%", 0)
        bot50 = shares.get("Bottom 50%", 0)
        print(f"  {name:<35} {gini:>8.4f} {top10:>9.2f}% {bot50:>11.2f}%")
    print(f"  {'─' * 68}")
    print("  Reference: USA Gini ≈ 0.85 (wealth), Sweden ≈ 0.58, Denmark ≈ 0.64")
    print()

# ──────────────────────────────────────────────
#   CSV Export
# ──────────────────────────────────────────────

def export_csv(wealths, scenario_name, filename):
    """Export final wealth distribution to CSV."""
    sorted_w = sorted(wealths, reverse=True)
    total = sum(wealths)
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Wealth", "Share_Pct", "Cumulative_Share_Pct"])
        cumulative = 0.0
        for rank, w in enumerate(sorted_w, 1):
            share = w / total * 100 if total > 0 else 0
            cumulative += share
            writer.writerow([rank, f"{w:.4f}", f"{share:.4f}", f"{cumulative:.4f}"])
    print(f"  Exported '{scenario_name}' to '{filename}'\n")

# ──────────────────────────────────────────────
#   CLI
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Wealth Inequality Simulator — The Yard-Sale Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python wealth_simulator.py                          # Interactive\n"
            "  python wealth_simulator.py -a 500 -t 2000000        # 500 agents, 2M tx\n"
            "  python wealth_simulator.py -a 500 -t 5000000 --compare\n"
            "  python wealth_simulator.py -a 1000 -t 5000000 --csv wealth.csv\n"
            "\n"
            "The yard-sale model shows that even perfectly fair random trades\n"
            "produce extreme inequality. Adding a small wealth tax produces a\n"
            "distribution that matches real-world Pareto tails.\n"
        ),
    )
    parser.add_argument("-a", "--agents", type=int, default=None,
                        help="Number of agents (default: 500)")
    parser.add_argument("-t", "--transactions", type=int, default=None,
                        help="Number of transactions (default: 2,000,000)")
    parser.add_argument("-g", "--gamma", type=float, default=None,
                        help="Trade size as fraction of poorer agent's wealth (default: 0.20)")
    parser.add_argument("--compare", action="store_true",
                        help="Run three scenarios: no tax, 1% tax, 3% tax")
    parser.add_argument("--csv", metavar="FILE", default=None,
                        help="Export final wealth distribution to CSV")
    parser.add_argument("--no-progress", action="store_true",
                        help="Suppress progress messages")
    parser.add_argument("--bins", type=int, default=25,
                        help="Number of histogram bins (default: 25)")
    return parser.parse_args()


def interactive_mode():
    print("=" * 65)
    print("  WEALTH INEQUALITY SIMULATOR — The Yard-Sale Model")
    print("  Based on econophysics research (Bouchaud & Mézard, 2000)")
    print("=" * 65)
    print()
    print("  Two agents meet at random. They stake a fraction γ of the")
    print("  POORER agent's wealth on a fair coin flip. Total wealth is")
    print("  conserved — yet inequality emerges spontaneously.")
    print()
    print("  With an optional wealth tax, the distribution stabilises into")
    print("  a Pareto tail matching real-world data.")
    print()

    while True:
        try:
            n = int(input("  Number of agents [500]: ").strip() or "500")
            if n < 10:
                raise ValueError
            break
        except ValueError:
            print("  ⚠  Please enter an integer ≥ 10.")

    while True:
        try:
            t_input = input("  Number of transactions [2,000,000]: ").strip() or "2000000"
            t = int(t_input.replace(",", ""))
            if t < 1000:
                raise ValueError
            break
        except ValueError:
            print("  ⚠  Please enter an integer ≥ 1,000.")

    while True:
        try:
            g_input = input("  Trade fraction γ (0.01–0.50) [0.20]: ").strip() or "0.20"
            g = float(g_input)
            if not 0.01 <= g <= 0.50:
                raise ValueError
            break
        except ValueError:
            print("  ⚠  Please enter a float between 0.01 and 0.50.")

    choice = input("\n  Run comparison mode (no-tax vs 1% vs 3% tax)? [y/N]: ").strip().lower()
    compare = choice in ("y", "yes")

    return n, t, g, compare

# ──────────────────────────────────────────────
#   Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()
    show_progress = not args.no_progress

    if args.agents is not None and args.transactions is not None:
        num_agents = args.agents
        num_transactions = args.transactions
        gamma = args.gamma if args.gamma is not None else 0.20
        compare = args.compare
        if num_agents < 10:
            print("Error: --agents must be ≥ 10")
            sys.exit(1)
        if num_transactions < 1000:
            print("Error: --transactions must be ≥ 1,000")
            sys.exit(1)
    else:
        num_agents, num_transactions, gamma, compare = interactive_mode()

    scenarios = {}

    if compare:
        # ── Scenario 1: Pure yard-sale (no redistribution) ──
        print("\n" + "║" * 65)
        print("║  SCENARIO 1: NO REDISTRIBUTION (Pure Yard-Sale)")
        print("║" * 65)
        w1 = simulate_yardsale(
            num_agents, num_transactions, gamma=gamma,
            tax_rate=0.0, redistribution_freq=0, progress=show_progress
        )
        print_results_table(w1, "No Redistribution", num_agents, num_transactions)
        print_wealth_histogram(w1, num_bins=args.bins)
        print_lorenz_ascii(w1)
        scenarios["No Tax"] = w1
        if args.csv:
            export_csv(w1, "No Tax", args.csv.replace(".csv", "_notax.csv"))

        # ── Scenario 2: 1% wealth tax ──
        print("\n" + "║" * 65)
        print("║  SCENARIO 2: 1% WEALTH TAX (every 10,000 transactions)")
        print("║" * 65)
        w2 = simulate_yardsale(
            num_agents, num_transactions, gamma=gamma,
            tax_rate=0.01, redistribution_freq=10000, progress=show_progress
        )
        print_results_table(w2, "1% Wealth Tax", num_agents, num_transactions)
        print_wealth_histogram(w2, num_bins=args.bins)
        print_lorenz_ascii(w2)
        scenarios["1% Wealth Tax"] = w2
        if args.csv:
            export_csv(w2, "1% Wealth Tax", args.csv.replace(".csv", "_1pct.csv"))

        # ── Scenario 3: 3% wealth tax ──
        print("\n" + "║" * 65)
        print("║  SCENARIO 3: 3% WEALTH TAX (every 10,000 transactions)")
        print("║" * 65)
        w3 = simulate_yardsale(
            num_agents, num_transactions, gamma=gamma,
            tax_rate=0.03, redistribution_freq=10000, progress=show_progress
        )
        print_results_table(w3, "3% Wealth Tax", num_agents, num_transactions)
        print_wealth_histogram(w3, num_bins=args.bins)
        print_lorenz_ascii(w3)
        scenarios["3% Wealth Tax"] = w3
        if args.csv:
            export_csv(w3, "3% Wealth Tax", args.csv.replace(".csv", "_3pct.csv"))

        # ── Comparison ──
        print_comparison(scenarios)

    else:
        # ── Single scenario: no tax ──
        wealths = simulate_yardsale(
            num_agents, num_transactions, gamma=gamma,
            tax_rate=0.0, redistribution_freq=0, progress=show_progress
        )
        print_results_table(wealths, "Yard-Sale (No Redistribution)",
                            num_agents, num_transactions)
        print_wealth_histogram(wealths, num_bins=args.bins)
        print_lorenz_ascii(wealths)

        if args.csv:
            export_csv(wealths, "Yard-Sale", args.csv)

    print("\n💡 Key insight: Even with perfectly fair coin flips and zero skill")
    print("   differences, random trades alone produce extreme inequality.")
    print("   A modest periodic wealth tax prevents oligarchy and produces a")
    print("   distribution resembling real-world Pareto tails.")


if __name__ == "__main__":
    main()