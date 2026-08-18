"""Million Dice Roll Statistics Simulator (Extended)
By Al Sweigart al@inventwithpython.com — Extended Edition

A flexible dice-roll statistics simulator supporting custom dice sides,
variable roll counts, theoretical probability comparison, CSV export,
and a text-based histogram.

View the original at https://nostarch.com/big-book-small-python-projects
Tags: small, intermediate, math, simulation, statistics, cli
"""

import argparse
import csv
import random
import sys
import time
from itertools import product


# ──────────────────────────────────────────────
#   Theoretical Probability Calculation
# ──────────────────────────────────────────────

def theoretical_probabilities(num_dice, dice_sides):
    """Compute exact probabilities for every possible sum by enumerating
    all combinations (brute-force).  Returns a dict {sum: percentage}."""
    outcomes = {}
    total_outcomes = dice_sides ** num_dice

    # Enumerate every possible roll combination
    for combo in product(range(1, dice_sides + 1), repeat=num_dice):
        s = sum(combo)
        outcomes[s] = outcomes.get(s, 0) + 1

    return {s: (count / total_outcomes) * 100 for s, count in outcomes.items()}


# ──────────────────────────────────────────────
#   Simulation Engine
# ──────────────────────────────────────────────

def simulate_rolls(num_dice, dice_sides, num_rolls, progress=True):
    """Roll *num_dice* dice with *dice_sides* faces each, *num_rolls* times.
    Returns a dict mapping each sum → occurrence count."""
    min_sum = num_dice
    max_sum = num_dice * dice_sides
    results = {i: 0 for i in range(min_sum, max_sum + 1)}

    interval = num_rolls / 100  # update roughly every 1%
    next_milestone = interval
    last_print_time = time.time()

    print(f"Simulating {num_rolls:,} rolls of {num_dice}× d{dice_sides}...")
    start = time.time()

    for i in range(num_rolls):
        # Roll all dice and sum
        total = 0
        for _ in range(num_dice):
            total += random.randint(1, dice_sides)
        results[total] += 1

        if progress and (i + 1) >= next_milestone:
            pct = round((i + 1) / num_rolls * 100, 1)
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  {pct}% done... ({rate:,.0f} rolls/sec)")
            next_milestone += interval
            last_print_time = time.time()

    elapsed = time.time() - start
    print(f"Done! Completed {num_rolls:,} rolls in {elapsed:.2f}s "
          f"({num_rolls / elapsed:,.0f} rolls/sec)\n")
    return results


# ──────────────────────────────────────────────
#   Output & Visualisation
# ──────────────────────────────────────────────

def print_table(results, theoretical, num_rolls, num_dice, dice_sides):
    """Pretty-print the results table with empirical %, theoretical %,
    difference, and an ASCII bar-chart histogram."""
    min_sum = num_dice
    max_sum = num_dice * dice_sides

    header = (
        f"{'SUM':>5} │ {'ROLLS':>10} │ {'EMPIRICAL':>10} │ "
        f"{'THEORETICAL':>11} │ {'DIFF':>7} │ HISTOGRAM"
    )
    print(header)
    print("─" * len(header))

    # Find max percentage for scaling the histogram bar width
    max_pct = max(
        (results[s] / num_rolls * 100) for s in range(min_sum, max_sum + 1)
    )

    for s in range(min_sum, max_sum + 1):
        rolls = results[s]
        emp_pct = rolls / num_rolls * 100
        theo_pct = theoretical.get(s, 0.0)
        diff = emp_pct - theo_pct
        # Scale bar to max 40 characters
        bar_len = int(emp_pct / max_pct * 40) if max_pct > 0 else 0
        bar = "█" * bar_len
        print(
            f"{s:>5} │ {rolls:>10,} │ {emp_pct:>9.2f}% │ "
            f"{theo_pct:>10.2f}% │ {diff:>+6.2f}% │ {bar}"
        )


def print_summary_stats(results, num_rolls, num_dice, dice_sides):
    """Print mean, median, mode, variance, and standard deviation."""
    min_sum = num_dice
    max_sum = num_dice * dice_sides

    total = 0
    total_sq = 0
    all_values = []
    for s in range(min_sum, max_sum + 1):
        count = results[s]
        total += s * count
        total_sq += s * s * count
        all_values.extend([s] * count)

    mean = total / num_rolls
    variance = (total_sq / num_rolls) - (mean ** 2)
    stdev = variance ** 0.5

    # Mode (most frequent sum)
    mode_sum = max(results, key=results.get)
    mode_count = results[mode_sum]

    # Median (approximate — find the sum at which cumulative count crosses 50%)
    cumulative = 0
    median = None
    for s in range(min_sum, max_sum + 1):
        cumulative += results[s]
        if cumulative >= num_rolls / 2:
            median = s
            break

    # Expected mean for n fair dice with s sides = n*(s+1)/2
    expected_mean = num_dice * (dice_sides + 1) / 2

    print("\n── Summary Statistics ───────────────────────")
    print(f"  Mean (empirical) : {mean:.4f}")
    print(f"  Mean (expected)  : {expected_mean:.4f}")
    print(f"  Median           : {median}")
    print(f"  Mode             : {mode_sum} (rolled {mode_count:,} times)")
    print(f"  Variance         : {variance:.4f}")
    print(f"  Std deviation    : {stdev:.4f}")
    print("─────────────────────────────────────────────\n")


def export_csv(results, theoretical, num_rolls, filename):
    """Export the results table to a CSV file."""
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Sum", "Rolls", "Empirical_Pct", "Theoretical_Pct", "Diff_Pct"])
        for s in sorted(results.keys()):
            emp_pct = results[s] / num_rolls * 100
            theo_pct = theoretical.get(s, 0.0)
            diff = emp_pct - theo_pct
            writer.writerow([s, results[s], f"{emp_pct:.4f}", f"{theo_pct:.4f}", f"{diff:+.4f}"])
    print(f"Results exported to '{filename}'\n")


# ──────────────────────────────────────────────
#   Interactive Mode
# ──────────────────────────────────────────────

def interactive_mode():
    """Prompt the user for parameters interactively."""
    print("=" * 60)
    print("  MILLION DICE ROLL STATISTICS SIMULATOR (Extended)")
    print("  Based on Al Sweigart's original project")
    print("=" * 60)

    while True:
        try:
            num_dice = int(input("\nNumber of dice to roll [2]: ").strip() or "2")
            if num_dice < 1:
                raise ValueError
            break
        except ValueError:
            print("  ⚠  Please enter a positive integer.")

    while True:
        try:
            dice_sides = int(input("Number of sides per die [6]: ").strip() or "6")
            if dice_sides < 2:
                raise ValueError
            break
        except ValueError:
            print("  ⚠  Please enter an integer ≥ 2.")

    while True:
        try:
            num_rolls_input = input("Number of simulation rolls [1,000,000]: ").strip() or "1000000"
            num_rolls = int(num_rolls_input.replace(",", ""))
            if num_rolls < 1:
                raise ValueError
            break
        except ValueError:
            print("  ⚠  Please enter a positive integer.")

    return num_dice, dice_sides, num_rolls


# ──────────────────────────────────────────────
#   CLI Argument Parsing
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Million Dice Roll Statistics Simulator (Extended)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python milliondicestats_ext.py\n"
            "  python milliondicestats_ext.py --dice 3 --sides 6 --rolls 5000000\n"
            "  python milliondicestats_ext.py -d 4 -s 20 -r 100000 --csv d20_results.csv\n"
            "  python milliondicestats_ext.py -d 1 -s 2 -r 1000000  # coin toss\n"
        ),
    )
    parser.add_argument("-d", "--dice", type=int, default=None,
                        help="Number of dice (default: 2)")
    parser.add_argument("-s", "--sides", type=int, default=None,
                        help="Number of sides per die (default: 6)")
    parser.add_argument("-r", "--rolls", type=int, default=None,
                        help="Number of simulation rolls (default: 1,000,000)")
    parser.add_argument("--csv", metavar="FILE", default=None,
                        help="Export results to a CSV file")
    parser.add_argument("--no-progress", action="store_true",
                        help="Suppress progress messages during simulation")
    parser.add_argument("--quiet", action="store_true",
                        help="Only print the table, skip summary stats")
    return parser.parse_args()


# ──────────────────────────────────────────────
#   Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    # Decide whether to use CLI args or interactive prompts
    cli_provided = args.dice is not None and args.sides is not None and args.rolls is not None
    if cli_provided:
        num_dice = args.dice
        dice_sides = args.sides
        num_rolls = args.rolls
        if num_dice < 1:
            print("Error: --dice must be ≥ 1")
            sys.exit(1)
        if dice_sides < 2:
            print("Error: --sides must be ≥ 2")
            sys.exit(1)
        if num_rolls < 1:
            print("Error: --rolls must be ≥ 1")
            sys.exit(1)
    else:
        num_dice, dice_sides, num_rolls = interactive_mode()

    show_progress = not args.no_progress

    # ── Compute theoretical probabilities ──
    print("Computing theoretical probabilities...", end=" ", flush=True)
    theoretical = theoretical_probabilities(num_dice, dice_sides)
    print("Done.")

    # ── Run the simulation ──
    results = simulate_rolls(num_dice, dice_sides, num_rolls, progress=show_progress)

    # ── Display results ──
    print_table(results, theoretical, num_rolls, num_dice, dice_sides)

    if not args.quiet:
        print_summary_stats(results, num_rolls, num_dice, dice_sides)

    # ── Export to CSV if requested ──
    if args.csv:
        export_csv(results, theoretical, num_rolls, args.csv)


if __name__ == "__main__":
    main()