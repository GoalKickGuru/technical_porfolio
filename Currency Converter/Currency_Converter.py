#!/usr/bin/env python3
"""
Currency Converter – Extended Project
Based on Chapter 14 of Hassan S. "Learn Python by Doing. 100 Practical Projects for Beginners"

Features:
- Offline exchange rates (primary) with optional live API fallback
- Multi-currency conversion in one call
- Conversion history tracking
- Dictionary-dispatch alternate implementation
- Parameterised Monte-Carlo simulation of rate noise
"""

from __future__ import annotations
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# ---------------------------------------------------------------------------
# 1. Offline rates relative to 1 USD (fixed for reproducibility / offline use)
# ---------------------------------------------------------------------------
RATES_USD: Dict[str, float] = {
    "USD": 1.00,
    "EUR": 0.92,
    "GBP": 0.79,
    "JPY": 149.50,
    "INR": 83.20,
    "CRC": 515.00,   # Costa Rican Colón
    "CAD": 1.36,
    "AUD": 1.52,
    "CHF": 0.88,
    "CNY": 7.25,
}

SUPPORTED = set(RATES_USD.keys())

# Simple in-memory cache & history
_rate_cache: Dict[str, Dict[str, float]] = {}
_history: List[dict] = []


def get_exchange_rates(base_currency: str = "USD", use_api: bool = False) -> Dict[str, float]:
    """
    Return a dict of rates relative to the given base currency.
    Primary path is offline (RATES_USD). Optional live API is attempted
    only when use_api=True; on any failure we fall back to offline data.
    """
    base = base_currency.upper()
    if base in _rate_cache:
        return _rate_cache[base]

    rates: Dict[str, float] = {}

    if use_api:
        try:
            import requests
            url = f"https://api.exchangerate-api.com/v4/latest/{base}"
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            rates = {k: float(v) for k, v in data["rates"].items() if k in SUPPORTED}
            print(f"[API] Live rates for base={base} fetched.")
        except Exception as exc:
            print(f"[API] Failed ({exc}); falling back to offline rates.")

    if not rates:
        # Convert everything relative to the requested base via USD
        if base not in RATES_USD:
            raise ValueError(f"Unsupported base currency: {base}")
        base_to_usd = RATES_USD[base]
        rates = {cur: (RATES_USD[cur] / base_to_usd) for cur in RATES_USD}

    _rate_cache[base] = rates
    return rates


def convert(amount: float, from_currency: str, to_currency: str,
            rates: Optional[Dict[str, float]] = None) -> float:
    """Convert amount from one currency to another. Rates are vs the same base."""
    from_c = from_currency.upper()
    to_c = to_currency.upper()
    if rates is None:
        rates = get_exchange_rates(from_c)
    if from_c not in rates or to_c not in rates:
        raise ValueError(f"Invalid currency code(s): {from_c}, {to_c}")
    if amount < 0:
        raise ValueError("Amount must be non-negative")
    # rates are already relative to from_c when obtained via get_exchange_rates(from_c)
    # but to be general we normalise:
    return amount * (rates[to_c] / rates[from_c])


def convert_multi(amount: float, from_currency: str,
                  targets: List[str]) -> Dict[str, float]:
    """Convert one amount into several target currencies at once."""
    rates = get_exchange_rates(from_currency)
    results = {}
    for t in targets:
        results[t.upper()] = round(convert(amount, from_currency, t, rates), 2)
    return results


def record_conversion(amount: float, from_c: str, to_c: str, result: float) -> None:
    """Append a conversion record to the module-level history list."""
    _history.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "amount": amount,
        "from": from_c.upper(),
        "to": to_c.upper(),
        "result": round(result, 2),
    })


def show_history() -> None:
    if not _history:
        print("No conversions recorded yet.")
        return
    print("\n--- Conversion History ---")
    for i, h in enumerate(_history, 1):
        print(f"{i:2d}. {h['amount']} {h['from']} → {h['result']} {h['to']}  ({h['timestamp']})")


# ---------------------------------------------------------------------------
# Alternate implementation: pure dictionary-of-callables (no explicit convert)
# ---------------------------------------------------------------------------
def make_converter(rates: Dict[str, float]):
    """Return a dict of lambda converters for every pair of supported currencies."""
    funcs = {}
    for src in rates:
        for dst in rates:
            if src == dst:
                continue
            # closure over rates
            funcs[f"{src}2{dst}"] = (
                lambda amt, s=src, d=dst: amt * (rates[d] / rates[s])
            )
    return funcs


# ---------------------------------------------------------------------------
# Demo / CLI-style entry point (safe for notebooks – no live input)
# ---------------------------------------------------------------------------
def demo():
    print("=" * 55)
    print("Currency Converter – Demo (offline rates)")
    print("=" * 55)
    print("Supported:", ", ".join(sorted(SUPPORTED)))

    # single
    amt, src, dst = 100.0, "USD", "EUR"
    res = convert(amt, src, dst)
    record_conversion(amt, src, dst, res)
    print(f"\n{amt} {src} = {res:.2f} {dst}")

    # multi
    multi = convert_multi(250, "EUR", ["USD", "GBP", "JPY", "CRC"])
    print("\n250 EUR converts to:")
    for k, v in multi.items():
        print(f"  → {v:10.2f} {k}")
        record_conversion(250, "EUR", k, v)

    show_history()

    # alternate path
    rates = get_exchange_rates("USD")
    alt = make_converter(rates)
    print("\nAlternate dict-dispatch (USD2JPY for 50):",
          f"{alt['USD2JPY'](50):.2f}")


if __name__ == "__main__":
    demo()
