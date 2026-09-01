"""HousingCT_Py — NYC one-bed rents, mean / median / mode.

Short-name companion to HousingCT_Py_Solution.ipynb.
Not rental advice. Advertised StreetEasy asks ≠ signed leases.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATA = Path(__file__).resolve().parent / "data"


def load() -> dict[str, pd.Series]:
    return {
        "Brooklyn": pd.read_csv(DATA / "brooklyn-one-bed.csv")["rent"],
        "Manhattan": pd.read_csv(DATA / "manhattan-one-bed.csv")["rent"],
        "Queens": pd.read_csv(DATA / "queens-one-bed.csv")["rent"],
    }


def central(x: pd.Series) -> dict:
    x = pd.Series(x).dropna().astype(float)
    mode_res = stats.mode(x, keepdims=True)
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "mode": float(mode_res.mode[0]),
        "mode_count": int(mode_res.count[0]),
        "mode_vc": int(x.value_counts().idxmax()),
        "mode_counter": int(Counter(x).most_common(1)[0][0]),
    }


def main() -> None:
    groups = load()
    print(f"{'Borough':<12} {'n':>5} {'mean':>10} {'median':>10} {'mode':>8} {'count':>6}")
    for name, x in groups.items():
        s = central(x)
        print(
            f"{name:<12} {s['n']:>5} {s['mean']:10.2f} {s['median']:10.1f} "
            f"{s['mode']:8.0f} {s['mode_count']:6d}"
        )


if __name__ == "__main__":
    main()
