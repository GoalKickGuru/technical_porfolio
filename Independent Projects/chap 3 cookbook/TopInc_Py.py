"""TopInc_Py — reusable helpers for World Top Incomes Database extracts.

Short name: TopInc_Py
Source pattern: Packt Practical Data Science Cookbook 2e, Chapter 3.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np

DEFAULT_SHARE_COLS = (
    "Top 10% income share",
    "Top 5% income share",
    "Top 1% income share",
    "Top 0.5% income share",
    "Top 0.1% income share",
)


def to_float(x) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def dataset(
    path: str | Path,
    filter_field: Optional[str] = None,
    filter_value: Optional[str] = None,
) -> Iterator[dict]:
    """Yield CSV rows. Optionally keep only rows matching one field."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if filter_field is None:
            yield from reader
        else:
            yield from (row for row in reader if row.get(filter_field) == filter_value)


def timeseries(rows: Iterable[dict], column: str) -> Iterator[tuple[int, float]]:
    for row in rows:
        val = to_float(row.get(column, ""))
        if val is not None:
            yield int(row["Year"]), val


def normalize(ts: Iterable[tuple[int, float]]) -> list[tuple[int, float]]:
    data = list(ts)
    arr = np.array([v for _, v in data], dtype="f8")
    arr = arr / arr.mean()
    return list(zip((y for y, _ in data), arr))


def delta(
    first: Iterable[tuple[int, float]],
    second: Iterable[tuple[int, float]],
) -> list[tuple[int, float]]:
    a = {y: v for y, v in first}
    b = {y: v for y, v in second}
    return [(y, a[y] - b[y]) for y in sorted(set(a) & set(b))]
