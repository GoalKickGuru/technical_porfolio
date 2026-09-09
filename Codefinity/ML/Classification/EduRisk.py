"""EduRisk — student on-track vs at-risk classification helpers.

Synthetic SIS extract: data/students.csv (7,208 × 24 letter codes before clean).
Target status: o = on-track, r = at-risk.

NOT a grading, placement, scholarship, or expulsion engine. A hold-out score
on this codebook does not transfer to a live roster.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_CANDIDATES = [
    Path("data/students.csv"),
    Path("/home/workdir/artifacts/data/students.csv"),
]


def find_data() -> Path:
    for p in DATA_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("students.csv not found under data/")


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else find_data()
    return pd.read_csv(path)


def clean(df: pd.DataFrame, unknown: str = "u", drop_dups: bool = True) -> pd.DataFrame:
    """Replace advisor '?' with `unknown` (task: 'u'). Optionally drop exact dups."""
    out = df.copy()
    if "advisor" in out.columns:
        out["advisor"] = out["advisor"].replace("?", unknown)
    if drop_dups:
        out = out.drop_duplicates().reset_index(drop=True)
    return out


def drop_zero_variance(df: pd.DataFrame, extra: list[str] | None = None) -> pd.DataFrame:
    drop = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if extra:
        drop = list(dict.fromkeys(drop + extra))
    return df.drop(columns=drop, errors="ignore")


def label_encode_all(df: pd.DataFrame):
    from sklearn.preprocessing import LabelEncoder

    encoded = df.copy()
    encoders = {}
    for col in encoded.columns:
        le = LabelEncoder()
        encoded[col] = le.fit_transform(encoded[col].astype(str))
        encoders[col] = le
    return encoded, encoders


def split_xy(encoded: pd.DataFrame, target: str = "status", test_size: float = 0.2, random_state: int = 42):
    from sklearn.model_selection import train_test_split

    X = encoded.drop(columns=[target])
    y = encoded[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def attendance_majority_rule(df: pd.DataFrame) -> pd.Series:
    """Map each attendance code to its majority status. Strong baseline on this table."""
    maj = df.groupby("attendance")["status"].agg(lambda s: s.value_counts().idxmax())
    return df["attendance"].map(maj)


def costly_errors(y_true, y_pred, pos_label=1) -> int:
    """Count actual-at-risk / predicted-on-track cells (missed intervention)."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    return int(((yt == pos_label) & (yp != pos_label)).sum())
