"""CrimRisk — incident priority classification helpers.

Synthetic RMS extract: data/incidents.csv.
Target priority: p = priority, r = routine.

NOT a bail, sentencing, charging, or person-profiling engine.
This flags *cases* from field codes, not people from demographics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_CANDIDATES = [
    Path("data/incidents.csv"),
    Path("/home/workdir/artifacts/data/incidents.csv"),
]


def find_data() -> Path:
    for p in DATA_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("incidents.csv not found under data/")


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else find_data()
    return pd.read_csv(path)


def clean(df: pd.DataFrame, unknown: str = "u", drop_dups: bool = True) -> pd.DataFrame:
    """Replace clearance '?' with `unknown`. Optionally drop exact dups."""
    out = df.copy()
    if "clearance" in out.columns:
        out["clearance"] = out["clearance"].replace("?", unknown)
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


def split_xy(encoded: pd.DataFrame, target: str = "priority", test_size: float = 0.2, random_state: int = 42):
    from sklearn.model_selection import train_test_split

    X = encoded.drop(columns=[target])
    y = encoded[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def weapon_majority_rule(df: pd.DataFrame) -> pd.Series:
    maj = df.groupby("weapon")["priority"].agg(lambda s: s.value_counts().idxmax())
    return df["weapon"].map(maj)


def costly_errors(y_true, y_pred, priority_code=0) -> int:
    """Count actual-priority / predicted-routine cells (missed priority case).

    Default LabelEncoder maps p→0, r→1, so priority_code=0.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    return int(((yt == priority_code) & (yp != priority_code)).sum())
