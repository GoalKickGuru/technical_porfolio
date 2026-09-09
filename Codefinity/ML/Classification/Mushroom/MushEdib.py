"""MushEdib — mushroom edibility classification helpers.

UCI mushrooms.csv (8,124 × 23 letter codes). Target class: e=edible, p=poisonous.

NOT a foraging tool. A 100% hold-out score on this table does not transfer to
a mushroom you picked this morning.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_CANDIDATES = [
    Path("data/mushrooms.csv"),
    Path("/home/workdir/artifacts/data/mushrooms.csv"),
]


def find_data() -> Path:
    for p in DATA_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("mushrooms.csv not found under data/")


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else find_data()
    return pd.read_csv(path)


def clean(df: pd.DataFrame, unknown: str = "u", drop_dups: bool = True) -> pd.DataFrame:
    """Replace stalk-root '?' with `unknown` (task: 'u'). Optionally drop exact dups."""
    out = df.copy()
    if "stalk-root" in out.columns:
        out["stalk-root"] = out["stalk-root"].replace("?", unknown)
    if drop_dups:
        out = out.drop_duplicates().reset_index(drop=True)
    return out


def drop_zero_variance(df: pd.DataFrame, extra: list[str] | None = None) -> pd.DataFrame:
    drop = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if extra:
        drop = list(dict.fromkeys(drop + extra))
    return df.drop(columns=drop, errors="ignore")


def label_encode_all(df: pd.DataFrame):
    """Fit a LabelEncoder per column. Returns encoded frame + encoder dict."""
    from sklearn.preprocessing import LabelEncoder

    encoded = df.copy()
    encoders = {}
    for col in encoded.columns:
        le = LabelEncoder()
        encoded[col] = le.fit_transform(encoded[col].astype(str))
        encoders[col] = le
    return encoded, encoders


def split_xy(encoded: pd.DataFrame, target: str = "class", test_size: float = 0.2, random_state: int = 42):
    from sklearn.model_selection import train_test_split

    X = encoded.drop(columns=[target])
    y = encoded[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def odor_majority_rule(df: pd.DataFrame) -> pd.Series:
    """Map each odor to its majority class. Strong baseline (~0.985 on this table)."""
    maj = df.groupby("odor")["class"].agg(lambda s: s.value_counts().idxmax())
    return df["odor"].map(maj)


def costly_errors(y_true, y_pred, pos_label=1) -> int:
    """Count actual-poisonous / predicted-edible cells (the field-costly mistake)."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    return int(((yt == pos_label) & (yp != pos_label)).sum())
