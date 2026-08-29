"""Energy_LogReg — generating-unit outage/derate logistic regression.

Teaching use only. Not an operating reliability tool. Synthetic blotter.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LESSON_FEATURES = [
    "age_years",
    "ambient_c",
    "capacity_mw",
    "load_factor",
    "days_since_maint",
    "is_peak_hour",
    "fuel",
    "region",
]


def load_units(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def make_xy(df: pd.DataFrame, features=None):
    features = list(dict.fromkeys(features or LESSON_FEATURES))
    X = pd.get_dummies(df[features], drop_first=True).astype(float)
    y = np.where(df["forced_outage"] == "No", 0, 1)
    return X, y


def fit_l1(X, y, test_size=0.2, seed=1, C=0.05, scale=False):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed)
    if scale:
        model = Pipeline(
            [
                ("sc", StandardScaler()),
                ("lr", LogisticRegression(C=C, penalty="l1", solver="liblinear")),
            ]
        )
    else:
        model = LogisticRegression(C=C, penalty="l1", solver="liblinear")
    model.fit(Xtr, ytr)
    return model, Xtr, Xte, ytr, yte


def report(model, Xte, yte, threshold=0.5):
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "acc": float(accuracy_score(yte, pred)),
        "prec": float(precision_score(yte, pred, zero_division=0)),
        "rec": float(recall_score(yte, pred, zero_division=0)),
        "auc": float(roc_auc_score(yte, proba)),
        "cm": confusion_matrix(yte, pred).tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/energy_units.csv")
    p.add_argument("--scale", action="store_true")
    args = p.parse_args()
    df = load_units(args.data)
    X, y = make_xy(df)
    model, Xtr, Xte, ytr, yte = fit_l1(X, y, scale=args.scale)
    print(df["forced_outage"].value_counts(normalize=True))
    print(report(model, Xte, yte))
    lr = model.named_steps["lr"] if args.scale else model
    print("intercept", float(lr.intercept_[0]))
    print(
        pd.DataFrame({"var": Xtr.columns, "coef": lr.coef_[0]})
        .query("coef.abs() > 0")
        .sort_values("coef")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
