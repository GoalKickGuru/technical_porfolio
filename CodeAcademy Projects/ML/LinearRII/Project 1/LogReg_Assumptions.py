"""
LogReg_Assumptions — reusable helpers for assumption checks,
sklearn logistic fit, threshold sweeps, ROC, and imbalance tools.

Short name (GitHub): LogReg_Assumptions
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


MEAN_FEATURES = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
]

CORE_FEATURES = [
    "radius_mean",
    "texture_mean",
    "compactness_mean",
    "symmetry_mean",
]


def load_breast_cancer(path="data/breast_cancer_data.csv"):
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0}).astype(int)
    return df


def events_per_variable(y, per=10):
    """Rule of thumb: smallest class size / per → max features."""
    counts = pd.Series(y).value_counts()
    return float(counts.min() / per)


def filter_high_quantile(df, col, q=0.99):
    hi = df[col].quantile(q)
    return df.loc[df[col] < hi].copy(), float(hi)


def make_logreg(class_weight=None, max_iter=4000):
    """Unregularized intercept model. penalty=None (sklearn ≥1.2)."""
    return LogisticRegression(
        penalty=None,
        fit_intercept=True,
        solver="lbfgs",
        max_iter=max_iter,
        class_weight=class_weight,
    )


def fit_eval(X_train, y_train, X_test, y_test, class_weight=None):
    model = make_logreg(class_weight=class_weight)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_prob),
        "cm": confusion_matrix(y_test, y_pred),
    }
    return model, y_pred, y_prob, metrics


def predict_at_threshold(proba, t=0.5):
    return (np.asarray(proba) >= t).astype(int)


def threshold_sweep(y_true, proba, ts=None):
    if ts is None:
        ts = np.linspace(0.05, 0.95, 19)
    rows = []
    for t in ts:
        pred = predict_at_threshold(proba, t)
        cm = confusion_matrix(y_true, pred)
        tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else (0, 0, 0, 0))
        rows.append(
            {
                "threshold": t,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "recall": recall_score(y_true, pred, zero_division=0),
                "precision": precision_score(y_true, pred, zero_division=0),
                "accuracy": accuracy_score(y_true, pred),
            }
        )
    return pd.DataFrame(rows)


def first_threshold_with_fn_at_most(y_true, proba, max_fn=2, grid=100):
    """Lowest threshold on a linspace that still keeps FN ≤ max_fn? 
    Raising t increases FN. We want the *highest* t that still has FN ≤ max_fn
    if the clinical goal is 'no more than max_fn misses' while limiting FP.
    The Codecademy checkpoint asked for the lowest t at which FN first reaches 2
    when sweeping 0→1 (i.e. the point FN becomes 2)."""
    ts = np.linspace(0, 1, grid)
    fns = []
    for t in ts:
        cm = confusion_matrix(y_true, predict_at_threshold(proba, t))
        fns.append(int(cm[1, 0]) if cm.shape == (2, 2) else 0)
    fns = np.asarray(fns)
    idx = np.argmax(fns >= max_fn)
    return float(ts[idx]), fns


def roc_points(y_true, proba):
    fpr, tpr, thr = roc_curve(y_true, proba)
    return fpr, tpr, thr, float(roc_auc_score(y_true, proba))


def stratified_split(X, y, test_size=0.3, random_state=6):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
