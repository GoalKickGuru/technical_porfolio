"""
LogReg_Cyber — phishing / auth-abuse logistic helpers.

Short name (GitHub): LogReg_Cyber
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

ALL_FEATURES = [
    "url_length",
    "path_length",
    "num_dots",
    "num_hyphens",
    "num_subdomains",
    "has_ip",
    "has_https",
    "has_at",
    "digit_ratio",
    "redirect_count",
    "request_hour",
]

CORE_FEATURES = [
    "url_length",
    "num_subdomains",
    "has_ip",
    "has_https",
    "digit_ratio",
]

LOGIN_FEATURES = [
    "failed_logins",
    "src_unique_ua",
    "geo_rare",
    "off_hours",
]


def load_phishing(path="data/cyber_phishing.csv"):
    df = pd.read_csv(path)
    df["phishing"] = df["phishing"].map({"PHISH": 1, "BENIGN": 0}).astype(int)
    return df


def load_logins(path="data/cyber_logins.csv"):
    return pd.read_csv(path)


def events_per_variable(y, per=10):
    return float(pd.Series(y).value_counts().min() / per)


def filter_high_quantile(df, col, q=0.99):
    hi = float(df[col].quantile(q))
    return df.loc[df[col] < hi].copy(), hi


def make_logreg(class_weight=None, max_iter=4000):
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
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "recall": recall_score(y_true, pred, zero_division=0),
                "precision": precision_score(y_true, pred, zero_division=0),
                "accuracy": accuracy_score(y_true, pred),
            }
        )
    return pd.DataFrame(rows)
