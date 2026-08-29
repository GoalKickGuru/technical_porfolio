"""
LogReg_Tour — hotel-cancellation / tour-conversion logistic helpers.

Short name (GitHub): LogReg_Tour
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
)
from sklearn.model_selection import train_test_split

ALL_FEATURES = [
    "lead_time",
    "nights",
    "adr",
    "booking_value",
    "party_size",
    "prev_cancels",
    "special_requests",
    "is_weekend",
    "online_channel",
    "has_deposit",
    "checkin_hour",
]

CORE_FEATURES = [
    "lead_time",
    "prev_cancels",
    "online_channel",
    "has_deposit",
    "special_requests",
]

SESSION_FEATURES = [
    "page_views",
    "session_min",
    "promo",
    "repeat_visitor",
    "mobile",
]


def load_hotels(path="data/tour_hotels.csv"):
    df = pd.read_csv(path)
    df["cancelled"] = df["cancelled"].map({"CANCEL": 1, "STAY": 0}).astype(int)
    return df


def load_sessions(path="data/tour_sessions.csv"):
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
    return model, y_pred, y_prob, {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_prob),
        "cm": confusion_matrix(y_test, y_pred),
    }


def predict_at_threshold(proba, t=0.5):
    return (np.asarray(proba) >= t).astype(int)
