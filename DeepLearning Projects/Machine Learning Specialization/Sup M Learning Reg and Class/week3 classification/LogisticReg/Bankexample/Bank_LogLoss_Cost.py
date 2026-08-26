"""Bank_LogLoss_Cost — score frozen PD scorecards with logistic loss / book cost J.

Not expected credit loss. Use ecl() when finance asks for dollars.
"""
from __future__ import annotations
import numpy as np


def sigmoid(z):
    z = np.clip(np.asarray(z, dtype=float), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def loan_loss(f, y):
    f = np.clip(np.asarray(f, dtype=float), 1e-15, 1.0 - 1e-15)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(f) + (1.0 - y) * np.log(1.0 - f))


def compute_cost_logistic(X, y, w, b):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    m = X.shape[0]
    acc = 0.0
    for i in range(m):
        z = (w * X[i] + b) if X.ndim == 1 else (np.dot(X[i], w) + b)
        acc += float(loan_loss(sigmoid(z), y[i]))
    return acc / m


def compute_cost_logistic_vec(X, y, w, b):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    z = (w * X + b) if X.ndim == 1 else (X @ w + b)
    return float(np.mean(loan_loss(sigmoid(z), y)))


def predict_pd(X, w, b):
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)
    z = (w * X + b) if X.ndim == 1 else (X @ w + b)
    return sigmoid(z)


def expected_credit_loss(pd, lgd, ead):
    return np.asarray(pd, dtype=float) * lgd * ead
