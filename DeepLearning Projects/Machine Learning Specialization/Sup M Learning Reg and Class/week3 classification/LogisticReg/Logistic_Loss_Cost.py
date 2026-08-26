"""Logistic_Loss_Cost — reusable helpers for C1_W3 Lab04/Lab05 style work.

Short name chosen because long GitHub paths were rejected on earlier projects.
"""
from __future__ import annotations

import numpy as np


def sigmoid(z):
    z = np.clip(np.asarray(z, dtype=float), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def logistic_loss(f, y):
    """Per-example binary logistic loss (compact form)."""
    f = np.clip(np.asarray(f, dtype=float), 1e-15, 1.0 - 1e-15)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(f) + (1.0 - y) * np.log(1.0 - f))


def compute_cost_logistic(X, y, w, b):
    """Loop form (matches the original Coursera lab)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = (w * X[i] + b) if X.ndim == 1 else (np.dot(X[i], w) + b)
        cost += float(logistic_loss(sigmoid(z_i), y[i]))
    return cost / m


def compute_cost_logistic_vec(X, y, w, b):
    """Vectorized binary cross-entropy."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    z = (w * X + b) if X.ndim == 1 else (X @ w + b)
    return float(np.mean(logistic_loss(sigmoid(z), y)))


def predict_proba(X, w, b):
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)
    z = (w * X + b) if X.ndim == 1 else (X @ w + b)
    return sigmoid(z)
