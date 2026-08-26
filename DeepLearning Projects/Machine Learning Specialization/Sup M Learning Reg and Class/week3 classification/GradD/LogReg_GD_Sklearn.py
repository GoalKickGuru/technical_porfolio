"""LogReg_GD_Sklearn — helpers for C1_W3 Lab06 (GD) + Lab07 (sklearn).

Short name chosen because GitHub rejected long paths on earlier projects.
"""
from __future__ import annotations

import copy
import math

import numpy as np


def sigmoid(z):
    z = np.clip(np.asarray(z, dtype=float), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def compute_cost_logistic(X, y, w, b):
    """Loop form of binary cross-entropy (matches the Coursera lab)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f = sigmoid(np.dot(X[i], w) + b)
        f = np.clip(f, 1e-15, 1.0 - 1e-15)
        cost += -(y[i] * np.log(f) + (1.0 - y[i]) * np.log(1.0 - f))
    return cost / m


def compute_cost_logistic_vec(X, y, w, b):
    """Vectorized binary cross-entropy."""
    f = sigmoid(np.asarray(X, dtype=float) @ np.asarray(w, dtype=float) + b)
    f = np.clip(f, 1e-15, 1.0 - 1e-15)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(f) + (1.0 - y) * np.log(1.0 - f)))


def compute_gradient_logistic(X, y, w, b):
    """Loop form of equations (2)–(3) in Lab06."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0.0
    for i in range(m):
        err = sigmoid(np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] += err * X[i, j]
        dj_db += err
    return dj_db / m, dj_dw / m


def compute_gradient_logistic_vec(X, y, w, b):
    """Vectorized gradient: (1/m) X.T @ (f - y), (1/m) sum(f - y)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    err = sigmoid(X @ w + b) - y
    m = X.shape[0]
    return float(np.mean(err)), (X.T @ err) / m


def gradient_descent(X, y, w_in, b_in, alpha, num_iters, cost_fn=None, grad_fn=None, verbose=True):
    """Batch GD. Returns w, b, J_history."""
    cost_fn = cost_fn or compute_cost_logistic
    grad_fn = grad_fn or compute_gradient_logistic
    J_history = []
    w = copy.deepcopy(np.asarray(w_in, dtype=float))
    b = float(b_in)
    for i in range(num_iters):
        dj_db, dj_dw = grad_fn(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i < 100000:
            J_history.append(cost_fn(X, y, w, b))
        if verbose and i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}")
    return w, b, J_history


def predict_proba(X, w, b):
    X = np.asarray(X, dtype=float)
    return sigmoid(X @ np.asarray(w, dtype=float) + b)


def predict(X, w, b, threshold=0.5):
    return (predict_proba(X, w, b) >= threshold).astype(int)
