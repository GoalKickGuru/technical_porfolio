"""FX_Quote_GD_Sklearn — logistic GD + sklearn for FX quote lift / conversion.

Currency-conversion adaptation of Coursera C1_W3 Lab06 + Lab07.
Short name chosen for GitHub path limits.
"""
from __future__ import annotations

import copy
import math

import numpy as np


def sigmoid(z):
    z = np.clip(np.asarray(z, dtype=float), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def compute_cost_logistic(X, y, w, b):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f = np.clip(sigmoid(np.dot(X[i], w) + b), 1e-15, 1.0 - 1e-15)
        cost += -(y[i] * np.log(f) + (1.0 - y[i]) * np.log(1.0 - f))
    return cost / m


def compute_cost_logistic_vec(X, y, w, b):
    f = np.clip(sigmoid(np.asarray(X, dtype=float) @ np.asarray(w, dtype=float) + b), 1e-15, 1.0 - 1e-15)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(f) + (1.0 - y) * np.log(1.0 - f)))


def compute_gradient_logistic(X, y, w, b):
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
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    err = sigmoid(X @ np.asarray(w, dtype=float) + b) - y
    m = X.shape[0]
    return float(np.mean(err)), (X.T @ err) / m


def gradient_descent(X, y, w_in, b_in, alpha, num_iters, grad_fn=None, verbose=True):
    grad_fn = grad_fn or compute_gradient_logistic
    J_history = []
    w = copy.deepcopy(np.asarray(w_in, dtype=float))
    b = float(b_in)
    for i in range(num_iters):
        dj_db, dj_dw = grad_fn(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        J_history.append(compute_cost_logistic(X, y, w, b))
        if verbose and i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}")
    return w, b, J_history


def predict_proba(X, w, b):
    return sigmoid(np.asarray(X, dtype=float) @ np.asarray(w, dtype=float) + b)


def predict_lift(X, w, b, threshold=0.5):
    """1 = client lifts the FX quote / completes the conversion."""
    return (predict_proba(X, w, b) >= threshold).astype(int)
