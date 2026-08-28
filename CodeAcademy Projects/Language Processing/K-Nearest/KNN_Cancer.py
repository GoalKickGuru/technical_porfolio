"""KNN_Cancer — from-scratch Euclidean KNN + sklearn helpers.

Short GitHub name: KNN_Cancer
"""
from __future__ import annotations

import numpy as np


def distance(a, b) -> float:
    """Euclidean distance between two equal-length sequences."""
    squared = 0.0
    for i in range(len(a)):
        squared += (a[i] - b[i]) ** 2
    return squared ** 0.5


def distance_np(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def min_max_normalize(lst):
    minimum, maximum = min(lst), max(lst)
    span = maximum - minimum
    if span == 0:
        return [0.0 for _ in lst]
    return [(v - minimum) / span for v in lst]


def fit_minmax(dataset):
    X = np.array(list(dataset.values()), dtype=float)
    return X.min(axis=0), X.max(axis=0)


def apply_minmax(point, mins, maxs):
    point = np.asarray(point, dtype=float)
    span = np.where(maxs - mins == 0, 1.0, maxs - mins)
    return ((point - mins) / span).tolist()


def classify(unknown, dataset, labels, k):
    distances = [[distance(unknown, point), title] for title, point in dataset.items()]
    distances.sort()
    neighbors = distances[:k]
    num_good = sum(1 for _, title in neighbors if labels[title] == 1)
    return 1 if num_good > k / 2 else 0


def classify_np(unknown, X, y, k):
    d = np.linalg.norm(X - np.asarray(unknown, dtype=float), axis=1)
    idx = np.argsort(d)[:k]
    return int(y[idx].sum() > k / 2)


def classify_manhattan(unknown, X, y, k):
    d = np.abs(X - np.asarray(unknown, dtype=float)).sum(axis=1)
    idx = np.argsort(d)[:k]
    return int(y[idx].sum() > k / 2)
