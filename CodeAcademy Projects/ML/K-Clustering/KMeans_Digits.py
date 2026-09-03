"""KMeans_Digits — from-scratch k-means + digit / iris helpers.

Drop-in stand-in for sklearn.cluster.KMeans when sklearn is not installed.
Swap the import later:

    from sklearn.cluster import KMeans
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "KMeans",
    "assign",
    "inertia",
    "map_clusters_to_truth",
    "adjusted_rand",
    "elbow_curve",
]


def assign(X, centroids):
    d2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return d2.argmin(axis=1)


def inertia(X, labels, centroids):
    return float(((X - centroids[labels]) ** 2).sum())


def _random_init(X, k, rng):
    idx = rng.choice(len(X), size=k, replace=False)
    return X[idx].copy()


def _kmeans_pp_init(X, k, rng):
    n = len(X)
    cents = np.empty((k, X.shape[1]), dtype=float)
    cents[0] = X[rng.integers(n)]
    closest = ((X - cents[0]) ** 2).sum(axis=1)
    for j in range(1, k):
        p = closest / closest.sum()
        cents[j] = X[rng.choice(n, p=p)]
        d2 = ((X - cents[j]) ** 2).sum(axis=1)
        closest = np.minimum(closest, d2)
    return cents


def _update(X, labels, k, rng):
    cents = np.zeros((k, X.shape[1]))
    for j in range(k):
        mask = labels == j
        if mask.any():
            cents[j] = X[mask].mean(axis=0)
        else:
            cents[j] = X[rng.integers(len(X))]
    return cents


class KMeans:
    """Minimal sklearn-shaped k-means.

    Parameters
    ----------
    n_clusters : int
    init : {'random', 'k-means++'}
    n_init : int
        Number of random restarts; keep the run with lowest inertia.
    max_iter, tol, random_state : as in sklearn
    """

    def __init__(
        self,
        n_clusters=8,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=1e-4,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        best_in = np.inf
        best = None
        parent = np.random.default_rng(self.random_state)
        for r in range(max(1, self.n_init)):
            seed = int(parent.integers(0, 2**31 - 1))
            rng = np.random.default_rng(seed)
            if self.init == "k-means++":
                cents = _kmeans_pp_init(X, self.n_clusters, rng)
            else:
                cents = _random_init(X, self.n_clusters, rng)
            n_iter = 0
            for n_iter in range(1, self.max_iter + 1):
                labels = assign(X, cents)
                new_c = _update(X, labels, self.n_clusters, rng)
                if np.linalg.norm(new_c - cents) < self.tol:
                    cents = new_c
                    break
                cents = new_c
            labels = assign(X, cents)
            inn = inertia(X, labels, cents)
            if inn < best_in:
                best_in = inn
                best = (cents, labels, n_iter)
        self.cluster_centers_, self.labels_, self.n_iter_ = best
        self.inertia_ = best_in
        return self

    def predict(self, X):
        if not hasattr(self, "cluster_centers_"):
            raise RuntimeError("Not fitted. Call fit() before predict().")
        return assign(np.asarray(X, dtype=float), self.cluster_centers_)

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_


def map_clusters_to_truth(labels, y, k=None):
    k = int(np.max(labels)) + 1 if k is None else k
    mapping, mapped = {}, np.full_like(labels, -1)
    correct = 0
    y = np.asarray(y)
    for j in range(k):
        mask = labels == j
        if not mask.any():
            mapping[j] = -1
            continue
        vals, cnts = np.unique(y[mask], return_counts=True)
        lab = int(vals[cnts.argmax()])
        mapping[j] = lab
        mapped[mask] = lab
        correct += int(cnts.max())
    return mapped, mapping, correct / len(y)


def adjusted_rand(y_true, y_pred):
    yt, yp = np.asarray(y_true, int), np.asarray(y_pred, int)
    kt, kp = yt.max() + 1, yp.max() + 1
    C = np.zeros((kt, kp), dtype=int)
    for a, b in zip(yt, yp):
        C[a, b] += 1
    n = C.sum()
    sum_c = 0.5 * (C * (C - 1)).sum()
    sum_r = 0.5 * (C.sum(1) * (C.sum(1) - 1)).sum()
    sum_k = 0.5 * (C.sum(0) * (C.sum(0) - 1)).sum()
    n_comb = 0.5 * n * (n - 1)
    expected = (sum_r * sum_k) / n_comb if n_comb else 0.0
    max_index = 0.5 * (sum_r + sum_k)
    if max_index == expected:
        return 1.0
    return float((sum_c - expected) / (max_index - expected))


def elbow_curve(X, ks, random_state=0, n_init=3):
    inertias = []
    for k in ks:
        model = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        model.fit(X)
        inertias.append(model.inertia_)
    return list(ks), inertias


if __name__ == "__main__":
    import pandas as pd

    iris = pd.read_csv("data/iris.csv")
    X = iris.drop(columns=["species"]).to_numpy()
    y = iris["species"].to_numpy().astype(int)
    m = KMeans(n_clusters=3, random_state=0, n_init=10).fit(X)
    _, mapping, acc = map_clusters_to_truth(m.labels_, y, 3)
    print("iris purity", round(acc, 4), "ARI", round(adjusted_rand(y, m.labels_), 4))
    print("inertia", round(m.inertia_, 2), "map", mapping)
