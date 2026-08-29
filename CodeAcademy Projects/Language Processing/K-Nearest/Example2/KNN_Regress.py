"""KNN_Regress helpers — from-scratch uniform / weighted KNN regression."""
import numpy as np


def distance(a, b):
    squared = 0.0
    for i in range(len(a)):
        squared += (a[i] - b[i]) ** 2
    return squared ** 0.5


def distance_np(a, b):
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


def min_max_normalize(lst):
    lo, hi = min(lst), max(lst)
    span = hi - lo if hi != lo else 1.0
    return [(x - lo) / span for x in lst]


def predict(unknown, dataset, movie_ratings, k):
    """Uniform average of the k nearest ratings. dataset: {title: features}."""
    distances = []
    for title, movie in dataset.items():
        distances.append([distance(movie, unknown), title])
    distances.sort()
    neighbors = distances[:k]
    total = 0.0
    for _, title in neighbors:
        total += movie_ratings[title]
    return total / len(neighbors)


def predict_weighted(unknown, dataset, movie_ratings, k, eps=1e-12):
    """Inverse-distance weighted average of the k nearest ratings."""
    distances = []
    for title, movie in dataset.items():
        distances.append([distance(movie, unknown), title])
    distances.sort()
    neighbors = distances[:k]
    numerator = 0.0
    denominator = 0.0
    for dist, title in neighbors:
        d = max(dist, eps)
        numerator += movie_ratings[title] / d
        denominator += 1.0 / d
    return numerator / denominator
