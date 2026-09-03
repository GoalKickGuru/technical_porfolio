"""PCATel — MAGIC Gamma Telescope PCA helpers.

Short name (GitHub): PCATel
Not an IACT trigger and not a discovery claim.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple


HILLAS = [
    "fLength", "fWidth", "fSize", "fConc", "fConc1",
    "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist",
]


def load_telescope(path: str = "data/telescope_data.csv") -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path, index_col=0).dropna()
    classes = df["class"]
    data_matrix = df.drop(columns=["class"])
    return data_matrix, classes


def standardize(X: pd.DataFrame, ddof: int = 1):
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=ddof).replace(0, 1.0)
    return (X - mu) / sd, mu, sd


def eigen_from_corr(X: pd.DataFrame):
    R = X.corr().values
    evals, evecs = np.linalg.eig(R)
    evals = np.real(evals)
    evecs = np.real(evecs)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    percents = 100.0 * evals / evals.sum()
    return evals, evecs, percents, np.cumsum(percents)


def project(X_std: np.ndarray, evecs: np.ndarray, k: int) -> np.ndarray:
    return X_std @ evecs[:, :k]


def reconstruction_mse(X_std: np.ndarray, evecs: np.ndarray, k: int) -> float:
    Z = project(X_std, evecs, k)
    Xhat = Z @ evecs[:, :k].T
    return float(np.mean((X_std - Xhat) ** 2))
