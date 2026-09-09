"""TourRFM — RFM + K-Means on a tourism booking extract.

Short name (GitHub): TourRFM

PMS / DMO bookings → clean stayed nights → Recency / Frequency / Monetary
→ log1p + z-score → K-Means → original-unit profiles + PCA slide.

Not a yield engine, not a visa decision, not a loyalty-tier launch.
Cluster ids are arbitrary integers; a human names the bins from median
Recency / Frequency / Monetary in the original units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

DATE_FMT = "%d.%m.%Y %H:%M"
RFM_COLS = ("Recency", "Frequency", "Monetary")


def load_book(path: str = "data/tour_bookings.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def clean_bookings(
    df: pd.DataFrame,
    *,
    require_amount_positive: bool = True,
    date_fmt: str = DATE_FMT,
) -> pd.DataFrame:
    """Drop unknown guests, non-positive nights (and optionally rate).

    Quantity > 0 removes cancellations (BookingNo starting with C
    almost always have negative nights). Zero-rate comps add nothing
    to Monetary but would inflate Frequency if left in.
    """
    out = df.dropna(subset=["GuestID"]).copy()
    out = out[out["Quantity"] > 0]
    if require_amount_positive:
        out = out[out["UnitRate"] > 0]
    out["TotalSum"] = out["Quantity"] * out["UnitRate"]
    out["StayDate"] = pd.to_datetime(out["StayDate"], format=date_fmt)
    out["GuestID"] = out["GuestID"].astype(int)
    return out.reset_index(drop=True)


def snapshot_date(sales: pd.DataFrame) -> pd.Timestamp:
    return sales["StayDate"].max() + pd.Timedelta(days=1)


def build_rfm(sales: pd.DataFrame, snap: pd.Timestamp | None = None) -> pd.DataFrame:
    if snap is None:
        snap = snapshot_date(sales)
    return (
        sales.groupby("GuestID")
        .agg(
            Recency=("StayDate", lambda x: (snap - x.max()).days),
            Frequency=("BookingNo", "nunique"),
            Monetary=("TotalSum", "sum"),
        )
        .reset_index()
    )


def log_scale(rfm: pd.DataFrame, cols: Sequence[str] = RFM_COLS):
    """Return (log1p frame, scaled ndarray, mean, std). Keep original rfm."""
    log = rfm.loc[:, list(cols)].apply(np.log1p)
    mu = log.mean(axis=0).to_numpy()
    sd = log.std(axis=0, ddof=0).to_numpy()
    sd = np.where(sd == 0, 1.0, sd)
    scaled = (log.to_numpy() - mu) / sd
    return log, scaled, mu, sd


def _kmeans_pp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = len(X)
    C = np.empty((k, X.shape[1]))
    C[0] = X[rng.integers(0, n)]
    d2 = ((X - C[0]) ** 2).sum(1)
    for j in range(1, k):
        p = d2 / max(d2.sum(), 1e-12)
        C[j] = X[rng.choice(n, p=p)]
        d2 = np.minimum(d2, ((X - C[j]) ** 2).sum(1))
    return C


def kmeans_fit(X: np.ndarray, k: int = 4, n_init: int = 10, max_iter: int = 80, seed: int = 42):
    """NumPy K-Means (k-means++). Returns centres, labels, inertia."""
    rng = np.random.default_rng(seed)
    best_J, best_lab, best_C = np.inf, None, None
    n = len(X)
    for _ in range(n_init):
        C = _kmeans_pp_init(X, int(k), rng)
        lab = np.zeros(n, dtype=int)
        for _it in range(max_iter):
            dist = ((X[:, None, :] - C[None, :, :]) ** 2).sum(2)
            lab_n = dist.argmin(1)
            C_n = np.vstack(
                [X[lab_n == j].mean(0) if (lab_n == j).any() else C[j] for j in range(int(k))]
            )
            if np.allclose(C_n, C):
                C, lab = C_n, lab_n
                break
            C, lab = C_n, lab_n
        J = float(((X - C[lab]) ** 2).sum())
        if J < best_J:
            best_J, best_lab, best_C = J, lab.copy(), C.copy()
    return best_C, best_lab, best_J


def fit_kmeans(X: np.ndarray, k: int = 4, **km_kw):
    """sklearn if present, else NumPy. Returns (model_or_None, labels)."""
    seed = km_kw.get("random_state", 42)
    n_init = km_kw.get("n_init", 10)
    try:
        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=int(k), random_state=seed, n_init=n_init)
        labels = model.fit_predict(X)
        return model, labels
    except ImportError:
        _, labels, _ = kmeans_fit(X, k=k, n_init=n_init, seed=seed)
        return None, labels


def elbow_inertias(X: np.ndarray, ks: Iterable[int] = range(1, 11), **km_kw):
    seed = km_kw.get("random_state", 42)
    n_init = km_kw.get("n_init", 8)
    out = []
    ks = list(ks)
    try:
        from sklearn.cluster import KMeans

        for k in ks:
            out.append(KMeans(n_clusters=int(k), random_state=seed, n_init=n_init).fit(X).inertia_)
    except ImportError:
        for k in ks:
            out.append(kmeans_fit(X, k=int(k), n_init=n_init, seed=seed)[2])
    return ks, out


def silhouette(X: np.ndarray, labels, cap: int = 2500, seed: int = 0) -> float:
    try:
        from sklearn.metrics import silhouette_score

        return float(silhouette_score(X, labels))
    except ImportError:
        n = len(X)
        if n > cap:
            rng = np.random.default_rng(seed)
            idx = rng.choice(n, cap, replace=False)
            X, labels = X[idx], np.asarray(labels)[idx]
        else:
            labels = np.asarray(labels)
        vals = []
        for i in range(len(X)):
            same = labels == labels[i]
            if same.sum() <= 1:
                continue
            a = np.sqrt(((X[same] - X[i]) ** 2).sum(1)).sum() / (same.sum() - 1)
            bs = []
            for c in np.unique(labels):
                if c == labels[i]:
                    continue
                m = labels == c
                if m.any():
                    bs.append(np.sqrt(((X[m] - X[i]) ** 2).sum(1)).mean())
            if not bs:
                continue
            b = min(bs)
            vals.append((b - a) / max(a, b, 1e-12))
        return float(np.mean(vals)) if vals else float("nan")


def profile_clusters(rfm: pd.DataFrame, labels, cols: Sequence[str] = RFM_COLS) -> pd.DataFrame:
    tmp = rfm.copy()
    tmp["Cluster"] = np.asarray(labels)
    return tmp.groupby("Cluster")[list(cols)].agg(["mean", "median", "count"]).round(1)


def name_from_medians(rfm: pd.DataFrame, labels) -> dict:
    tmp = rfm.copy()
    tmp["Cluster"] = np.asarray(labels)
    med = tmp.groupby("Cluster")[list(RFM_COLS)].median()
    names = {}
    for c, row in med.iterrows():
        if (
            row["Frequency"] >= med["Frequency"].median()
            and row["Monetary"] >= med["Monetary"].median()
            and row["Recency"] <= med["Recency"].median()
        ):
            names[int(c)] = "Loyal / repeat guests"
        elif row["Recency"] >= med["Recency"].median() and row["Frequency"] <= med["Frequency"].median():
            names[int(c)] = "Lost / lapsed guests"
        elif row["Recency"] <= med["Recency"].median() and row["Frequency"] <= med["Frequency"].median():
            names[int(c)] = "Recent first-timers"
        else:
            names[int(c)] = "Cooling repeaters"
    return names


def pca2(X: np.ndarray):
    try:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=42)
        Z = pca.fit_transform(X)
        return Z, pca.explained_variance_ratio_, pca.components_
    except ImportError:
        Xc = X - X.mean(0)
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        Z = Xc @ Vt[:2].T
        ev = (S[:2] ** 2) / (S ** 2).sum()
        return Z, ev, Vt[:2]


def rfm_quintile_scores(rfm: pd.DataFrame) -> pd.DataFrame:
    """Classic 5-5-5 RFM scores (R inverted: recent = 5)."""
    out = rfm.copy()
    out["R"] = pd.qcut(out["Recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    out["F"] = pd.qcut(out["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    out["M"] = pd.qcut(out["Monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    out["RFM"] = out["R"] * 100 + out["F"] * 10 + out["M"]
    return out


@dataclass
class SimResult:
    k: int
    n: int
    noise: float
    inertia: float
    silhouette: float


def simulate(X: np.ndarray, k=4, n=None, noise=0.0, n_init=10, seed=0) -> SimResult:
    rng = np.random.default_rng(seed)
    n_use = len(X) if n is None else min(int(n), len(X))
    idx = rng.choice(len(X), size=n_use, replace=False)
    Xn = X[idx] + rng.normal(0.0, noise, size=(n_use, X.shape[1]))
    try:
        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=int(k), random_state=seed, n_init=n_init).fit(Xn)
        labels, J = model.labels_, float(model.inertia_)
    except ImportError:
        _, labels, J = kmeans_fit(Xn, k=int(k), n_init=n_init, seed=seed)
    sil = float("nan")
    if len(set(labels)) > 1 and n_use > k:
        sil = float(silhouette(Xn, labels, cap=2000, seed=seed))
    return SimResult(k=int(k), n=n_use, noise=float(noise), inertia=J, silhouette=sil)


if __name__ == "__main__":
    sales = clean_bookings(load_book())
    rfm = build_rfm(sales)
    _, X, _, _ = log_scale(rfm)
    _, labels = fit_kmeans(X, k=4)
    print("guests", len(rfm), "spend", round(rfm["Monetary"].sum(), 2))
    print(profile_clusters(rfm, labels))
    print("names", name_from_medians(rfm, labels))
