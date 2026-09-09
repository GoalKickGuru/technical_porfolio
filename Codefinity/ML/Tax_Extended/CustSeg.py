"""CustSeg — RFM + K-Means customer segmentation helpers.

Short name (GitHub): CustSeg

Online Retail line items → clean sales → Recency / Frequency / Monetary
→ log1p + StandardScaler → K-Means → original-unit profiles + PCA slide.

Not a loyalty-tier engine and not a next-purchase model. Cluster ids are
arbitrary integers; a human names the bins from median Recency / Frequency /
Monetary in the original units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

DATE_FMT = "%d.%m.%Y %H:%M"
RFM_COLS = ("Recency", "Frequency", "Monetary")


def load_retail(path: str = "data/online_retail.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def clean_sales(
    df: pd.DataFrame,
    *,
    require_price_positive: bool = True,
    date_fmt: str = DATE_FMT,
) -> pd.DataFrame:
    """Drop unknown customers, non-positive qty (and optionally price).

    Quantity > 0 removes returns/cancellations (InvoiceNo starting with C
    almost always have negative qty). Zero-qty rows would inflate Frequency
    without adding Monetary.
    """
    out = df.dropna(subset=["CustomerID"]).copy()
    out = out[out["Quantity"] > 0]
    if require_price_positive:
        out = out[out["UnitPrice"] > 0]
    out["TotalSum"] = out["Quantity"] * out["UnitPrice"]
    out["InvoiceDate"] = pd.to_datetime(out["InvoiceDate"], format=date_fmt)
    out["CustomerID"] = out["CustomerID"].astype(int)
    return out.reset_index(drop=True)


def snapshot_date(sales: pd.DataFrame) -> pd.Timestamp:
    return sales["InvoiceDate"].max() + pd.Timedelta(days=1)


def build_rfm(sales: pd.DataFrame, snap: pd.Timestamp | None = None) -> pd.DataFrame:
    if snap is None:
        snap = snapshot_date(sales)
    rfm = (
        sales.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (snap - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalSum", "sum"),
        )
        .reset_index()
    )
    return rfm


def log_scale(rfm: pd.DataFrame, cols: Sequence[str] = RFM_COLS):
    """Return (log1p frame, scaled ndarray, mean, std).

    Keep the original rfm intact — you will name clusters from raw units.
    """
    log = rfm.loc[:, list(cols)].apply(np.log1p)
    mu = log.mean(axis=0).to_numpy()
    sd = log.std(axis=0, ddof=0).to_numpy()
    sd = np.where(sd == 0, 1.0, sd)
    scaled = (log.to_numpy() - mu) / sd
    return log, scaled, mu, sd


def elbow_inertias(X: np.ndarray, ks: Iterable[int] = range(1, 11), **km_kw):
    from sklearn.cluster import KMeans

    kw = dict(random_state=42, n_init=10)
    kw.update(km_kw)
    out = []
    for k in ks:
        out.append(KMeans(n_clusters=int(k), **kw).fit(X).inertia_)
    return list(ks), out


def fit_kmeans(X: np.ndarray, k: int = 4, **km_kw):
    from sklearn.cluster import KMeans

    kw = dict(random_state=42, n_init=10)
    kw.update(km_kw)
    model = KMeans(n_clusters=int(k), **kw)
    labels = model.fit_predict(X)
    return model, labels


def profile_clusters(rfm: pd.DataFrame, labels, cols: Sequence[str] = RFM_COLS) -> pd.DataFrame:
    tmp = rfm.copy()
    tmp["Cluster"] = np.asarray(labels)
    return (
        tmp.groupby("Cluster")[list(cols)]
        .agg(["mean", "median", "count"])
        .round(1)
    )


def name_from_medians(rfm: pd.DataFrame, labels) -> dict:
    """Heuristic labels for the k=4 solution on this extract.

    Champions   : lowest Recency, highest Frequency & Monetary
    Lost        : highest Recency, lowest Frequency & Monetary
    Recent occ. : low Recency, low Frequency
    Slipping    : mid/high Recency, mid Frequency
    """
    tmp = rfm.copy()
    tmp["Cluster"] = np.asarray(labels)
    med = tmp.groupby("Cluster")[list(RFM_COLS)].median()
    names = {}
    for c, row in med.iterrows():
        if row["Frequency"] >= med["Frequency"].median() and row["Monetary"] >= med["Monetary"].median() and row["Recency"] <= med["Recency"].median():
            names[int(c)] = "Champions"
        elif row["Recency"] >= med["Recency"].median() and row["Frequency"] <= med["Frequency"].median():
            names[int(c)] = "Lost / hibernating"
        elif row["Recency"] <= med["Recency"].median() and row["Frequency"] <= med["Frequency"].median():
            names[int(c)] = "Recent occasional"
        else:
            names[int(c)] = "Slipping mid-value"
    return names


def pca2(X: np.ndarray):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X)
    return Z, pca.explained_variance_ratio_, pca.components_


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
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(seed)
    n = len(X) if n is None else min(int(n), len(X))
    idx = rng.choice(len(X), size=n, replace=False)
    Xn = X[idx] + rng.normal(0.0, noise, size=(n, X.shape[1]))
    model = KMeans(n_clusters=int(k), random_state=seed, n_init=n_init).fit(Xn)
    sil = float("nan")
    if len(set(model.labels_)) > 1 and n > k:
        sil = float(silhouette_score(Xn, model.labels_))
    return SimResult(k=int(k), n=n, noise=float(noise), inertia=float(model.inertia_), silhouette=sil)


if __name__ == "__main__":
    sales = clean_sales(load_retail())
    rfm = build_rfm(sales)
    _, X, _, _ = log_scale(rfm)
    model, labels = fit_kmeans(X, k=4)
    print("customers", len(rfm), "revenue", round(rfm["Monetary"].sum(), 2))
    print(profile_clusters(rfm, labels))
    print("names", name_from_medians(rfm, labels))
