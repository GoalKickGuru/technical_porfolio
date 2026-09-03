"""LaptopPrice — feature engineering + Random Forest for listed laptop prices.

Teaching catalog only. Not a store-pricing engine.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


UNIT_COLS = ["ram_gb", "ssd", "hdd", "graphic_card_gb"]


def load_catalog(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def strip_units(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in UNIT_COLS:
        out[col] = (
            out[col].astype(str).str.replace(" GB", "", regex=False).astype(int)
        )
    out["total_storage"] = out["ssd"] + out["hdd"]
    return out.drop(columns=["ssd", "hdd"])


def clean_artifacts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["os_bit"] = (
        out["os_bit"].astype(str).str.replace("-bit", "", regex=False).astype(int)
    )
    out["rating"] = (
        out["rating"]
        .astype(str)
        .str.replace(" stars", "", regex=False)
        .str.replace(" star", "", regex=False)
        .astype(int)
    )
    out["processor_gnrtn"] = (
        out["processor_gnrtn"]
        .astype(str)
        .str.replace("Not Available", "0", regex=False)
        .str.replace("th", "", regex=False)
        .astype(int)
    )
    return out


def encode_split(df: pd.DataFrame, random_state: int = 42):
    y = df["Price"]
    X = df.drop(columns=["Price"])
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    cat_cols = Xtr.select_dtypes(include=["object"]).columns.tolist()
    low = [c for c in cat_cols if df[c].nunique() < 5]
    high = [c for c in cat_cols if df[c].nunique() >= 5]
    Xtr_enc, Xte_enc = Xtr.copy(), Xte.copy()
    for col in high:
        means = pd.concat([Xtr_enc[col], ytr], axis=1).groupby(col)[ytr.name].mean()
        Xtr_enc[col] = Xtr_enc[col].map(means)
        Xte_enc[col] = Xte_enc[col].map(means).fillna(ytr.mean())
    for col in low:
        dtr = pd.get_dummies(Xtr_enc[col], prefix=col, drop_first=True)
        dte = pd.get_dummies(Xte_enc[col], prefix=col, drop_first=True)
        dte = dte.reindex(columns=dtr.columns, fill_value=0)
        Xtr_enc = pd.concat([Xtr_enc.drop(columns=[col]), dtr], axis=1)
        Xte_enc = pd.concat([Xte_enc.drop(columns=[col]), dte], axis=1)
    return Xtr_enc, Xte_enc, ytr, yte


def fit_forest(Xtr, ytr, random_state: int = 42) -> RandomForestRegressor:
    model = RandomForestRegressor(random_state=random_state)
    model.fit(Xtr, ytr)
    return model


def main(path: str = "data/laptop_price.csv") -> None:
    df = clean_artifacts(strip_units(load_catalog(path)))
    Xtr, Xte, ytr, yte = encode_split(df)
    model = fit_forest(Xtr, ytr)
    pred = model.predict(Xte)
    print(f"n={len(df)}  train={len(ytr)}  test={len(yte)}")
    print(f"MAE {mean_absolute_error(yte, pred):,.2f}")
    print(f"R2  {r2_score(yte, pred):.4f}")
    print(f"baseline MAE {mean_absolute_error(yte, np.full_like(yte, ytr.mean(), dtype=float)):,.2f}")
    imp = pd.Series(model.feature_importances_, index=Xtr.columns).sort_values(ascending=False)
    print(imp.head(8).round(4).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/laptop_price.csv")
    main(parser.parse_args().data)
