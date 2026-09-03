"""TaxCreate — parse a draft-return extract and estimate the assessment.

Teaching catalog only. Not tax advice and not a filing engine.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

MONEY_COLS = ["wages_usd", "business_usd", "capgains_usd", "deduction_usd", "credits_usd"]


def load_extract(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def parse_money(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in MONEY_COLS:
        out[col] = (
            out[col].astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .astype(float)
        )
    out["gross_income"] = out["wages_usd"] + out["business_usd"] + out["capgains_usd"]
    return out


def clean_artifacts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["state_rate"] = (
        out["state_rate"].astype(str).str.replace("%", "", regex=False).astype(float) / 100
    )
    out["compliance_stars"] = (
        out["compliance_stars"].astype(str)
        .str.replace(" stars", "", regex=False)
        .str.replace(" star", "", regex=False)
        .astype(int)
    )
    out["tax_year_n"] = (
        out["tax_year"].astype(str)
        .str.replace("Not Available", "0", regex=False)
        .str.replace(" TY", "", regex=False)
        .astype(int)
    )
    out = out.drop(columns=["tax_year"])
    out["dependents"] = out["dependents"].astype(str).str.extract(r"(\d+)", expand=False).astype(int)
    out["return_pages"] = (
        out["return_pages"].astype(str).str.replace("-page", "", regex=False).astype(int)
    )
    return out


def encode_split(df: pd.DataFrame, random_state: int = 42):
    y = df["tax_due"]
    X = df.drop(columns=["tax_due"])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=random_state)
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


def main(path: str = "data/tax_returns.csv") -> None:
    df = clean_artifacts(parse_money(load_extract(path)))
    Xtr, Xte, ytr, yte = encode_split(df)
    model = RandomForestRegressor(random_state=42)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    print(f"n={len(df)}  train={len(ytr)}  test={len(yte)}")
    print(f"MAE {mean_absolute_error(yte, pred):,.2f}")
    print(f"R2  {r2_score(yte, pred):.4f}")
    print(
        "baseline MAE "
        f"{mean_absolute_error(yte, np.full_like(yte, ytr.mean(), dtype=float)):,.2f}"
    )
    imp = pd.Series(model.feature_importances_, index=Xtr.columns).sort_values(ascending=False)
    print(imp.head(8).round(4).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/tax_returns.csv")
    main(parser.parse_args().data)
