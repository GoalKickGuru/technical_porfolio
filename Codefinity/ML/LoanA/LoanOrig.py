"""LoanOrig — parse a draft application book and estimate the offered principal.

Teaching book only. Not a credit decision and not lending advice.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

MONEY = ["annual_income_usd", "other_debt_usd", "requested_usd", "down_payment_usd", "reserves_usd"]


def load_book(path):
    return pd.read_csv(path)


def parse_money(df):
    out = df.copy()
    for col in MONEY:
        out[col] = (
            out[col].astype(str).str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False).astype(float)
        )
    out["net_capacity"] = out["annual_income_usd"] - out["other_debt_usd"]
    return out


def clean_artifacts(df):
    out = df.copy()
    out["note_rate"] = out["note_rate"].astype(str).str.replace("%", "", regex=False).astype(float) / 100
    out["bureau_stars"] = (
        out["bureau_stars"].astype(str)
        .str.replace(" stars", "", regex=False).str.replace(" star", "", regex=False).astype(int)
    )
    out["orig_year_n"] = (
        out["orig_vintage"].astype(str)
        .str.replace("Not Available", "0", regex=False).str.replace(" VY", "", regex=False).astype(int)
    )
    out = out.drop(columns=["orig_vintage"])
    out["borrowers"] = out["borrowers"].astype(str).str.extract(r"(\d+)", expand=False).astype(int)
    out["term_months"] = out["term_months"].astype(str).str.replace("-month", "", regex=False).astype(int)
    return out


def encode_split(df, random_state=42):
    y = df["offer_amount"]
    X = df.drop(columns=["offer_amount"])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=random_state)
    cat = Xtr.select_dtypes(include=["object"]).columns.tolist()
    low = [c for c in cat if df[c].nunique() < 5]
    high = [c for c in cat if df[c].nunique() >= 5]
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


def main(path="data/loan_applications.csv"):
    df = clean_artifacts(parse_money(load_book(path)))
    Xtr, Xte, ytr, yte = encode_split(df)
    model = RandomForestRegressor(random_state=42).fit(Xtr, ytr)
    pred = model.predict(Xte)
    print(f"n={len(df)} train={len(ytr)} test={len(yte)}")
    print(f"MAE {mean_absolute_error(yte, pred):,.2f}")
    print(f"R2  {r2_score(yte, pred):.4f}")
    print(f"baseline MAE {mean_absolute_error(yte, np.full_like(yte, ytr.mean(), dtype=float)):,.2f}")
    print(pd.Series(model.feature_importances_, index=Xtr.columns).sort_values(ascending=False).head(8).round(4).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/loan_applications.csv")
    main(p.parse_args().data)
