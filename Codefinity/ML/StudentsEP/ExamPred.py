"""ExamPred — parse a student roster and estimate exam_score.

Teaching roster only. Not a grading engine and not an admissions model.
"""
from __future__ import annotations
import argparse
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

HOUR_COLS = ["study_hours", "prep_hours", "sleep_hours"]


def load_roster(path):
    return pd.read_csv(path)


def parse_units(df):
    out = df.copy()
    for col in HOUR_COLS:
        out[col] = out[col].astype(str).str.replace(" hours", "", regex=False).astype(float)
    out["prior_gpa"] = out["prior_gpa"].astype(str).str.replace(" GPA", "", regex=False).astype(float)
    out["study_load"] = out["study_hours"] + out["prep_hours"]
    return out


def clean_artifacts(df):
    out = df.copy()
    out["attendance"] = out["attendance"].astype(str).str.replace("%", "", regex=False).astype(float) / 100
    out["effort_stars"] = (
        out["effort_stars"].astype(str)
        .str.replace(" stars", "", regex=False).str.replace(" star", "", regex=False).astype(int)
    )
    out["cohort_n"] = (
        out["cohort"].astype(str)
        .str.replace("Not Available", "0", regex=False).str.replace(" CY", "", regex=False).astype(int)
    )
    out = out.drop(columns=["cohort"])
    out["study_partners"] = out["study_partners"].astype(str).str.extract(r"(\d+)", expand=False).astype(int)
    out["term_weeks"] = out["term_weeks"].astype(str).str.replace("-week", "", regex=False).astype(int)
    out["credits"] = out["credits"].astype(str).str.replace(" cr", "", regex=False).astype(int)
    return out


def encode_split(df, random_state=42):
    y = df["exam_score"]
    X = df.drop(columns=["exam_score"])
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


def main(path="data/student_exams.csv"):
    df = clean_artifacts(parse_units(load_roster(path)))
    Xtr, Xte, ytr, yte = encode_split(df)
    model = RandomForestRegressor(random_state=42).fit(Xtr, ytr)
    pred = model.predict(Xte)
    print(f"n={len(df)} train={len(ytr)} test={len(yte)}")
    print(f"MAE {mean_absolute_error(yte, pred):.3f}")
    print(f"R2  {r2_score(yte, pred):.4f}")
    print(f"baseline MAE {mean_absolute_error(yte, np.full_like(yte, ytr.mean(), dtype=float)):.3f}")
    print(pd.Series(model.feature_importances_, index=Xtr.columns).sort_values(ascending=False).head(8).round(4).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/student_exams.csv")
    main(p.parse_args().data)
