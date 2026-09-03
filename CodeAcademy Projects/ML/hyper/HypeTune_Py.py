"""HypeTune_Py — raisins feature importance + GridSearchCV + RandomizedSearchCV.

Teaching helper for the extended project. Not a packing-line classifier.
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

TREE_GRID = {"max_depth": [3, 5, 7], "min_samples_split": [2, 3, 4]}
LR_DIST = {"penalty": ["l1", "l2"], "C": uniform(loc=0, scale=100)}


def load_raisins(path: str = "data/Raisin_Dataset.csv"):
    df = pd.read_csv(path)
    X = df.drop(columns="Class")
    y = df["Class"]
    return df, X, y


def split(X, y, random_state: int = 19):
    return train_test_split(X, y, random_state=random_state)


def gini_table(estimator, columns) -> pd.Series:
    return pd.Series(estimator.feature_importances_, index=columns).sort_values(
        ascending=False
    )


def tree_grid(X_train, y_train, random_state=None, cv: int = 5) -> GridSearchCV:
    est = DecisionTreeClassifier(random_state=random_state)
    grid = GridSearchCV(est, TREE_GRID, cv=cv)
    grid.fit(X_train, y_train)
    return grid


def lr_random(
    X_train, y_train, n_iter: int = 8, random_state: int = 19, cv: int = 5
) -> RandomizedSearchCV:
    est = LogisticRegression(solver="liblinear", max_iter=2000)
    clf = RandomizedSearchCV(
        est, LR_DIST, n_iter=n_iter, random_state=random_state, cv=cv
    )
    clf.fit(X_train, y_train)
    return clf


def scaled_lr_grid(X_train, y_train, cv: int = 5) -> GridSearchCV:
    pipe = Pipeline(
        [
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(solver="liblinear", max_iter=2000)),
        ]
    )
    grid = GridSearchCV(
        pipe,
        {"lr__penalty": ["l1", "l2"], "lr__C": [0.01, 0.1, 1, 10, 100]},
        cv=cv,
    )
    grid.fit(X_train, y_train)
    return grid


def forest(X_train, y_train, n_estimators: int = 200, random_state: int = 19):
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)
    return rf


def cv_table(search, score_name: str = "Score") -> pd.DataFrame:
    return pd.concat(
        [
            pd.DataFrame(search.cv_results_["params"]),
            pd.DataFrame(search.cv_results_["mean_test_score"], columns=[score_name]),
        ],
        axis=1,
    ).sort_values(score_name, ascending=False)


if __name__ == "__main__":
    df, X, y = load_raisins()
    X_train, X_test, y_train, y_test = split(X, y)
    grid = tree_grid(X_train, y_train, random_state=19)
    clf = lr_random(X_train, y_train)
    rf = forest(X_train, y_train)
    print("n =", len(df), "features =", list(X.columns))
    print("DT grid", grid.best_params_, "CV", round(grid.best_score_, 4),
          "test", round(grid.score(X_test, y_test), 4))
    print("LR rand", clf.best_params_, "CV", round(clf.best_score_, 4),
          "test", round(clf.score(X_test, y_test), 4))
    print("RF Gini\n", gini_table(rf, X.columns).round(3))
