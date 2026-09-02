"""WineReg — Predict red-wine quality with regularized logistic regression.

UCI red-wine table, binary target quality > 5. Lesson flow:
  scale → 80/20 split → unregularized LR → F1 → default L2 → coarse C
  → GridSearchCV → hold-out validate → L1 LogisticRegressionCV (density → 0).

Short name: WineReg
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score


def load_wine(path: str = "data/wine_quality.csv"):
    df = pd.read_csv(path)
    y = df["quality"]
    features = df.drop(columns=["quality"])
    return df, features, y


def scale_features(features: pd.DataFrame):
    scaler = StandardScaler().fit(features)
    X = scaler.transform(features)
    return X, scaler


def split(X, y, test_size=0.2, random_state=99):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def fit_unregularized(X_train, y_train):
    # sklearn ≥1.8: penalty=None (or C=np.inf). Lesson used penalty='none'.
    clf = LogisticRegression(penalty=None, max_iter=2000)
    clf.fit(X_train, y_train)
    return clf


def fit_l2(X_train, y_train, C=1.0):
    clf = LogisticRegression(C=C, penalty="l2", max_iter=2000)
    clf.fit(X_train, y_train)
    return clf


def report_f1(model, X_train, y_train, X_test, y_test, tag=""):
    tr = f1_score(y_train, model.predict(X_train))
    te = f1_score(y_test, model.predict(X_test))
    print(f"{tag} train F1={tr:.4f}  test F1={te:.4f}")
    return tr, te


def coarse_C_search(X_train, y_train, X_test, y_test, C_array=None):
    if C_array is None:
        C_array = [0.0001, 0.001, 0.01, 0.1, 1]
    train_scores, test_scores = [], []
    for C in C_array:
        clf = fit_l2(X_train, y_train, C=C)
        train_scores.append(f1_score(y_train, clf.predict(X_train)))
        test_scores.append(f1_score(y_test, clf.predict(X_test)))
    return C_array, train_scores, test_scores


def fine_ridge_search(X_train, y_train, n_C=100):
    C_array = np.logspace(-4, -2, n_C)
    gs = GridSearchCV(
        LogisticRegression(penalty="l2", max_iter=2000),
        param_grid={"C": C_array},
        scoring="f1",
        cv=5,
    )
    gs.fit(X_train, y_train)
    return gs


def fit_l1_cv(X, y):
    C_array = np.logspace(-2, 2, 100)
    clf = LogisticRegressionCV(
        Cs=C_array,
        cv=5,
        penalty="l1",
        scoring="f1",
        solver="liblinear",
        max_iter=2000,
    )
    clf.fit(X, y)
    return clf


def plot_coefs(coef_row, names, title, path=None):
    s = pd.Series(np.ravel(coef_row), index=names).sort_values()
    ax = s.plot(kind="barh", title=title)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.show()
    plt.clf()
    return s


if __name__ == "__main__":
    df, features, y = load_wine()
    print("rows", len(df), "good rate", float(y.mean()))
    X, _ = scale_features(features)
    X_train, X_test, y_train, y_test = split(X, y)

    clf0 = fit_unregularized(X_train, y_train)
    report_f1(clf0, X_train, y_train, X_test, y_test, tag="unreg")

    clf1 = fit_l2(X_train, y_train, C=1)
    report_f1(clf1, X_train, y_train, X_test, y_test, tag="L2 C=1")

    C_array, tr, te = coarse_C_search(X_train, y_train, X_test, y_test)
    print("coarse C", C_array)
    print("train", [round(v, 4) for v in tr])
    print("test ", [round(v, 4) for v in te])

    gs = fine_ridge_search(X_train, y_train)
    print("best C", gs.best_params_, "cv F1", gs.best_score_)
    best = fit_l2(X_train, y_train, C=gs.best_params_["C"])
    print("hold-out F1", f1_score(y_test, best.predict(X_test)),
          "AUC", roc_auc_score(y_test, best.predict_proba(X_test)[:, 1]),
          "acc", accuracy_score(y_test, best.predict(X_test)))

    l1 = fit_l1_cv(X, y)
    print("L1 best C", l1.C_)
    print(pd.Series(l1.coef_.ravel(), index=features.columns).round(4))
