"""WrapFS_Obes — wrapper feature selection helpers for the obesity survey.

Short GitHub name: WrapFS_Obes
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE, SequentialFeatureSelector as SkSFS
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split


def load_obesity(path="data/obesity.csv"):
    df = pd.read_csv(path)
    X = df.drop(columns=["NObeyesdad"])
    y = df["NObeyesdad"]
    return df, X, y


def make_lr(max_iter=1000, C=1.0, penalty="l2"):
    kw = dict(max_iter=max_iter, C=C)
    if penalty == "l1":
        kw.update(penalty="l1", solver="liblinear")
    return LogisticRegression(**kw)


def baseline_acc(X, y, max_iter=1000):
    lr = make_lr(max_iter=max_iter)
    return float(lr.fit(X, y).score(X, y))


def rfe_names(X, y, n=6, max_iter=1000):
    feats = list(X.columns)
    Xs = StandardScaler().fit_transform(X)
    rfe = RFE(estimator=make_lr(max_iter=max_iter), n_features_to_select=n)
    rfe.fit(Xs, y)
    kept = [f for f, s in zip(feats, rfe.support_) if s]
    return kept, float(rfe.score(Xs, y)), rfe.ranking_


def sklearn_sfs_names(X, y, k=6, direction="forward"):
    sfs = SkSFS(
        make_lr(),
        n_features_to_select=k,
        direction=direction,
        scoring="accuracy",
        cv=None,
    )
    sfs.fit(X, y)
    return list(X.columns[sfs.get_support()])


def l1_kept(X, y, C=0.08):
    Xs = StandardScaler().fit_transform(X)
    clf = make_lr(C=C, penalty="l1").fit(Xs, y)
    return [f for f, c in zip(X.columns, clf.coef_.ravel()) if abs(c) > 1e-8], float(clf.score(Xs, y))


def mi_topk(X, y, k=6, seed=0):
    scores = mutual_info_classif(X, y, random_state=seed)
    order = np.argsort(scores)[::-1][:k]
    return list(X.columns[order]), scores


def holdout_acc(X, y, cols=None, test_size=0.3, seed=0):
    use = X if cols is None else X[list(cols)]
    Xtr, Xte, ytr, yte = train_test_split(
        use, y, test_size=test_size, random_state=seed, stratify=y
    )
    return float(make_lr().fit(Xtr, ytr).score(Xte, yte))
