"""
PTEdu.py — helpers for the education-sector PyTorch lab (roster → outcome band).

Short name (GitHub): PTEdu
Adaptation of PTIntro (Iris tensors + MLP) to a student-outcome table.
Not a grading engine, early-alert product, or placement tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


OUTCOME_ORDER = ("support", "on_track", "honors")
FEATURE_COLS = ("attendance_pct", "study_hours", "prior_gpa", "assignment_avg")


def load_roster(path: str = "data/students_outcomes.csv"):
    """Return float32 X (n, 4), int64 y in {0,1,2}, raw frame."""
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"roster missing columns {missing}")
    X = df.loc[:, list(FEATURE_COLS)].to_numpy(dtype=np.float32)
    y = df["outcome"].map({s: i for i, s in enumerate(OUTCOME_ORDER)}).to_numpy(dtype=np.int64)
    if np.isnan(y).any():
        unknown = sorted(set(df["outcome"]) - set(OUTCOME_ORDER))
        raise ValueError(f"unexpected outcome labels: {unknown}")
    return X, y, df


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
):
    """Stratified (default) shuffle split. No sklearn dependency."""
    rng = np.random.RandomState(random_state)
    n = len(X)
    if not stratify:
        idx = rng.permutation(n)
        n_te = int(round(n * test_size))
        te, tr = idx[:n_te], idx[n_te:]
        return X[tr], X[te], y[tr], y[te]
    tr_parts, te_parts = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_te = int(round(len(idx) * test_size))
        te_parts.append(idx[:n_te])
        tr_parts.append(idx[n_te:])
    tr = np.concatenate(tr_parts)
    te = np.concatenate(te_parts)
    rng.shuffle(tr)
    rng.shuffle(te)
    return X[tr], X[te], y[tr], y[te]


def scale_train_test(X_train, X_test):
    """Train-only mean/sd. Education features live on incompatible units."""
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0) + 1e-6
    return (X_train - mu) / sd, (X_test - mu) / sd, mu, sd


def to_tensors(X_train, X_test, y_train, y_test):
    return (
        torch.as_tensor(X_train, dtype=torch.float32),
        torch.as_tensor(X_test, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.long),
        torch.as_tensor(y_test, dtype=torch.long),
    )


class OutcomeNet(nn.Module):
    """4 → hidden (ReLU) → 3 logits. Softmax lives inside CrossEntropyLoss."""

    def __init__(self, input_size: int = 4, hidden_size: int = 16, output_size: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


def make_sequential(input_size: int = 4, hidden_size: int = 16, output_size: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )


@dataclass
class TrainResult:
    model: nn.Module
    losses: list
    test_acc: float
    pred: np.ndarray
    y_test: np.ndarray


def train_roster(
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    hidden_size: int = 16,
    lr: float = 0.01,
    epochs: int = 150,
    seed: int = 42,
    optimizer: str = "adam",
    label_noise: float = 0.0,
    scale: bool = True,
    model: Optional[nn.Module] = None,
) -> TrainResult:
    """Full-batch train. scale=True applies train-only z-scores (recommended)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    y_use = np.asarray(y_train).copy()
    if label_noise > 0:
        n_flip = int(label_noise * len(y_use))
        idx = np.random.choice(len(y_use), n_flip, replace=False)
        y_use[idx] = np.random.randint(0, int(y_use.max()) + 1, size=n_flip)
    Xs = np.asarray(X_train, dtype=np.float32)
    Xse = None if X_test is None else np.asarray(X_test, dtype=np.float32)
    if scale:
        Xs, Xse_s, _, _ = scale_train_test(Xs, Xs if Xse is None else Xse)
        if X_test is not None:
            Xse = Xse_s
    Xt = torch.as_tensor(Xs, dtype=torch.float32)
    yt = torch.as_tensor(y_use, dtype=torch.long)
    if model is None:
        n_out = int(np.max(y_train)) + 1
        model = OutcomeNet(Xt.shape[1], hidden_size, n_out)
    if optimizer.lower() == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    losses = []
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = crit(model(Xt), yt)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    pred = np.array([])
    acc = float("nan")
    y_te = np.asarray(y_test) if y_test is not None else np.array([])
    if Xse is not None and y_test is not None:
        model.eval()
        with torch.no_grad():
            pred = model(torch.as_tensor(Xse, dtype=torch.float32)).argmax(1).numpy()
        acc = float((pred == y_te).mean())
    return TrainResult(model, losses, acc, pred, y_te)


def xor_data(n: int = 200, seed: int = 0, noise: float = 0.08):
    """Linearly inseparable 2-bit XOR — the ReLU drill, sector-agnostic."""
    rng = np.random.RandomState(seed)
    bits = rng.randint(0, 2, size=(n, 2)).astype(np.float32)
    y = (bits[:, 0].astype(int) ^ bits[:, 1].astype(int)).astype(np.int64)
    X = bits + rng.normal(0, noise, size=bits.shape).astype(np.float32)
    return X, y
