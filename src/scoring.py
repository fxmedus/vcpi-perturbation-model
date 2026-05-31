"""
wMSE scoring + held-out-compound cross-validation.
CRITICAL: weights must be the OFFICIAL per-gene weight vector from the
hackathon repo. Until you supply it, uniform weights are used with a warning.
"""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import KFold


def wmse(pred: np.ndarray, true: np.ndarray, weights: np.ndarray | None = None) -> float:
    per_gene = ((pred - true) ** 2).mean(axis=0)
    if weights is None:
        return float(per_gene.mean())
    return float(np.average(per_gene, weights=weights))


def cv_compounds(model_fn, X, Y, weights=None, n_splits: int = 5, seed: int = 0, **kw):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tr, va in kf.split(X):
        pred, _ = model_fn(X[tr], Y[tr], X[va], **kw)
        scores.append(wmse(pred, Y[va], weights))
    return float(np.mean(scores)), scores


def warn_if_default_weights(weights) -> None:
    if weights is None:
        print("[WARN] Using uniform gene weights. Local wMSE will NOT match "
              "the leaderboard until you supply the official per-gene weight vector.")
