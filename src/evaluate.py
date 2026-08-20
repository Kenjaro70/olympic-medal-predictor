"""Model evaluation utilities."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def evaluate_model(model, X, y) -> dict[str, float]:
    """Compute classification metrics on a held-out set."""
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds)),
        "f1": float(f1_score(y, preds)),
        "roc_auc": float(roc_auc_score(y, proba)),
    }


def feature_importances(model, feature_names: list[str]) -> dict[str, float]:
    """Best-effort global importance ranking for explanation prompts."""
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
    else:
        return {}
    total = values.sum() or 1.0
    ranked = sorted(zip(feature_names, values / total), key=lambda kv: -kv[1])
    return {name: round(float(v), 4) for name, v in ranked}
