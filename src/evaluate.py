"""Model evaluation utilities."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Default operating point when no threshold has been tuned.
DEFAULT_THRESHOLD = 0.5

# pr_auc is the headline metric. At a ~15% medal rate a model that never
# predicts a medal scores 85% accuracy, and ROC-AUC stays flattering under
# imbalance; average precision is scored against the base rate instead.
METRIC_NAMES = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "pr_auc_lift",
    "base_rate",
    "threshold",
]


def choose_threshold(
    y_true, proba, target_precision: float = 0.6
) -> tuple[float, dict[str, float]]:
    """Pick the operating point that maximizes recall at >= target_precision.

    Must be called on a VALIDATION slice, never on the test set -- tuning a
    threshold on the data you report is leakage, and a subtle kind, because
    the model itself was trained honestly.

    Among thresholds meeting the precision target, the one with the highest
    recall is chosen: given the constraint "be precise", catching as many
    medalists as possible is the remaining free objective. If no threshold
    reaches the target (the target is above the model's ceiling on this
    slice), the best achievable precision is used instead and reported in
    the returned diagnostics, so the caller can see the target was missed
    rather than silently accepting a weaker operating point.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # precision/recall have one more element than thresholds; drop the last.
    precision, recall = precision[:-1], recall[:-1]

    feasible = precision >= target_precision
    if feasible.any():
        # Highest recall among points that clear the precision bar.
        idx = int(np.argmax(np.where(feasible, recall, -1.0)))
        met = True
    else:
        idx = int(np.argmax(precision))
        met = False

    return float(thresholds[idx]), {
        "target_precision": float(target_precision),
        "target_met": met,
        "val_precision": float(precision[idx]),
        "val_recall": float(recall[idx]),
    }


def evaluate_model(model, X, y, threshold: float = DEFAULT_THRESHOLD) -> dict[str, float]:
    """Compute classification metrics on a held-out set.

    ``threshold`` sets the operating point for the hard-label metrics
    (precision/recall/f1/accuracy). pr_auc and roc_auc are threshold-free
    and describe ranking quality regardless of where the cut is made.
    """
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)
    base_rate = float(np.mean(y))
    pr_auc = float(average_precision_score(y, proba))
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds)),
        "f1": float(f1_score(y, preds)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": pr_auc,
        # How many times better than always-guess-the-base-rate. A value of
        # 1.0 means the model has learned nothing useful about who medals.
        "pr_auc_lift": float(pr_auc / base_rate) if base_rate else 0.0,
        "base_rate": base_rate,
        "threshold": float(threshold),
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
