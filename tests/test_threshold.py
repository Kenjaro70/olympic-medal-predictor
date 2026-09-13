"""Tests for operating-point selection in src/evaluate.py."""

import numpy as np
import pytest

from src.evaluate import DEFAULT_THRESHOLD, choose_threshold, evaluate_model


class _FakeModel:
    """Returns a fixed probability column, so thresholds are the only variable."""

    def __init__(self, proba):
        self._proba = np.asarray(proba, dtype=float)

    def predict_proba(self, X):
        return np.column_stack([1 - self._proba, self._proba])


@pytest.fixture
def separable():
    """Scores where a threshold near 0.5 separates the classes cleanly."""
    proba = np.array([0.05, 0.1, 0.2, 0.35, 0.45, 0.6, 0.7, 0.8, 0.9, 0.95])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    return y, proba


def test_threshold_meets_precision_target(separable):
    y, proba = separable
    threshold, info = choose_threshold(y, proba, target_precision=0.9)
    assert info["target_met"] is True
    assert info["val_precision"] >= 0.9
    preds = (proba >= threshold).astype(int)
    assert preds.sum() > 0, "a usable threshold must predict some positives"


def test_higher_target_precision_is_never_looser(separable):
    y, proba = separable
    low, _ = choose_threshold(y, proba, target_precision=0.5)
    high, _ = choose_threshold(y, proba, target_precision=0.95)
    assert high >= low


def test_unreachable_target_is_reported_not_silently_accepted():
    # Pure noise: precision can never reach 0.99.
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    proba = rng.random(200)
    _, info = choose_threshold(y, proba, target_precision=0.99)
    assert info["target_met"] is False
    assert info["val_precision"] < 0.99


def test_maximizes_recall_among_feasible_points():
    # Two thresholds both hit precision 1.0; the lower one catches more.
    proba = np.array([0.1, 0.2, 0.6, 0.7, 0.8])
    y = np.array([0, 0, 1, 1, 1])
    threshold, info = choose_threshold(y, proba, target_precision=1.0)
    assert info["target_met"] is True
    assert info["val_recall"] == pytest.approx(1.0)
    assert (proba >= threshold).sum() == 3


def test_evaluate_model_respects_threshold(separable):
    y, proba = separable
    model = _FakeModel(proba)
    strict = evaluate_model(model, None, y, threshold=0.85)
    loose = evaluate_model(model, None, y, threshold=0.5)
    # Raising the bar trades recall for precision.
    assert strict["precision"] >= loose["precision"]
    assert strict["recall"] <= loose["recall"]
    assert strict["threshold"] == 0.85
    # Ranking metrics are threshold-free and must not move.
    assert strict["pr_auc"] == pytest.approx(loose["pr_auc"])
    assert strict["roc_auc"] == pytest.approx(loose["roc_auc"])


def test_evaluate_model_defaults_to_half(separable):
    y, proba = separable
    assert evaluate_model(_FakeModel(proba), None, y)["threshold"] == DEFAULT_THRESHOLD


def test_pooling_folds_concatenates_in_order():
    from src.evaluate import pool_fold_predictions

    y, proba = pool_fold_predictions(
        [([0, 1], [0.1, 0.9]), ([1, 0, 1], [0.8, 0.2, 0.7])]
    )
    assert list(y) == [0, 1, 1, 0, 1]
    assert list(proba) == pytest.approx([0.1, 0.9, 0.8, 0.2, 0.7])


def test_pooling_rejects_no_folds():
    from src.evaluate import pool_fold_predictions

    with pytest.raises(ValueError, match="no folds"):
        pool_fold_predictions([])


def test_stability_reports_spread_across_folds():
    from src.evaluate import threshold_stability

    # Fold 1 is clean at this cut, fold 2 is not -- the spread must show it.
    folds = [
        ([1, 1, 0], [0.9, 0.8, 0.1]),
        ([1, 0, 0], [0.9, 0.8, 0.1]),
    ]
    info = threshold_stability(folds, threshold=0.5)
    assert info["fold_precision_max"] == pytest.approx(1.0)
    assert info["fold_precision_min"] == pytest.approx(0.5)
    assert info["fold_precision_spread"] == pytest.approx(0.5)
    assert info["fold_precision_n"] == 2


def test_stability_skips_folds_selecting_nothing():
    from src.evaluate import threshold_stability

    folds = [([1, 0], [0.9, 0.1]), ([1, 0], [0.2, 0.1])]
    info = threshold_stability(folds, threshold=0.5)
    # Second fold predicts no positives: precision is undefined, not zero.
    assert info["fold_precision_n"] == 1
    assert info["fold_precision_min"] == pytest.approx(1.0)


def test_stability_with_no_usable_fold_is_empty():
    from src.evaluate import threshold_stability

    assert threshold_stability([([1, 0], [0.1, 0.2])], threshold=0.9) == {}
