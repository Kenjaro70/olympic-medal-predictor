"""Model validation tests.

Trains a small logistic regression on the committed sample dataset
(data/sample.csv) so the tests run in CI without the full 35 MB dataset
or a prior training run.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.preprocess import features_target, prepare_datasets

SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "sample.csv"


@pytest.fixture(scope="module")
def trained():
    df = pd.read_csv(SAMPLE_PATH)
    train, test, _ = prepare_datasets(df, test_size=0.2, random_state=42)
    X_train, y_train = features_target(train)
    X_test, y_test = features_target(test)
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X_train, y_train)
    return model, X_test, y_test


def test_predictions_have_correct_type_and_shape(trained):
    model, X_test, y_test = trained
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(X_test),)
    assert set(np.unique(preds)) <= {0, 1}
    assert proba.shape == (len(X_test), 2)
    assert ((proba >= 0) & (proba <= 1)).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)


def test_model_meets_minimum_performance(trained):
    from sklearn.metrics import roc_auc_score

    model, X_test, y_test = trained
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    # Medal prediction from demographics is hard; anything well above chance
    # confirms the pipeline produces signal, not garbage.
    assert auc >= 0.60, f"ROC AUC {auc:.3f} below minimum threshold 0.60"
