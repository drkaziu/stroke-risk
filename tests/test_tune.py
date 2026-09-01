"""Fast guard test for the tuning module (small search, logistic regression)."""

import pytest
from sklearn.model_selection import RandomizedSearchCV

from stroke_risk.data import load_raw, split_data
from stroke_risk.features import split_features_target
from stroke_risk.model import build_logreg
from stroke_risk.tune import LOGREG_SPACE, tune


@pytest.fixture(scope="module")
def train_xy():
    return split_features_target(split_data(load_raw()).train)


def test_tune_returns_fitted_search(train_xy):
    X, y = train_xy
    search = tune(build_logreg(), LOGREG_SPACE, X.head(400), y.head(400), n_iter=2)
    assert isinstance(search, RandomizedSearchCV)
    assert 0.0 <= search.best_score_ <= 1.0
    assert hasattr(search, "best_estimator_")
