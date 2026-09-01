"""Smoke tests for the model pipelines.

These check the models are wired correctly and produce valid probabilities;
predictive quality is assessed in the modelling notebook, not here.
"""

import numpy as np
import pytest

from stroke_risk.data import load_raw, split_data
from stroke_risk.features import split_features_target
from stroke_risk.model import build_models, scale_pos_weight


@pytest.fixture(scope="module")
def train_xy():
    train = split_data(load_raw()).train
    return split_features_target(train)


def test_scale_pos_weight_is_greater_than_one(train_xy):
    _, y = train_xy
    # The positive class is the minority, so negatives/positives > 1.
    assert scale_pos_weight(y) > 1


def test_models_fit_and_produce_valid_probabilities(train_xy):
    X, y = train_xy
    sample = X.head(300)
    y_sample = y.head(300)
    for pipe in build_models(y_sample).values():
        pipe.fit(sample, y_sample)
        proba = pipe.predict_proba(sample)[:, 1]
        assert proba.shape == (len(sample),)
        assert np.all((proba >= 0) & (proba <= 1))
