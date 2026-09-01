"""Tests for the data split and preprocessing pipeline.

The point of these tests is to guard the two properties that are easy to break
and disastrous when broken: correct partitioning and no target/test leakage.
"""

import numpy as np
import pytest

from stroke_risk import config
from stroke_risk.data import load_raw, split_data
from stroke_risk.features import build_preprocessor, split_features_target


@pytest.fixture(scope="module")
def raw():
    return load_raw()


@pytest.fixture(scope="module")
def splits(raw):
    return split_data(raw)


def test_splits_sum_to_total(raw, splits):
    assert len(splits.train) + len(splits.val) + len(splits.test) == len(raw)


def test_no_overlap_between_splits(splits):
    ids = [set(part[config.ID_COL]) for part in (splits.train, splits.val, splits.test)]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])


def test_stratification_preserved(raw, splits):
    overall = raw[config.TARGET].mean()
    for part in (splits.train, splits.val, splits.test):
        assert part[config.TARGET].mean() == pytest.approx(overall, abs=0.005)


def test_split_is_deterministic(raw):
    a = split_data(raw)
    b = split_data(raw)
    assert list(a.train[config.ID_COL]) == list(b.train[config.ID_COL])


def test_preprocessor_drops_id_and_target(splits):
    X_train, _ = split_features_target(splits.train)
    pre = build_preprocessor().fit(X_train)
    used = set(pre.feature_names_in_)
    assert config.TARGET not in used
    # ``id`` is only dropped via remainder, so it may appear in feature_names_in_
    # but must not survive into the transformed output.
    out_names = pre.get_feature_names_out()
    assert not any(
        name == config.ID_COL or name.endswith(f"__{config.ID_COL}")
        for name in out_names
    )


def test_bmi_imputer_learns_train_median(splits):
    """Leak-free check: the fill value comes from the training set alone."""
    X_train, _ = split_features_target(splits.train)
    pre = build_preprocessor().fit(X_train)
    imputer = pre.named_transformers_["num"].named_steps["impute"]
    bmi_index = config.NUMERIC_FEATURES.index("bmi")
    learned = imputer.statistics_[bmi_index]
    assert learned == pytest.approx(X_train["bmi"].median())


def test_no_nan_after_transform(splits):
    X_train, _ = split_features_target(splits.train)
    pre = build_preprocessor().fit(X_train)
    for part in (splits.train, splits.val, splits.test):
        X, _ = split_features_target(part)
        transformed = pre.transform(X)
        assert not np.isnan(transformed).any()
        assert transformed.shape[0] == len(part)
