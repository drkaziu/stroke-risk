"""Data loading and splitting.

This module deliberately does the split *before* any statistical cleaning so
that later steps (imputation, outlier handling) can be fit on the training set
alone. That ordering is what prevents test-set information from leaking into
the model.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from stroke_risk import config


@dataclass(frozen=True)
class DataSplits:
    """Stratified train/validation/test partitions, each still full DataFrames."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load the raw dataset exactly as stored, with no cleaning applied."""
    # Resolve at call time so the path can be overridden (e.g. in tests/CI).
    return pd.read_csv(path or config.DATA_PATH)


def split_data(
    df: pd.DataFrame,
    *,
    target: str = config.TARGET,
    test_size: float = config.TEST_SIZE,
    val_size: float = config.VAL_SIZE,
    seed: int = config.RANDOM_SEED,
) -> DataSplits:
    """Split into train/val/test, stratified on the target to preserve the
    (heavily imbalanced) class ratio in every partition.
    """
    stratify = df[target]
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=stratify
    )

    # Size of the validation slice relative to the remaining train+val portion.
    val_fraction = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_fraction,
        random_state=seed,
        stratify=train_val[target],
    )

    return DataSplits(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )
