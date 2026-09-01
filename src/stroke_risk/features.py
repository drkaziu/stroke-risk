"""Preprocessing pipeline.

Every learned transformation (median imputation, scaling, category encoding)
lives inside a scikit-learn ``Pipeline``. Because a pipeline is *fit* on the
training data and only *transforms* validation/test data, the fit/transform
boundary mechanically prevents test-set information from leaking into training.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from stroke_risk import config


def split_features_target(
    df: pd.DataFrame,
    *,
    target: str = config.TARGET,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the target column from the feature columns."""
    return df.drop(columns=[target]), df[target]


def build_preprocessor() -> ColumnTransformer:
    """Build the leak-free feature preprocessor.

    - numeric: median-impute (only ``bmi`` has gaps) then standard-scale
    - binary: pass 0/1 flags through untouched
    - categorical: one-hot encode, ignoring categories unseen at fit time
    """
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    return ColumnTransformer(
        transformers=[
            ("num", numeric, config.NUMERIC_FEATURES),
            ("bin", "passthrough", config.BINARY_FEATURES),
            ("cat", categorical, config.CATEGORICAL_FEATURES),
        ],
        # Drops everything else (notably ``id``).
        remainder="drop",
    )
