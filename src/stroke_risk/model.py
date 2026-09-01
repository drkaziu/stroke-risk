"""Model definitions.

Each model is a full ``Pipeline`` that starts with the leak-free preprocessor,
so imputation/scaling/encoding are always fit on training folds only. Class
imbalance is handled with class weights (no resampling), which keeps the whole
process leak-free and reproducible.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from stroke_risk import config
from stroke_risk.features import build_preprocessor


def scale_pos_weight(y: pd.Series) -> float:
    """Ratio of negatives to positives, used to rebalance the XGBoost loss."""
    positives = int(y.sum())
    negatives = len(y) - positives
    return negatives / positives


def build_logreg() -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=config.RANDOM_SEED,
                ),
            ),
        ]
    )


def build_xgboost(pos_weight: float) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor()),
            (
                "clf",
                XGBClassifier(
                    n_estimators=400,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    scale_pos_weight=pos_weight,
                    random_state=config.RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_models(y_train: pd.Series) -> dict[str, Pipeline]:
    """Return the candidate models, keyed by display name."""
    return {
        "Logistic Regression": build_logreg(),
        "XGBoost": build_xgboost(scale_pos_weight(y_train)),
    }
