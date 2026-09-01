"""Hyperparameter tuning via randomized search.

The search runs inside stratified cross-validation on the **training set only**,
so the validation and test splits stay untouched. We optimise ``average_precision``
(PR-AUC), the project's lead metric for this rare-event problem.
"""

import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from stroke_risk import config
from stroke_risk.model import build_logreg, build_xgboost, scale_pos_weight

# Parameter names are prefixed with the pipeline step they belong to (``clf``).
LOGREG_SPACE = {
    "clf__C": loguniform(1e-3, 1e2),
}

XGB_SPACE = {
    "clf__n_estimators": randint(200, 800),
    "clf__max_depth": randint(2, 7),
    "clf__learning_rate": loguniform(1e-2, 3e-1),
    "clf__subsample": uniform(0.6, 0.4),  # 0.6 .. 1.0
    "clf__colsample_bytree": uniform(0.6, 0.4),
    "clf__min_child_weight": randint(1, 10),
    "clf__gamma": uniform(0.0, 5.0),
    "clf__reg_lambda": loguniform(1e-1, 1e2),
    "clf__reg_alpha": loguniform(1e-3, 1e1),
}


def build_search_specs(y_train: pd.Series) -> dict[str, tuple]:
    """Return ``{name: (pipeline, param_space)}`` for each candidate model."""
    xgb = build_xgboost(scale_pos_weight(y_train))
    # Single-threaded trees so the outer search can parallelise cleanly.
    xgb.set_params(clf__n_jobs=1)
    return {
        "Logistic Regression": (build_logreg(), LOGREG_SPACE),
        "XGBoost": (xgb, XGB_SPACE),
    }


def tune(
    pipeline,
    param_space: dict,
    X,
    y,
    *,
    n_iter: int = 30,
    seed: int = config.RANDOM_SEED,
) -> RandomizedSearchCV:
    """Run randomized search, optimising PR-AUC via stratified CV."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_space,
        n_iter=n_iter,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        random_state=seed,
        refit=True,
    )
    search.fit(X, y)
    return search
