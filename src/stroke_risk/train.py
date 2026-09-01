"""Train and persist the deployable model.

Run as a script::

    uv run python -m stroke_risk.train

This tunes both candidate models on the training split, picks the winner by
validation PR-AUC, tunes the decision threshold on validation, then refits the
winner on train+validation and saves a single artifact (model + threshold +
metadata) for the serving layer.
"""

from __future__ import annotations

import datetime as dt
import warnings

import joblib
import pandas as pd

from stroke_risk import config, evaluate, tune
from stroke_risk.data import load_raw, split_data
from stroke_risk.features import split_features_target


def train_model(n_iter: int = 30) -> dict:
    """Tune, select, and refit the deployable model; return the artifact dict."""
    splits = split_data(load_raw())
    X_train, y_train = split_features_target(splits.train)
    X_val, y_val = split_features_target(splits.val)

    specs = tune.build_search_specs(y_train)
    searches = {
        name: tune.tune(pipe, space, X_train, y_train, n_iter=n_iter)
        for name, (pipe, space) in specs.items()
    }

    # Choose the winner by validation PR-AUC.
    val_pr_auc = {}
    for name, search in searches.items():
        proba_val = search.best_estimator_.predict_proba(X_val)[:, 1]
        val_pr_auc[name] = evaluate.summarize(y_val, proba_val, 0.5)["PR-AUC"]
    winner = max(val_pr_auc, key=val_pr_auc.get)

    best = searches[winner].best_estimator_
    threshold = evaluate.choose_threshold(y_val, best.predict_proba(X_val)[:, 1])

    # Refit on train + validation to give the deployed model more data.
    X_fit = pd.concat([X_train, X_val])
    y_fit = pd.concat([y_train, y_val])
    best.fit(X_fit, y_fit)

    return {
        "model": best,
        "threshold": float(threshold),
        "model_name": winner,
        "params": searches[winner].best_params_,
        "feature_columns": list(X_fit.columns),
        "trained_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }


def main() -> None:
    warnings.filterwarnings("ignore")
    artifact = train_model()
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, config.MODEL_PATH)
    print(f"Saved {artifact['model_name']} (threshold={artifact['threshold']:.3f})")
    print(f"-> {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
