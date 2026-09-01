"""Evaluation: metrics, threshold selection, and editorial evaluation charts.

Metric choices reflect a rare, high-cost event: we lead with PR-AUC (average
precision) and recall, and treat ROC-AUC as secondary because it flatters models
on imbalanced data. The decision threshold is chosen on validation data, never
on the test set.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from stroke_risk import plotting as pl


def choose_threshold(y_true, y_proba) -> float:
    """Pick the probability threshold that maximises F1 on the given data."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision/recall have one more element than thresholds; drop the last.
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[np.argmax(f1)])


def summarize(y_true, y_proba, threshold: float) -> dict[str, float]:
    """Threshold-independent (AUC) and threshold-dependent metrics together."""
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "PR-AUC": average_precision_score(y_true, y_proba),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "threshold": threshold,
    }


def plot_pr_curve(ax, y_true, y_proba, label: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    ax.plot(recall, precision, color=pl.ACCENT, lw=2, label=f"{label} (AP={ap:.2f})")
    baseline = np.mean(y_true)
    ax.axhline(baseline, color=pl.SUBTLE, lw=1, ls="--")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1)
    ax.grid(False)
    ax.legend(loc="upper right")


def plot_roc_curve(ax, y_true, y_proba, label: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    ax.plot(fpr, tpr, color=pl.ACCENT, lw=2, label=f"{label} (AUC={auc:.2f})")
    ax.plot([0, 1], [0, 1], color=pl.SUBTLE, lw=1, ls="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.grid(False)
    ax.legend(loc="lower right")


def plot_confusion_matrix(ax, y_true, y_pred) -> None:
    cm = confusion_matrix(y_true, y_pred)
    ax.imshow(cm, cmap="Reds")
    labels = ["No stroke", "Stroke"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.grid(False)
    threshold = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:,}", ha="center", va="center",
                color="white" if cm[i, j] > threshold else pl.INK,
                fontsize=13, fontweight="bold",
            )


def metrics_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Turn per-model metric dicts into a tidy, rounded comparison table."""
    return pd.DataFrame(results).T.round(3)
