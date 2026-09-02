"""Export the project's editorial charts as image files.

Figures are written to ``reports/figures/`` and reused by the README and the
HTML case-study deck. EDA charts use the training split only; model charts use
the calibrated artifact (train it first via ``python -m stroke_risk.train``).

Run with::

    uv run python scripts/make_figures.py
"""

import warnings

import matplotlib

matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import gaussian_kde

from stroke_risk import config, data, evaluate, model
from stroke_risk import plotting as pl
from stroke_risk.features import split_features_target

warnings.filterwarnings("ignore")
FIG_DIR = config.PROJECT_ROOT / "reports" / "figures"

CATEGORICAL = [
    "gender", "ever_married", "work_type", "Residence_type", "smoking_status"
]
NUMERIC = [
    ("age", "Age"),
    ("avg_glucose_level", "Average glucose level"),
    ("bmi", "BMI"),
]


def _save(fig, name: str) -> None:
    fig.savefig(FIG_DIR / name, bbox_inches="tight", dpi=140, facecolor="white")
    plt.close(fig)


def eda_figures(train: pd.DataFrame) -> None:
    # Target balance
    rate = train["stroke"].mean()
    fig, ax = plt.subplots(figsize=(9, 1.9))
    ax.barh(0, (1 - rate) * 100, color=pl.MUTED)
    ax.barh(0, rate * 100, left=(1 - rate) * 100, color=pl.ACCENT)
    ax.text((1 - rate) * 50, 0, f"No stroke   {100 * (1 - rate):.1f}%",
            ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    ax.text(100, 0.75, f"Stroke   {100 * rate:.1f}%",
            ha="right", va="bottom", color=pl.ACCENT, fontsize=11, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 0.6)
    ax.axis("off")
    pl.add_titles(ax, "Only about 1 in 20 patients had a stroke",
                  "Share of patients by outcome (training set)")
    _save(fig, "target_balance.png")

    # Age -> stroke rate
    bins = [0, 20, 30, 40, 50, 60, 70, 80, 120]
    names = ["<20", "20s", "30s", "40s", "50s", "60s", "70s", "80+"]
    band = pd.cut(train["age"], bins=bins, labels=names, right=False)
    rate_by_age = train.groupby(band, observed=True)["stroke"].mean().mul(100)
    fig, ax = plt.subplots()
    bars = ax.bar(rate_by_age.index.astype(str), rate_by_age.values, color=pl.MUTED)
    for b, name in zip(bars, rate_by_age.index, strict=True):
        if name in ("70s", "80+"):
            b.set_color(pl.ACCENT)
    ax.bar_label(bars, fmt="%.0f%%", padding=3, color=pl.SUBTLE, fontsize=9)
    ax.set_yticks([])
    ax.grid(False)
    pl.despine(ax, left=True)
    pl.add_titles(ax, "Stroke risk climbs steeply with age",
                  "Percentage of patients in each age group who had a stroke")
    _save(fig, "age_risk.png")

    # Correlation heatmap
    corr = train.select_dtypes("number").drop(columns="id").corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr.mask(mask), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            if not mask[i, j]:
                val = corr.iloc[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(val) > 0.55 else pl.INK)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Pearson correlation")
    pl.add_titles(ax, "Age is the strongest numeric correlate of stroke",
                  "Pearson correlation between numeric features")
    _save(fig, "correlation.png")

    # KDE distributions
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (col, label) in zip(axes, NUMERIC, strict=True):
        for cls in (1, 0):
            v = train.loc[train["stroke"] == cls, col].dropna()
            xs = np.linspace(v.min(), v.max(), 200)
            ys = gaussian_kde(v)(xs)
            ax.fill_between(xs, ys, color=pl.STROKE_COLORS[cls], alpha=0.55,
                            label=pl.STROKE_LABELS[cls])
            ax.plot(xs, ys, color=pl.STROKE_COLORS[cls], lw=1.5)
        ax.set_title(label, loc="left", fontsize=12, fontweight="bold", color=pl.INK)
        ax.set_yticks([])
        ax.grid(False)
        pl.despine(ax, left=True)
    axes[0].legend(loc="upper right")
    fig.text(0, 1.02, "Stroke patients skew toward higher age, glucose and BMI",
             fontsize=15, fontweight="bold", color=pl.INK)
    fig.tight_layout()
    _save(fig, "distributions.png")

    # Ranked categorical stroke rates
    rows = []
    for col in CATEGORICAL:
        for level, sub in train.groupby(col):
            rows.append({"group": f"{col} = {level}",
                         "rate": sub["stroke"].mean() * 100, "n": len(sub)})
    ranked = pd.DataFrame(rows).sort_values("rate")
    fig, ax = plt.subplots(figsize=(8.5, 7))
    cut = ranked["rate"].median()
    colors = [pl.ACCENT if r >= cut else pl.MUTED for r in ranked["rate"]]
    bars = ax.barh(ranked["group"], ranked["rate"], color=colors)
    ax.bar_label(bars, labels=[f"{r:.1f}%  (n={n})"
                 for r, n in zip(ranked["rate"], ranked["n"], strict=True)],
                 padding=4, color=pl.SUBTLE, fontsize=8)
    ax.set_xticks([])
    ax.set_xlim(0, ranked["rate"].max() * 1.25)
    ax.grid(False)
    pl.despine(ax, left=False, bottom=True)
    pl.add_titles(ax, "Which patient groups have the highest stroke rate?",
                  "Stroke rate within each category level (training set)")
    _save(fig, "categorical_ranked.png")


def model_figures(splits) -> None:
    X_test, y_test = split_features_target(splits.test)
    artifact = joblib.load(config.MODEL_PATH)
    proba = artifact["model"].predict_proba(X_test)[:, 1]
    threshold = artifact["threshold"]

    # PR + ROC
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    evaluate.plot_pr_curve(axes[0], y_test, proba, "XGBoost")
    axes[0].set_title("Precision\u2013Recall", loc="left", fontsize=12,
                      fontweight="bold", color=pl.INK)
    evaluate.plot_roc_curve(axes[1], y_test, proba, "XGBoost")
    axes[1].set_title("ROC", loc="left", fontsize=12, fontweight="bold", color=pl.INK)
    fig.text(0, 1.02, "High ROC-AUC hides a hard precision\u2013recall trade-off",
             fontsize=15, fontweight="bold", color=pl.INK)
    fig.tight_layout()
    _save(fig, "pr_roc.png")

    # Confusion matrix
    y_pred = (proba >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    evaluate.plot_confusion_matrix(ax, y_test, y_pred)
    pl.add_titles(ax, "Catching strokes at the chosen threshold",
                  "Test-set confusion matrix")
    _save(fig, "confusion_matrix.png")

    # Calibration curve
    fig, ax = plt.subplots(figsize=(5.5, 5))
    evaluate.plot_calibration_curve(ax, y_test, proba, "Calibrated XGBoost")
    pl.add_titles(ax, "Predicted probabilities are well calibrated",
                  "Reliability curve on the test set")
    _save(fig, "calibration.png")


def shap_figure(splits) -> None:
    X_train, y_train = split_features_target(splits.train)
    xgb = model.build_xgboost(model.scale_pos_weight(y_train)).fit(X_train, y_train)
    pre = xgb.named_steps["preprocess"]
    clf = xgb.named_steps["clf"]
    X_t = pre.transform(X_train)
    names = pre.get_feature_names_out()
    shap_values = shap.TreeExplainer(clf).shap_values(X_t)
    shap.summary_plot(shap_values, X_t, feature_names=names, show=False)
    fig = plt.gcf()
    fig.savefig(FIG_DIR / "shap_summary.png", bbox_inches="tight", dpi=140,
                facecolor="white")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pl.set_theme()
    splits = data.split_data(data.load_raw())
    eda_figures(splits.train)
    model_figures(splits)
    shap_figure(splits)
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
