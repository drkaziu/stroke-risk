# Model Card — Stroke Risk

> ⚠️ **Educational project. Not a medical device.** This model must **not** be
> used for diagnosis, screening, or any clinical decision. See the repository
> [disclaimer](README.md).

## Model details

- **Task:** binary classification — estimate the probability that a patient has
  had (or will have) a stroke, from clinical and demographic features.
- **Algorithm:** gradient-boosted trees (**XGBoost**) inside a scikit-learn
  `Pipeline` (median imputation → scaling → one-hot encoding).
- **Class imbalance:** handled with class weights (`scale_pos_weight`), not resampling.
- **Probability calibration:** isotonic regression (`CalibratedClassifierCV`,
  cross-fitted), so the output probabilities are meaningful, not just rankings.
- **Decision threshold:** tuned on a validation split (not the test set).
- **Selection:** best of Logistic Regression vs XGBoost by validation PR-AUC.

## Intended use

- **In scope:** learning and demonstration — data pipelines, leak-free modelling,
  calibration, evaluation, and deployment.
- **Out of scope:** any real medical, clinical, insurance, or individual-level use.

## Training data

- Public **Stroke Prediction Dataset** (Kaggle, `fedesoriano`), ~5,110 patients,
  ~5% positive (stroke) — a strong class imbalance.
- The dataset is **not redistributed** here; see [data/README.md](data/README.md).
- Split **stratified** 60/20/20 into train/validation/test. All learned
  transformations are fit on the training split only (no leakage).

## Metrics (held-out test set)

Reported once, on data untouched during training and tuning:

| Metric | Value |
| --- | --- |
| ROC-AUC | ~0.83 |
| PR-AUC (average precision) | ~0.23 |
| Recall | ~0.68 |
| Precision | ~0.15 |
| F1 | ~0.25 |
| Brier score (calibration) | ~0.042 |

PR-AUC is the headline metric because ROC-AUC flatters imbalanced problems.
After calibration, the mean predicted probability (~0.046) closely matches the
true stroke rate (~0.049).

## Limitations

- **Modest predictive power.** Stroke is rare and hard to predict from these
  features; PR-AUC is low. The model finds signal but is far from clinically useful.
- **Precision/recall trade-off.** At the chosen threshold it catches most strokes
  (high recall) at the cost of many false alarms (low precision).
- **Dataset limitations.** Small, single-source, possibly biased sampling; some
  features (e.g. glucose) may be measured *after* a stroke, confounding causality.
- **Correlation, not causation.** Associations (e.g. "ever married" ↔ risk)
  largely reflect age, not causal effects.

## Ethical considerations

- Health predictions can cause real harm if misused. This model is deliberately
  gated behind a prominent disclaimer and must not inform real decisions.
- No individual is identifiable in the (public, de-identified) training data.

## How to use (educational)

```bash
uv run python -m stroke_risk.train      # train + calibrate + save artifact
uv run uvicorn app.main:app --reload    # serve the demo API + UI
```
