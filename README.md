# Stroke Risk

Predicting stroke risk from clinical and demographic features — rebuilt as a
professional, reproducible data science project.

> Educational project. **Not** for real clinical use.

## Status

Phase 4 — deployment (tuned model served via FastAPI).

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

On macOS, XGBoost needs the OpenMP runtime:

```bash
brew install libomp
```

Place the dataset at `data/dataset.csv` (see [data/README.md](data/README.md)).

## Train and serve

```bash
uv run python -m stroke_risk.train          # tune, select, save models/stroke_model.joblib
uv run uvicorn app.main:app --reload        # serve at http://127.0.0.1:8000
```

Open http://127.0.0.1:8000 for the UI, or http://127.0.0.1:8000/docs for the API.

## Development

```bash
uv run pytest      # run tests
uv run ruff check  # lint
```

## Project layout

```
src/stroke_risk/   # importable package (data, features, model, tune, train)
app/               # FastAPI serving layer + UI
notebooks/         # EDA, modelling, tuning
tests/             # pytest suite
data/              # raw dataset (not tracked)
models/            # trained artifacts (not tracked)
```
