# Stroke Risk

Predicting stroke risk from clinical and demographic features — rebuilt as a
professional, reproducible data science project.

> Educational project. **Not** for real clinical use.

## Status

Phase 1 — leak-free, reproducible data pipeline (in progress).

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Place the dataset at `data/dataset.csv` (see [data/README.md](data/README.md)).

## Development

```bash
uv run pytest      # run tests
uv run ruff check  # lint
```

## Project layout

```
src/stroke_risk/   # importable package (data pipeline, models)
tests/             # pytest suite
data/              # raw dataset (not tracked)
```
