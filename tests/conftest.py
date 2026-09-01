"""Shared test fixtures.

When the real (git-ignored) dataset is absent — e.g. on CI — fall back to the
committed synthetic sample so the full suite can still run.
"""

import pytest

from stroke_risk import config


@pytest.fixture(autouse=True, scope="session")
def _dataset_fallback():
    if not config.DATA_PATH.exists():
        config.DATA_PATH = config.PROJECT_ROOT / "data" / "sample.csv"
    yield
