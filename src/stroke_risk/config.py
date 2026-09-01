"""Central configuration: paths, seed, and split sizes.

Keeping these in one place avoids hardcoded paths and magic numbers scattered
across the codebase, and makes every run reproducible.
"""

from pathlib import Path

# Repo root is two levels up from this file: src/stroke_risk/config.py -> repo/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "dataset.csv"

RANDOM_SEED = 42

# Column roles known from the raw CSV header (not from any modelling decision).
ID_COL = "id"
TARGET = "stroke"

# Split proportions of the full dataset: 60% train / 20% validation / 20% test.
TEST_SIZE = 0.2
VAL_SIZE = 0.2
