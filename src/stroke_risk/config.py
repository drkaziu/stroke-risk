"""Central configuration: paths, seed, and split sizes.

Keeping these in one place avoids hardcoded paths and magic numbers scattered
across the codebase, and makes every run reproducible.
"""

from pathlib import Path

# Repo root is two levels up from this file: src/stroke_risk/config.py -> repo/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "stroke_model.joblib"

RANDOM_SEED = 42

# Column roles known from the raw CSV header (not from any modelling decision).
ID_COL = "id"
TARGET = "stroke"

# Feature groups for preprocessing (discovered during EDA on the training split).
NUMERIC_FEATURES = ["age", "avg_glucose_level", "bmi"]
BINARY_FEATURES = ["hypertension", "heart_disease"]
CATEGORICAL_FEATURES = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]

# Split proportions of the full dataset: 60% train / 20% validation / 20% test.
TEST_SIZE = 0.2
VAL_SIZE = 0.2
