"""Generate a small synthetic sample dataset for CI and quick local runs.

The real Kaggle dataset is not redistributed, so tests fall back to this
synthetic file (``data/sample.csv``) when the real one is absent. Values are
fake but share the schema, categories, class imbalance, and a few quirks
(missing BMI, a rare gender) of the real data.

Run with::

    uv run python scripts/make_sample.py
"""

import numpy as np
import pandas as pd

from stroke_risk import config

N = 400
SEED = 42


def make_sample() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    age = rng.uniform(1, 90, N).round(1)
    # Stroke risk rises with age; keep the overall rate low (~8%).
    stroke_prob = np.clip((age - 40) / 200, 0.01, 0.35)
    stroke = (rng.random(N) < stroke_prob).astype(int)

    bmi = rng.normal(28, 6, N).round(1)
    # Punch some holes in BMI, like the real data.
    bmi[rng.random(N) < 0.04] = np.nan

    df = pd.DataFrame(
        {
            "id": rng.permutation(90000)[:N],
            "gender": rng.choice(
                ["Male", "Female", "Other"], N, p=[0.41, 0.585, 0.005]
            ),
            "age": age,
            "hypertension": (rng.random(N) < 0.1).astype(int),
            "heart_disease": (rng.random(N) < 0.06).astype(int),
            "ever_married": rng.choice(["Yes", "No"], N, p=[0.65, 0.35]),
            "work_type": rng.choice(
                ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
                N, p=[0.57, 0.16, 0.13, 0.13, 0.01],
            ),
            "Residence_type": rng.choice(["Urban", "Rural"], N),
            "avg_glucose_level": rng.normal(106, 45, N).clip(55, 280).round(2),
            "bmi": bmi,
            "smoking_status": rng.choice(
                ["never smoked", "Unknown", "formerly smoked", "smokes"],
                N, p=[0.37, 0.30, 0.17, 0.16],
            ),
            "stroke": stroke,
        }
    )
    return df


def main() -> None:
    path = config.PROJECT_ROOT / "data" / "sample.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    make_sample().to_csv(path, index=False)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
