"""FastAPI serving layer for the stroke-risk model.

Every input is validated by a Pydantic schema before it reaches the model, and
the single-row frame is built with the exact raw column names the training
pipeline expects.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from stroke_risk import config

app = FastAPI(title="Stroke Risk API", version="1.0.0")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


@lru_cache
def load_artifact() -> dict:
    """Load the trained model artifact once and cache it."""
    return joblib.load(config.MODEL_PATH)


class PatientFeatures(BaseModel):
    """Validated patient inputs. Field names match the raw dataset columns."""

    gender: Literal["Male", "Female", "Other"]
    age: float = Field(ge=0, le=120)
    hypertension: Literal[0, 1]
    heart_disease: Literal[0, 1]
    ever_married: Literal["Yes", "No"]
    work_type: Literal[
        "Private", "Self-employed", "Govt_job", "children", "Never_worked"
    ]
    Residence_type: Literal["Urban", "Rural"]
    avg_glucose_level: float = Field(gt=0, le=500)
    # BMI may be unknown; the pipeline imputes it.
    bmi: float | None = Field(default=None, gt=0, le=100)
    smoking_status: Literal["formerly smoked", "never smoked", "smokes", "Unknown"]

    def to_row(self) -> dict:
        row = self.model_dump()
        if row["bmi"] is None:
            row["bmi"] = np.nan
        return row


class Prediction(BaseModel):
    probability: float
    risk: bool
    threshold: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html")


@app.post("/predict", response_model=Prediction)
def predict(features: PatientFeatures) -> Prediction:
    artifact = load_artifact()
    model = artifact["model"]
    threshold = artifact["threshold"]

    row = pd.DataFrame([features.to_row()])
    probability = float(model.predict_proba(row)[0, 1])
    return Prediction(
        probability=probability,
        risk=probability >= threshold,
        threshold=threshold,
    )
