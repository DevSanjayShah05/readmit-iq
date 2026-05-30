"""
Pydantic schemas for the ReadmitIQ HTTP API.

These describe the JSON shapes the API accepts and returns. They're
distinct from the domain models in readmit_iq.models — the API contract
shouldn't leak internal fields (database id, created_at) or be coupled
to internal naming.

Two-way mapping helpers live alongside the schemas: PatientRequest.to_domain()
turns an inbound request into our internal Patient object.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from readmit_iq.models import Patient

# ---------- Request schemas ----------


class PatientRequest(BaseModel):
    """A patient sent in for scoring or explanation."""

    mrn: str = Field(
        ..., min_length=1, max_length=64, description="Medical record number"
    )
    age: int = Field(..., ge=0, le=120, description="Age in years")
    sex: Literal["F", "M", "O"] = Field(..., description="F, M, or O")
    admission_date: date = Field(..., description="Date of admission")
    discharge_date: date = Field(..., description="Date of discharge")
    primary_diagnosis: str | None = Field(
        None, max_length=10, description="ICD-10 code, or null"
    )

    def to_domain(self) -> Patient:
        """Convert this API model into the internal domain Patient."""
        return Patient(
            mrn=self.mrn,
            age=self.age,
            sex=self.sex,
            admission_date=self.admission_date,
            discharge_date=self.discharge_date,
            primary_diagnosis=self.primary_diagnosis,
            readmitted_30d=False,  # unknown at prediction time
        )


class BatchPredictRequest(BaseModel):
    """Wrapper for predicting on many patients in one call."""

    patients: list[PatientRequest] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="One to one thousand patients to score",
    )


# ---------- Response schemas ----------


class PredictionResponse(BaseModel):
    """Single patient prediction."""

    mrn: str
    readmission_probability: float = Field(..., ge=0.0, le=1.0)
    risk_band: Literal["low", "medium", "high"]


class BatchPredictionResponse(BaseModel):
    """Many predictions in one call."""

    predictions: list[PredictionResponse]


class FeatureContributionResponse(BaseModel):
    """One feature's contribution to one patient's predicted probability."""

    feature_name: str
    feature_value: float
    shap_value: float


class ExplanationResponse(BaseModel):
    """SHAP explanation for one patient."""

    mrn: str
    predicted_probability: float
    baseline_probability: float
    contributions: list[FeatureContributionResponse]


class BatchExplanationResponse(BaseModel):
    """Many explanations in one call."""

    explanations: list[ExplanationResponse]


class RetrievedCitationResponse(BaseModel):
    """One retrieved biomedical citation relevant to a patient's prediction."""

    pmid: str
    title: str
    journal: str
    year: str
    authors: list[str]
    score: float = Field(..., description="Cosine similarity score (0-1)")
    pubmed_url: str


class ExplanationWithCitationsResponse(BaseModel):
    """SHAP explanation plus retrieved literature for one patient."""

    mrn: str
    predicted_probability: float
    baseline_probability: float
    risk_band: Literal["low", "medium", "high"]
    contributions: list[FeatureContributionResponse]
    query: str = Field(..., description="The composed query used for retrieval")
    citations: list[RetrievedCitationResponse]


class BatchExplanationWithCitationsResponse(BaseModel):
    """Many explanations-with-citations in one call."""

    results: list[ExplanationWithCitationsResponse]


# ---------- Helper functions ----------


def probability_to_risk_band(probability: float) -> Literal["low", "medium", "high"]:
    """
    Bucket a continuous probability into a categorical risk band for display.

    These thresholds are deliberately conservative for clinical work — the cost
    of a missed readmission (false negative) is higher than the cost of a
    false alarm. In a real deployment these would be calibrated against
    operational capacity (how many high-risk flags can the care team act on?).
    """
    if probability < 0.15:
        return "low"
    if probability < 0.30:
        return "medium"
    return "high"