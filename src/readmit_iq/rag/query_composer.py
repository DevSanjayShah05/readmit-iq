"""
Query composition for ReadmitIQ RAG retrieval.

Turns a Patient record into a clinician-sounding natural-language query
that we can embed and use for similarity search. This matters because
embedding models trained on language work best on language: a query like
"30-day readmission risk in an elderly heart failure patient" retrieves
better than "age=78 dx_I50.9 length_of_stay_days=7".

ICD-10 lookups cover the diagnoses our model is trained on. Unknown
codes pass through verbatim.
"""
from __future__ import annotations

from datetime import date


# Map ICD-10 codes to clinical names. Covers the diagnoses ReadmitIQ
# trains on; for production, you would use a real ICD-10 reference.
ICD10_NAMES: dict[str, str] = {
    "I50.9": "heart failure",
    "I21.9": "acute myocardial infarction",
    "I63.9": "ischemic stroke",
    "J18.9": "pneumonia",
    "J44.9": "chronic obstructive pulmonary disease (COPD)",
    "E11.9": "type 2 diabetes mellitus",
    "A41.9": "sepsis",
    "N17.9": "acute kidney injury",
    "I10": "essential hypertension",
}


def age_descriptor(age: int) -> str:
    """Map age in years to a clinical age descriptor."""
    if age < 18:
        return "pediatric"
    if age < 40:
        return "young adult"
    if age < 65:
        return "middle-aged"
    if age < 80:
        return "older adult"
    return "very elderly"


def los_descriptor(length_of_stay_days: int) -> str:
    """Map length of stay (days) to a qualitative descriptor."""
    if length_of_stay_days <= 0:
        return "same-day discharge"
    if length_of_stay_days <= 3:
        return "short stay"
    if length_of_stay_days <= 7:
        return "typical stay"
    if length_of_stay_days <= 14:
        return "extended stay"
    return "prolonged hospitalization"


def diagnosis_name(icd10_code: str | None) -> str:
    """Map an ICD-10 code to its clinical name, or pass through if unknown."""
    if not icd10_code:
        return "unspecified diagnosis"
    return ICD10_NAMES.get(icd10_code, icd10_code)


def compose_query(
    age: int,
    sex: str,
    admission_date: str,
    discharge_date: str,
    primary_diagnosis: str | None,
) -> str:
    """
    Build a natural-language clinical query string from patient fields.

    Args:
        age: patient age in years
        sex: "F", "M", or "O"
        admission_date: ISO date string, e.g. "2024-06-15"
        discharge_date: ISO date string
        primary_diagnosis: ICD-10 code or None

    Returns:
        A query string suitable for embedding and similarity search.
    """
    los = (date.fromisoformat(discharge_date) - date.fromisoformat(admission_date)).days
    age_desc = age_descriptor(age)
    los_desc = los_descriptor(los)
    dx_name = diagnosis_name(primary_diagnosis)

    # Compose a clinician-sounding question. The phrasing is deliberate: it
    # mirrors how a discharge planner might describe a patient to a colleague.
    return (
        f"30-day hospital readmission risk in {age_desc} patient "
        f"with {dx_name} after {los_desc} ({los} days)"
    )
