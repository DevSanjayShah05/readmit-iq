"""
Tests for the cohort service.

These tests seed a small known dataset and verify the service returns the
right slices. We're testing the *service* — the DAO is exercised
incidentally, but we're not retesting the DAO's own behavior.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from readmit_iq.dao.patient_dao import PatientDAO
from readmit_iq.models import Patient
from readmit_iq.services.cohort_service import CohortService, ELDERLY_AGE_THRESHOLD


def _make_patient(
    mrn: str,
    age: int = 70,
    diagnosis: str | None = "I50.9",
    admission: date = date(2024, 6, 15),
    readmitted: bool = False,
) -> Patient:
    """Build a Patient with sensible defaults; override only what each test needs."""
    return Patient(
        mrn=mrn,
        age=age,
        sex="M",
        admission_date=admission,
        discharge_date=admission + timedelta(days=5),
        primary_diagnosis=diagnosis,
        readmitted_30d=readmitted,
    )


def test_heart_failure_cohort_includes_only_hf(clean_patient_table) -> None:
    """heart_failure_cohort should return I50.9 patients only."""
    dao = PatientDAO()
    dao.insert(_make_patient("HF1", diagnosis="I50.9"))
    dao.insert(_make_patient("HF2", diagnosis="I50.9"))
    dao.insert(_make_patient("PN1", diagnosis="J18.9"))
    dao.insert(_make_patient("MI1", diagnosis="I21.9"))

    service = CohortService()
    cohort = service.heart_failure_cohort()
    mrns = {p.mrn for p in cohort}
    assert mrns == {"HF1", "HF2"}


def test_cardiovascular_cohort_includes_hf_mi_stroke(clean_patient_table) -> None:
    """cardiovascular_cohort should include all three CV diagnoses."""
    dao = PatientDAO()
    dao.insert(_make_patient("HF1", diagnosis="I50.9"))
    dao.insert(_make_patient("MI1", diagnosis="I21.9"))
    dao.insert(_make_patient("ST1", diagnosis="I63.9"))
    dao.insert(_make_patient("PN1", diagnosis="J18.9"))

    service = CohortService()
    cohort = service.cardiovascular_cohort()
    mrns = {p.mrn for p in cohort}
    assert mrns == {"HF1", "MI1", "ST1"}


def test_recent_admissions_respects_window(clean_patient_table) -> None:
    """recent_admissions(days=30) should exclude admissions older than 30 days."""
    dao = PatientDAO()
    today = date.today()
    dao.insert(_make_patient("RECENT", admission=today - timedelta(days=5)))
    dao.insert(_make_patient("OLD", admission=today - timedelta(days=120)))

    service = CohortService()
    recent = service.recent_admissions(days=30)
    mrns = {p.mrn for p in recent}
    assert mrns == {"RECENT"}


def test_elderly_cohort_threshold(clean_patient_table) -> None:
    """elderly_cohort should include only age >= ELDERLY_AGE_THRESHOLD."""
    dao = PatientDAO()
    dao.insert(_make_patient("OLDER", age=ELDERLY_AGE_THRESHOLD))
    dao.insert(_make_patient("OLDEST", age=ELDERLY_AGE_THRESHOLD + 20))
    dao.insert(_make_patient("YOUNGER", age=ELDERLY_AGE_THRESHOLD - 1))

    service = CohortService()
    cohort = service.elderly_cohort()
    mrns = {p.mrn for p in cohort}
    assert mrns == {"OLDER", "OLDEST"}


def test_summarize_cohort_computes_rate(clean_patient_table) -> None:
    """summarize_cohort should compute n_patients, n_readmitted, and the rate."""
    dao = PatientDAO()
    dao.insert(_make_patient("A", readmitted=True))
    dao.insert(_make_patient("B", readmitted=True))
    dao.insert(_make_patient("C", readmitted=False))
    dao.insert(_make_patient("D", readmitted=False))

    service = CohortService()
    cohort = service.heart_failure_cohort()
    summary = service.summarize_cohort("hf_test", cohort)
    assert summary.n_patients == 4
    assert summary.n_readmitted == 2
    assert summary.readmit_rate == pytest.approx(0.5)


def test_summarize_empty_cohort_does_not_divide_by_zero(clean_patient_table) -> None:
    """An empty cohort should produce a 0.0 rate, not a crash."""
    service = CohortService()
    summary = service.summarize_cohort("empty", [])
    assert summary.n_patients == 0
    assert summary.readmit_rate == 0.0
