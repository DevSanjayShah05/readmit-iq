"""
Tests for the PatientDAO.

These are integration tests — they exercise real SQL against a real
Postgres. The `clean_patient_table` fixture resets the table between
tests so they don't see each other's data.
"""
from __future__ import annotations

from datetime import date

import pytest

from readmit_iq.dao.patient_dao import PatientDAO
from readmit_iq.models import Patient


def _sample_patient(mrn: str = "MRN-001") -> Patient:
    """A reasonable default Patient for tests. Override mrn for unique inserts."""
    return Patient(
        mrn=mrn,
        age=65,
        sex="M",
        admission_date=date(2024, 1, 15),
        discharge_date=date(2024, 1, 22),
        primary_diagnosis="I50.9",
        readmitted_30d=False,
    )


def test_insert_returns_patient_with_id(clean_patient_table) -> None:
    """Insert should populate id and created_at on the returned object."""
    dao = PatientDAO()
    inserted = dao.insert(_sample_patient())
    assert inserted.id is not None
    assert inserted.created_at is not None
    assert inserted.mrn == "MRN-001"


def test_find_by_id_round_trip(clean_patient_table) -> None:
    """A patient we just inserted should be findable by id with identical data."""
    dao = PatientDAO()
    inserted = dao.insert(_sample_patient())
    found = dao.find_by_id(inserted.id)
    assert found is not None
    assert found.mrn == inserted.mrn
    assert found.age == inserted.age
    assert found.admission_date == inserted.admission_date


def test_find_by_id_missing_returns_none(clean_patient_table) -> None:
    """find_by_id on a nonexistent id should return None, not raise."""
    dao = PatientDAO()
    assert dao.find_by_id(99999) is None


def test_find_by_mrn(clean_patient_table) -> None:
    """Lookup by mrn should work the same as lookup by id."""
    dao = PatientDAO()
    dao.insert(_sample_patient(mrn="LOOKUP-ME"))
    found = dao.find_by_mrn("LOOKUP-ME")
    assert found is not None
    assert found.mrn == "LOOKUP-ME"


def test_count(clean_patient_table) -> None:
    """count() should reflect the number of rows."""
    dao = PatientDAO()
    assert dao.count() == 0
    dao.insert(_sample_patient(mrn="A"))
    dao.insert(_sample_patient(mrn="B"))
    dao.insert(_sample_patient(mrn="C"))
    assert dao.count() == 3


def test_find_admissions_between(clean_patient_table) -> None:
    """Range query should return only patients within the window, ordered."""
    dao = PatientDAO()
    # Three patients, one before the window, one inside, one after
    dao.insert(Patient(mrn="BEFORE", age=60, sex="F",
                       admission_date=date(2023, 12, 1),
                       discharge_date=date(2023, 12, 5),
                       primary_diagnosis=None, readmitted_30d=False))
    dao.insert(Patient(mrn="INSIDE", age=70, sex="M",
                       admission_date=date(2024, 1, 15),
                       discharge_date=date(2024, 1, 22),
                       primary_diagnosis=None, readmitted_30d=False))
    dao.insert(Patient(mrn="AFTER", age=55, sex="F",
                       admission_date=date(2024, 3, 1),
                       discharge_date=date(2024, 3, 5),
                       primary_diagnosis=None, readmitted_30d=False))
    in_window = dao.find_admissions_between(date(2024, 1, 1), date(2024, 1, 31))
    assert len(in_window) == 1
    assert in_window[0].mrn == "INSIDE"


def test_delete_by_id(clean_patient_table) -> None:
    """Delete should remove the row and return True; second delete returns False."""
    dao = PatientDAO()
    inserted = dao.insert(_sample_patient())
    assert dao.delete_by_id(inserted.id) is True
    assert dao.delete_by_id(inserted.id) is False
    assert dao.find_by_id(inserted.id) is None


def test_unique_mrn_constraint(clean_patient_table) -> None:
    """Inserting two patients with the same mrn should fail at the DB level."""
    import psycopg
    dao = PatientDAO()
    dao.insert(_sample_patient(mrn="DUPLICATE"))
    with pytest.raises(psycopg.errors.UniqueViolation):
        dao.insert(_sample_patient(mrn="DUPLICATE"))
