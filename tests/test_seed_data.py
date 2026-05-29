"""
Tests for the data seed script.

These tests verify the generator produces the right *shape* of data
(correct count, plausible distributions) rather than exact values, which
would change with the seed.
"""
from __future__ import annotations

from readmit_iq.dao.patient_dao import PatientDAO
from readmit_iq.scripts.seed_data import seed_patients


def test_seed_inserts_correct_count(clean_patient_table) -> None:
    """Seeder should insert exactly the requested number of rows."""
    n_inserted = seed_patients(count=50, seed=1, truncate=False)
    assert n_inserted == 50
    assert PatientDAO().count() == 50


def test_seed_produces_realistic_readmit_rate(clean_patient_table) -> None:
    """Aggregate readmit rate should land in a plausible window (10%-30%)."""
    seed_patients(count=500, seed=42, truncate=False)
    dao = PatientDAO()
    readmitted = sum(
        1 for p in dao.find_admissions_between(__import__("datetime").date(2000, 1, 1),
                                                __import__("datetime").date(2099, 12, 31))
        if p.readmitted_30d
    )
    rate = readmitted / dao.count()
    assert 0.10 <= rate <= 0.30, f"Readmit rate {rate:.1%} out of expected band"


def test_seed_is_reproducible(clean_patient_table) -> None:
    """Same seed should produce the same data."""
    seed_patients(count=20, seed=99, truncate=False)
    dao = PatientDAO()
    first_mrns = {p.mrn for p in dao.find_admissions_between(
        __import__("datetime").date(2000, 1, 1),
        __import__("datetime").date(2099, 12, 31)
    )}
    first_ages = sorted(p.age for p in dao.find_admissions_between(
        __import__("datetime").date(2000, 1, 1),
        __import__("datetime").date(2099, 12, 31)
    ))

    # Truncate and reseed with same seed
    seed_patients(count=20, seed=99, truncate=True)
    second_mrns = {p.mrn for p in dao.find_admissions_between(
        __import__("datetime").date(2000, 1, 1),
        __import__("datetime").date(2099, 12, 31)
    )}
    second_ages = sorted(p.age for p in dao.find_admissions_between(
        __import__("datetime").date(2000, 1, 1),
        __import__("datetime").date(2099, 12, 31)
    ))
    assert first_mrns == second_mrns
    assert first_ages == second_ages
