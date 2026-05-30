"""Tests for the RAG query composer."""
from __future__ import annotations

import pytest

from readmit_iq.rag.query_composer import (
    age_descriptor,
    compose_query,
    diagnosis_name,
    los_descriptor,
)


# ---------- age_descriptor ----------


@pytest.mark.parametrize(
    "age,expected",
    [
        (5, "pediatric"),
        (17, "pediatric"),
        (18, "young adult"),
        (30, "young adult"),
        (40, "middle-aged"),
        (64, "middle-aged"),
        (65, "older adult"),
        (78, "older adult"),
        (80, "very elderly"),
        (95, "very elderly"),
    ],
)
def test_age_descriptor(age, expected):
    assert age_descriptor(age) == expected


# ---------- los_descriptor ----------


@pytest.mark.parametrize(
    "los,expected",
    [
        (0, "same-day discharge"),
        (-1, "same-day discharge"),
        (1, "short stay"),
        (3, "short stay"),
        (4, "typical stay"),
        (7, "typical stay"),
        (10, "extended stay"),
        (14, "extended stay"),
        (15, "prolonged hospitalization"),
        (30, "prolonged hospitalization"),
    ],
)
def test_los_descriptor(los, expected):
    assert los_descriptor(los) == expected


# ---------- diagnosis_name ----------


def test_diagnosis_name_known_code():
    assert diagnosis_name("I50.9") == "heart failure"
    assert diagnosis_name("J44.9") == "chronic obstructive pulmonary disease (COPD)"
    assert diagnosis_name("A41.9") == "sepsis"


def test_diagnosis_name_unknown_code_passes_through():
    """Unknown ICD-10 codes are passed through verbatim, not silently dropped."""
    assert diagnosis_name("Z51.11") == "Z51.11"


def test_diagnosis_name_none_returns_unspecified():
    assert diagnosis_name(None) == "unspecified diagnosis"


def test_diagnosis_name_empty_string_returns_unspecified():
    assert diagnosis_name("") == "unspecified diagnosis"


# ---------- compose_query ----------


def test_compose_query_elderly_hf_typical_stay():
    """An 78-year-old with heart failure, 7-day stay."""
    query = compose_query(
        age=78,
        sex="M",
        admission_date="2024-06-15",
        discharge_date="2024-06-22",
        primary_diagnosis="I50.9",
    )
    assert "older adult" in query
    assert "heart failure" in query
    assert "typical stay" in query
    assert "7 days" in query
    assert "30-day" in query


def test_compose_query_middle_aged_copd_short_stay():
    """A 45-year-old with COPD, 1-day stay."""
    query = compose_query(
        age=45,
        sex="F",
        admission_date="2024-06-15",
        discharge_date="2024-06-16",
        primary_diagnosis="J44.9",
    )
    assert "middle-aged" in query
    assert "COPD" in query  # acronym should survive
    assert "short stay" in query


def test_compose_query_very_elderly_sepsis_prolonged():
    """An 88-year-old with sepsis, 17-day stay."""
    query = compose_query(
        age=88,
        sex="F",
        admission_date="2024-06-15",
        discharge_date="2024-07-02",
        primary_diagnosis="A41.9",
    )
    assert "very elderly" in query
    assert "sepsis" in query
    assert "prolonged hospitalization" in query


def test_compose_query_no_diagnosis():
    """A patient with no diagnosis should still get a sensible query."""
    query = compose_query(
        age=30,
        sex="M",
        admission_date="2024-06-15",
        discharge_date="2024-06-17",
        primary_diagnosis=None,
    )
    assert "unspecified diagnosis" in query
    assert "young adult" in query


def test_compose_query_returns_natural_language():
    """The composed query should look like a clinical question, not feature dump."""
    query = compose_query(
        age=70,
        sex="M",
        admission_date="2024-06-15",
        discharge_date="2024-06-22",
        primary_diagnosis="I50.9",
    )
    # Should not contain feature-engineering artifacts
    assert "dx_" not in query
    assert "age_bucket_" not in query
    assert "=" not in query  # not "age=70"
    # Should contain clinical descriptors
    assert "patient" in query.lower()
