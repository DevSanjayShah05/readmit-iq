"""
Cohort service.

A cohort is a clinically-meaningful subset of patients. The service layer
defines named cohorts in *business* terms (heart failure, elderly, recent
admissions) and uses the DAO to fetch them. Callers — the ML training
pipeline, the analytics dashboard, the API — ask for cohorts by name and
don't have to know which diagnosis codes belong to "heart failure" or what
"elderly" means today.

If clinical definitions change (e.g., the threshold for "elderly" moves
from 65 to 70 in 2026), we change the constants here. No downstream code
needs to know.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Sequence

from loguru import logger

from readmit_iq.dao.patient_dao import PatientDAO
from readmit_iq.models import Patient

# Clinical cohort definitions. Each is a *set* of ICD-10 codes that map
# to a clinically-meaningful concept. In a real project, this would be
# maintained by clinical informatics; here it's hand-curated for the
# diagnoses our seeder produces.
HEART_FAILURE_CODES = ("I50.9",)
COPD_CODES = ("J44.9",)
PNEUMONIA_CODES = ("J18.9",)
DIABETES_CODES = ("E11.9",)
CARDIOVASCULAR_CODES = ("I50.9", "I21.9", "I63.9")  # HF + MI + stroke

# Demographic cohort thresholds.
ELDERLY_AGE_THRESHOLD = 65


@dataclass(frozen=True)
class CohortSummary:
    """A summary of a cohort, used for logging and quick inspection."""

    name: str
    n_patients: int
    n_readmitted: int
    readmit_rate: float


def _summarize(name: str, patients: Sequence[Patient]) -> CohortSummary:
    """Compute summary statistics for a cohort."""
    n = len(patients)
    if n == 0:
        return CohortSummary(name=name, n_patients=0, n_readmitted=0, readmit_rate=0.0)
    n_readmitted = sum(1 for p in patients if p.readmitted_30d)
    return CohortSummary(
        name=name,
        n_patients=n,
        n_readmitted=n_readmitted,
        readmit_rate=n_readmitted / n,
    )


class CohortService:
    """High-level cohort queries built on top of PatientDAO."""

    def __init__(self, dao: PatientDAO | None = None) -> None:
        self.dao = dao or PatientDAO()

    def heart_failure_cohort(
        self,
        start: date | None = None,
        end: date | None = None,
    ) -> Sequence[Patient]:
        """All patients admitted with a heart-failure diagnosis."""
        return self.dao.find_by_diagnoses(HEART_FAILURE_CODES, start=start, end=end)

    def cardiovascular_cohort(
        self,
        start: date | None = None,
        end: date | None = None,
    ) -> Sequence[Patient]:
        """Patients with HF, MI, or stroke. The 'big three' cardiovascular admissions."""
        return self.dao.find_by_diagnoses(CARDIOVASCULAR_CODES, start=start, end=end)

    def recent_admissions(self, days: int = 90) -> Sequence[Patient]:
        """All patients admitted in the last `days` days."""
        end = date.today()
        start = end - timedelta(days=days)
        return self.dao.find_admissions_between(start, end)

    def elderly_cohort(self) -> Sequence[Patient]:
        """All patients aged 65 or older. Computed in Python from a full pull."""
        # This is intentionally inefficient (pulls everyone, filters in Python)
        # to illustrate when to push filtering down to SQL vs not. For 1000
        # rows it's fine; for 10M it would be wrong, and we'd add a DAO method
        # that filters by age range at the database level.
        all_patients = self.dao.find_admissions_between(date(1900, 1, 1), date.today())
        return [p for p in all_patients if p.age >= ELDERLY_AGE_THRESHOLD]

    def summarize_cohort(self, name: str, patients: Sequence[Patient]) -> CohortSummary:
        """Compute summary stats for any cohort. Useful for logs and dashboards."""
        summary = _summarize(name, patients)
        logger.info(
            f"Cohort '{summary.name}': n={summary.n_patients:,}, "
            f"readmitted={summary.n_readmitted:,} ({summary.readmit_rate:.1%})"
        )
        return summary
