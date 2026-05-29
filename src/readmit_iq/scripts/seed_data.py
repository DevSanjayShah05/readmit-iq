"""
Seed the patient table with realistic-ish fake data for development.

Why this exists: we need data to develop against. The real eventual data
source is the Synthea synthetic patient generator (or MIMIC-III), but for
local iteration we want something fast and predictable. This script
generates ~1000-10000 patients with realistic age distributions, diagnosis
mixes, and readmission rates that vary by diagnosis — so an ML model
trained on this data has actual signal to learn.

Usage:
    python -m readmit_iq.scripts.seed_data --count 1000
    python -m readmit_iq.scripts.seed_data --count 5000 --seed 42 --truncate
"""

from __future__ import annotations

import argparse
import random
from datetime import date, timedelta

from faker import Faker
from loguru import logger

from readmit_iq.dao.patient_dao import PatientDAO
from readmit_iq.models import Patient
from readmit_iq.utils.db import get_connection

# Realistic-ish diagnosis distribution and per-diagnosis readmission rate.
# These numbers are loosely based on published readmission research; they're
# not authoritative, but they create the kind of signal a model can learn.
DIAGNOSES = [
    # (icd10_code, label, weight, readmit_rate)
    ("I50.9", "Heart failure", 0.20, 0.25),
    ("J44.9", "COPD", 0.15, 0.22),
    ("J18.9", "Pneumonia", 0.12, 0.17),
    ("E11.9", "Type 2 diabetes", 0.10, 0.18),
    ("N17.9", "Acute kidney injury", 0.08, 0.20),
    ("I63.9", "Stroke", 0.07, 0.15),
    ("I21.9", "Acute MI", 0.06, 0.13),
    ("K92.2", "GI bleed", 0.05, 0.14),
    ("A41.9", "Sepsis", 0.05, 0.21),
    ("Z51.5", "Palliative encounter", 0.03, 0.05),
    (None, "Unspecified", 0.09, 0.10),  # null diagnosis
]


def _pick_diagnosis(rng: random.Random) -> tuple[str | None, float]:
    """Pick a diagnosis weighted by frequency. Returns (code, readmit_rate)."""
    codes = [d[0] for d in DIAGNOSES]
    weights = [d[2] for d in DIAGNOSES]
    rates = {d[0]: d[3] for d in DIAGNOSES}
    code = rng.choices(codes, weights=weights, k=1)[0]
    return code, rates[code]


def _generate_patient(faker: Faker, rng: random.Random, mrn: str) -> Patient:
    """Generate one realistic-ish patient with diagnosis-correlated readmission risk."""
    # Age: skewed older (gamma-ish distribution: most patients 55-85)
    age = max(18, min(99, int(rng.gauss(67, 14))))

    # Sex: roughly balanced with a small "other" category
    sex = rng.choices(["M", "F", "O"], weights=[0.49, 0.49, 0.02])[0]

    # Admission: random date in the last 2 years
    days_ago = rng.randint(30, 730)
    admission = date.today() - timedelta(days=days_ago)

    # Length of stay: log-normal-ish, 2-14 days typically
    los = max(1, min(30, int(rng.lognormvariate(1.4, 0.6))))
    discharge = admission + timedelta(days=los)

    # Diagnosis + readmission probability driven by diagnosis
    diagnosis_code, base_readmit_rate = _pick_diagnosis(rng)

    # Age bumps the readmission rate: older patients readmit more
    age_factor = 1.0 + (age - 65) * 0.01  # +1% per year over 65
    adjusted_rate = max(0.02, min(0.6, base_readmit_rate * age_factor))
    readmitted = rng.random() < adjusted_rate

    return Patient(
        mrn=mrn,
        age=age,
        sex=sex,
        admission_date=admission,
        discharge_date=discharge,
        primary_diagnosis=diagnosis_code,
        readmitted_30d=readmitted,
    )


def seed_patients(count: int, seed: int = 42, truncate: bool = False) -> int:
    """
    Generate and insert `count` synthetic patients into the database.

    Args:
        count: number of patients to insert.
        seed: random seed for reproducibility. Same seed -> same data.
        truncate: if True, wipe the patient table before inserting.

    Returns:
        Number of rows inserted.
    """
    rng = random.Random(seed)
    faker = Faker()
    Faker.seed(seed)

    if truncate:
        logger.warning("Truncating patient table before seeding")
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE patient RESTART IDENTITY;")
                conn.commit()

    dao = PatientDAO()
    inserted = 0
    for i in range(count):
        # MRN: zero-padded so they sort nicely. SEED-### prefix so we can
        # distinguish seeded data from real data in the future.
        mrn = f"SEED-{i:08d}"
        try:
            dao.insert(_generate_patient(faker, rng, mrn))
            inserted += 1
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to insert mrn={mrn}: {exc}")
        if (i + 1) % 500 == 0:
            logger.info(f"  ... {i + 1:,} / {count:,} inserted")

    logger.success(
        f"Seeded {inserted:,} patients (readmit rate: {_readmit_rate(dao):.1%})"
    )
    return inserted


def _readmit_rate(dao: PatientDAO) -> float:
    """Quick observability: what fraction of patients are readmitted?"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT AVG(CASE WHEN readmitted_30d THEN 1.0 ELSE 0.0 END) FROM patient;"
            )
            row = cur.fetchone()
    return float(row[0]) if row and row[0] is not None else 0.0


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Seed the patient table with synthetic data for development."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="How many patients to insert (default 1000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--truncate", action="store_true", help="Wipe the table before inserting"
    )
    args = parser.parse_args()

    seed_patients(count=args.count, seed=args.seed, truncate=args.truncate)


if __name__ == "__main__":
    main()
