"""
Data Access Object for the `patient` table.

This class is the *only* place in the application that writes SQL against
the patient table. All other code calls these methods; nobody outside this
file should ever do `cursor.execute('SELECT ... FROM patient ...')`.

This is the "anticorruption layer" between the application's Python world
(Patient objects) and the database's relational world (rows and columns).
"""
from __future__ import annotations

from datetime import date
from typing import Sequence

from loguru import logger

from readmit_iq.models import Patient
from readmit_iq.utils.db import get_connection


# The full set of columns we read back, in a fixed order. Defined once at
# module scope so insert / select / row-parsing all use the same shape.
_COLUMNS = (
    "id",
    "mrn",
    "age",
    "sex",
    "admission_date",
    "discharge_date",
    "primary_diagnosis",
    "readmitted_30d",
    "created_at",
)


def _row_to_patient(row: tuple) -> Patient:
    """Convert a database row (tuple) into a Patient object."""
    (
        id_,
        mrn,
        age,
        sex,
        admission_date,
        discharge_date,
        primary_diagnosis,
        readmitted_30d,
        created_at,
    ) = row
    return Patient(
        id=id_,
        mrn=mrn,
        age=age,
        sex=sex,
        admission_date=admission_date,
        discharge_date=discharge_date,
        primary_diagnosis=primary_diagnosis,
        readmitted_30d=readmitted_30d,
        created_at=created_at,
    )


class PatientDAO:
    """Read and write Patient records."""

    def insert(self, patient: Patient) -> Patient:
        """
        Insert a new patient. Returns a new Patient with id and created_at
        populated by the database.
        """
        sql = """
            INSERT INTO patient
                (mrn, age, sex, admission_date, discharge_date,
                 primary_diagnosis, readmitted_30d)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id, mrn, age, sex, admission_date, discharge_date,
                      primary_diagnosis, readmitted_30d, created_at;
        """
        params = (
            patient.mrn,
            patient.age,
            patient.sex,
            patient.admission_date,
            patient.discharge_date,
            patient.primary_diagnosis,
            patient.readmitted_30d,
        )
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
                conn.commit()
        if row is None:
            raise RuntimeError("Insert returned no row")
        logger.info(f"Inserted patient mrn={patient.mrn}")
        return _row_to_patient(row)

    def find_by_id(self, patient_id: int) -> Patient | None:
        """Look up a patient by primary key. Returns None if not found."""
        sql = f"SELECT {', '.join(_COLUMNS)} FROM patient WHERE id = %s;"
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (patient_id,))
                row = cur.fetchone()
        return _row_to_patient(row) if row else None

    def find_by_mrn(self, mrn: str) -> Patient | None:
        """Look up a patient by medical record number. Returns None if not found."""
        sql = f"SELECT {', '.join(_COLUMNS)} FROM patient WHERE mrn = %s;"
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (mrn,))
                row = cur.fetchone()
        return _row_to_patient(row) if row else None

    def find_admissions_between(self, start: date, end: date) -> Sequence[Patient]:
        """
        Return all patients admitted in the [start, end] date range (inclusive).
        Useful for cohort selection in our ML pipeline.
        """
        sql = f"""
            SELECT {', '.join(_COLUMNS)}
            FROM patient
            WHERE admission_date BETWEEN %s AND %s
            ORDER BY admission_date, id;
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (start, end))
                rows = cur.fetchall()
        return [_row_to_patient(r) for r in rows]

    def count(self) -> int:
        """Return the total number of patient rows. Used in healthchecks."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM patient;")
                row = cur.fetchone()
        return int(row[0]) if row else 0

    def delete_by_id(self, patient_id: int) -> bool:
        """Delete a patient by id. Returns True if a row was deleted."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM patient WHERE id = %s;", (patient_id,))
                deleted = cur.rowcount
                conn.commit()
        return deleted > 0
