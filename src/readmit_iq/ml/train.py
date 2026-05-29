"""
Model training pipeline for the 30-day readmission classifier.

This script:
  1. Pulls all patients from the database via PatientDAO.
  2. Engineers features into a model-ready DataFrame.
  3. Splits into train/test sets with stratification on the label.
  4. Trains a RandomForestClassifier.
  5. Evaluates on the held-out test set (AUC, precision, recall).
  6. Logs feature importance.
  7. Saves the trained model to disk as a .joblib file.

Designed to be run as a script or imported as a function.

Usage:
    python -m readmit_iq.ml.train --output models/readmit_rf.joblib
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from readmit_iq.dao.patient_dao import PatientDAO
from readmit_iq.features.feature_engineering import patients_to_features
from readmit_iq.features.feature_spec import LABEL_COLUMN, feature_names
from readmit_iq.models import Patient


@dataclass(frozen=True)
class TrainingResult:
    """Outputs of one training run, useful for logging and tests."""

    n_train: int
    n_test: int
    auc: float
    precision: float
    recall: float
    average_precision: float
    feature_importance: dict[str, float]
    model_path: Path


def _load_patients() -> Sequence[Patient]:
    """Pull all patients from the database."""
    dao = PatientDAO()
    patients = dao.find_admissions_between(date(1900, 1, 1), date(2099, 12, 31))
    logger.info(f"Loaded {len(patients):,} patients from database")
    return patients


def _split_features_label(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the engineered DataFrame into X (features) and y (label)."""
    X = df[list(feature_names())]
    y = df[LABEL_COLUMN].astype(int)
    return X, y


def train_model(
    patients: Sequence[Patient],
    output_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
) -> TrainingResult:
    """
    Train a RandomForestClassifier on the given patient cohort.

    Args:
        patients: training cohort.
        output_path: where to save the trained model (.joblib).
        test_size: fraction of data held out for evaluation.
        random_state: reproducibility seed.
        n_estimators: number of trees in the forest.

    Returns:
        TrainingResult with metrics, feature importance, and model path.
    """
    if len(patients) < 30:
        raise ValueError(
            f"Need at least 30 patients to train; got {len(patients)}. "
            "Seed the database first with: python -m readmit_iq.scripts.seed_data --count 1000 --truncate"
        )

    # 1. Engineer features
    df = patients_to_features(patients, include_label=True)
    X, y = _split_features_label(df)
    logger.info(f"Feature matrix: {X.shape}, label positive rate: {y.mean():.1%}")

    # 2. Train/test split. stratify=y preserves the readmission rate in both
    # halves — critical when the positive class is the minority.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

    # 3. Train. class_weight='balanced' tells the model to weight rare-class
    # examples more heavily, which improves recall on minority classes.
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,  # use all CPU cores
    )
    model.fit(X_train, y_train)
    logger.info(f"Trained RandomForestClassifier with {n_estimators} trees")

    # 4. Evaluate on the held-out test set
    # predict_proba returns shape (n, 2); column 1 is P(class=1).
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = float(roc_auc_score(y_test, y_pred_proba))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    avg_precision = float(average_precision_score(y_test, y_pred_proba))

    logger.success(
        f"Eval — AUC: {auc:.3f} | Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | Avg Precision: {avg_precision:.3f}"
    )

    # 5. Feature importance. Random forests report this directly; higher
    # values mean the feature contributed more to classification decisions.
    importance = dict(zip(X.columns, model.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info(f"Top 5 features: {[(f, round(s, 3)) for f, s in top_features]}")

    # 6. Save the trained model
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    logger.success(f"Model saved to {output_path}")

    return TrainingResult(
        n_train=len(X_train),
        n_test=len(X_test),
        auc=auc,
        precision=precision,
        recall=recall,
        average_precision=avg_precision,
        feature_importance={k: float(v) for k, v in importance.items()},
        model_path=output_path,
    )


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Train the readmission classifier.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/readmit_rf.joblib"),
        help="Where to save the trained model (default: models/readmit_rf.joblib)",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=100)
    args = parser.parse_args()

    patients = _load_patients()
    train_model(
        patients=patients,
        output_path=args.output,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
    )


if __name__ == "__main__":
    main()
