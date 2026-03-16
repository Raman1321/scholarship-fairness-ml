"""Dataset loading — from uploaded CSV or synthetic fallback."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


# ── Synthetic generator (used when no CSV is uploaded) ───────────────────────

def generate_dataset(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic scholarship eligibility dataset as fallback."""
    rng = np.random.default_rng(random_state)

    sgpa = rng.normal(loc=7.2, scale=1.4, size=n_samples).clip(4.0, 10.0)
    jee_score = rng.normal(loc=160, scale=55, size=n_samples).clip(0, 360).astype(int)
    marks_12 = rng.normal(loc=75, scale=12, size=n_samples).clip(35, 100)
    attendance = rng.normal(loc=80, scale=12, size=n_samples).clip(40, 100)
    gender = rng.integers(0, 2, size=n_samples)

    composite = (
        0.35 * (sgpa / 10)
        + 0.30 * (jee_score / 360)
        + 0.25 * (marks_12 / 100)
        + 0.10 * (attendance / 100)
    )
    base_eligible = np.where(gender == 1, composite >= 0.52, composite >= 0.58)
    noise = rng.random(n_samples) < 0.05
    eligible = np.where(noise, ~base_eligible, base_eligible).astype(int)

    return pd.DataFrame({
        "sgpa": sgpa.round(2),
        "jee_score": jee_score,
        "marks_12": marks_12.round(2),
        "attendance": attendance.round(2),
        "gender": gender,
        "eligible": eligible,
    })


# ── CSV loader ────────────────────────────────────────────────────────────────

# Maps common user column name variants → standard internal names
_COLUMN_MAP = {
    "sgpa": "sgpa", "gpa": "sgpa", "semester_gpa": "sgpa", "cgpa": "sgpa",
    "jee_score": "jee_score", "jee": "jee_score", "jee_main": "jee_score",
    "entrance_score": "jee_score", "entrance": "jee_score",
    "marks_12": "marks_12", "class_12": "marks_12", "hsc_marks": "marks_12",
    "twelve_marks": "marks_12", "marks12": "marks_12", "percentage_12": "marks_12",
    "attendance": "attendance", "attendance_pct": "attendance",
    "attendance_percentage": "attendance",
    "gender": "gender", "sex": "gender",
    "eligible": "eligible", "scholarship": "eligible",
    "scholarship_awarded": "eligible", "awarded": "eligible",
    "is_eligible": "eligible", "result": "eligible",
}


def _derive_labels(df: pd.DataFrame) -> pd.Series:
    """Derive eligibility from composite score when no label column exists."""
    composite = (
        0.35 * (df["sgpa"] / 10)
        + 0.30 * (df["jee_score"] / 360)
        + 0.25 * (df["marks_12"] / 100)
        + 0.10 * (df["attendance"] / 100)
    )
    return (composite >= 0.55).astype(int)


def load_from_csv(csv_path: str, min_records: int = 30) -> pd.DataFrame:
    """
    Load and validate student data from an uploaded CSV file for ML training.

    Expected columns (flexible naming — common variants are auto-mapped):
        sgpa / gpa / cgpa           — Semester GPA (0–10)
        jee_score / jee / entrance  — JEE score (0–360)
        marks_12 / hsc_marks        — Class 12 percentage (0–100)
        attendance                  — Attendance % (0–100)
        gender / sex                — male / female / 0 / 1
        eligible (optional)         — 1/0, yes/no, true/false
                                      Auto-derived if missing.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")

    df_raw = pd.read_csv(csv_path)
    logger.info(f"CSV loaded: {len(df_raw)} rows, columns: {list(df_raw.columns)}")

    if len(df_raw) < min_records:
        raise ValueError(
            f"Only {len(df_raw)} rows in CSV — need at least {min_records} for training."
        )

    # Normalize column names
    df = df_raw.rename(
        columns={c: _COLUMN_MAP[c.lower()] for c in df_raw.columns if c.lower() in _COLUMN_MAP}
    )

    # Check required feature columns
    required = ["sgpa", "jee_score", "marks_12", "attendance", "gender"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Your CSV has: {list(df_raw.columns)}"
        )

    # Encode gender
    if df["gender"].dtype == object:
        df["gender"] = (
            df["gender"].str.lower()
            .map({"female": 0, "f": 0, "0": 0, "male": 1, "m": 1, "1": 1, "other": 0})
            .fillna(0).astype(int)
        )
    else:
        df["gender"] = pd.to_numeric(df["gender"], errors="coerce").fillna(0).astype(int)

    # Eligibility label
    if "eligible" in df.columns:
        if df["eligible"].dtype == object:
            df["eligible"] = (
                df["eligible"].str.lower()
                .map({"yes": 1, "true": 1, "1": 1, "awarded": 1,
                      "no": 0, "false": 0, "0": 0})
                .fillna(0).astype(int)
            )
        else:
            df["eligible"] = pd.to_numeric(df["eligible"], errors="coerce").fillna(0).astype(int)
        logger.info(
            f"Using real labels — eligible: {df['eligible'].sum()}, "
            f"not eligible: {(df['eligible']==0).sum()}"
        )
    else:
        df["eligible"] = _derive_labels(df)
        logger.info(
            f"No 'eligible' column — derived labels. "
            f"Eligible: {df['eligible'].sum()}/{len(df)} ({df['eligible'].mean():.1%})"
        )

    # Clip to valid ranges and drop bad rows
    df["sgpa"] = pd.to_numeric(df["sgpa"], errors="coerce").clip(0, 10).fillna(7.0)
    df["jee_score"] = pd.to_numeric(df["jee_score"], errors="coerce").clip(0, 360).fillna(150).astype(int)
    df["marks_12"] = pd.to_numeric(df["marks_12"], errors="coerce").clip(0, 100).fillna(70.0)
    df["attendance"] = pd.to_numeric(df["attendance"], errors="coerce").clip(0, 100).fillna(75.0)

    df = df[["sgpa", "jee_score", "marks_12", "attendance", "gender", "eligible"]].dropna()
    logger.info(f"Training dataset ready: {len(df)} clean records")
    return df
