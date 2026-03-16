"""CSV training data upload route: POST /upload-training-data."""
import os
import shutil

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from loguru import logger

from app.core.security import require_role
from app.ml.trainer import UPLOADED_CSV_PATH
from app.ml.data_generator import load_from_csv

router = APIRouter(tags=["Model Management"])


@router.post("/upload-training-data")
async def upload_training_data(
    file: UploadFile = File(..., description="CSV file with student training data"),
    user: dict = Depends(require_role("admin", "analyst")),
):
    """
    Upload a CSV file of student records for ML model training.

    **Required columns** (flexible naming — common variants are auto-mapped):
    - `sgpa` / `gpa` / `cgpa` — Semester GPA (0–10)
    - `jee_score` / `jee` / `entrance_score` — JEE score (0–360)
    - `marks_12` / `hsc_marks` / `percentage_12` — Class 12 percentage (0–100)
    - `attendance` — Attendance percentage (0–100)
    - `gender` / `sex` — male/female or 1/0

    **Optional:**
    - `eligible` / `scholarship_awarded` / `is_eligible` — 1/0 or yes/no.
      If absent, eligibility is derived automatically from the composite score.

    After uploading, call **POST /v1/retrain** to train the model on this data.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted")

    # Save to a temp location first, validate, then move to final path
    tmp_path = UPLOADED_CSV_PATH + ".tmp"
    os.makedirs(os.path.dirname(UPLOADED_CSV_PATH), exist_ok=True)

    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Validate the CSV before accepting it
        df = load_from_csv(tmp_path, min_records=10)

        # All good — promote to final path
        shutil.move(tmp_path, UPLOADED_CSV_PATH)

        logger.info(
            f"Training CSV uploaded by {user.get('sub')}: "
            f"{len(df)} records, columns={list(df.columns)}"
        )

    except ValueError as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

    eligibility_rate = float(df["eligible"].mean())
    return {
        "status": "success",
        "filename": file.filename,
        "records": len(df),
        "columns": list(df.columns),
        "eligibility_rate": round(eligibility_rate, 3),
        "message": (
            f"Uploaded {len(df)} student records "
            f"({eligibility_rate:.1%} eligible). "
            "Call POST /v1/retrain to train the model on this data."
        ),
    }


@router.delete("/upload-training-data")
async def delete_training_data(
    user: dict = Depends(require_role("admin")),
):
    """Remove the uploaded CSV so the next retrain falls back to synthetic data."""
    if not os.path.exists(UPLOADED_CSV_PATH):
        raise HTTPException(status_code=404, detail="No uploaded training data found")
    os.remove(UPLOADED_CSV_PATH)
    logger.info(f"Training CSV deleted by {user.get('sub')}")
    return {"status": "success", "message": "Training data removed. Next retrain will use synthetic data."}


@router.get("/upload-training-data/info")
async def training_data_info(
    _user: dict = Depends(require_role("admin", "analyst")),
):
    """Check whether a training CSV is currently uploaded."""
    if not os.path.exists(UPLOADED_CSV_PATH):
        return {"uploaded": False, "message": "No CSV uploaded — retrain uses synthetic data"}

    df = pd.read_csv(UPLOADED_CSV_PATH)
    size_kb = os.path.getsize(UPLOADED_CSV_PATH) / 1024
    return {
        "uploaded": True,
        "records": len(df),
        "columns": list(df.columns),
        "size_kb": round(size_kb, 1),
        "path": UPLOADED_CSV_PATH,
    }
