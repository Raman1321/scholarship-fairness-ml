"""Prediction route: POST /predict."""
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.rate_limit import limiter

from app.core.security import get_current_user
from app.db.database import get_db
from app.db.models import Prediction, Student, AuditLog
from app.ml.predictor import async_predict
from app.ml.explainability import async_explain
from app.schemas.schemas import PredictRequest, PredictResponse, PredictionOut

router = APIRouter(tags=["Predictions"])


@router.post("/predict", response_model=PredictResponse)
async def predict_eligibility(
    payload: PredictRequest,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    try:
        result = await async_predict(
            sgpa=payload.sgpa,
            jee_score=payload.jee_score,
            marks_12=payload.marks_12,
            attendance=payload.attendance,
            gender=payload.gender,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    # Optional: get local SHAP values
    try:
        shap_result = await async_explain(
            sgpa=payload.sgpa,
            jee_score=payload.jee_score,
            marks_12=payload.marks_12,
            attendance=payload.attendance,
            gender=payload.gender,
        )
        shap_values = shap_result.get("feature_contributions")
    except Exception as e:
        logger.warning(f"SHAP failed during predict: {e}")
        shap_values = None

    # Persist prediction record if student_id provided
    student_id = payload.student_id
    if student_id:
        pred_record = Prediction(
            student_id=student_id,
            eligible=result["eligible"],
            probability=result["probability"],
            model_version=result["model_version"],
            shap_values=shap_values,
        )
        db.add(pred_record)

        # Audit log
        log_entry = AuditLog(
            action="PREDICT",
            user_id=None,
            details={
                "student_id": student_id,
                "eligible": result["eligible"],
                "probability": result["probability"],
                "username": user.get("sub"),
            },
        )
        db.add(log_entry)
        await db.flush()
        await db.commit()

    eligible_str = "✅ Eligible for Scholarship" if result["eligible"] else "❌ Not Eligible"
    msg = f"{eligible_str} — Probability: {result['probability']:.1%} ({result['confidence']} confidence)"

    logger.info(
        f"Prediction: student_id={student_id} eligible={result['eligible']} "
        f"prob={result['probability']:.4f} user={user.get('sub')}"
    )

    return PredictResponse(
        student_id=student_id,
        eligible=result["eligible"],
        probability=result["probability"],
        confidence=result["confidence"],
        model_version=result["model_version"],
        shap_values=shap_values,
        message=msg,
    )


@router.get("/predictions", response_model=List[PredictionOut])
async def list_predictions(
    skip: int = 0,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(get_current_user),
):
    result = await db.execute(
        select(Prediction).order_by(Prediction.created_at.desc()).offset(skip).limit(limit)
    )
    return result.scalars().all()
