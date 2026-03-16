"""SHAP explanation route: GET /explain/{student_id}."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.security import get_current_user
from app.db.database import get_db
from app.db.models import Student, Prediction
from app.ml.explainability import async_explain
from app.schemas.schemas import ShapExplanation

router = APIRouter(tags=["Explainability"])


@router.get("/explain/{student_id}", response_model=ShapExplanation)
async def explain_student(
    student_id: int,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(get_current_user),
):
    result = await db.execute(select(Student).where(Student.id == student_id))
    student = result.scalar_one_or_none()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    try:
        explanation = await async_explain(
            sgpa=student.sgpa,
            jee_score=student.jee_score,
            marks_12=student.marks_12,
            attendance=student.attendance,
            gender=student.gender,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    logger.info(f"SHAP explanation for student_id={student_id}")

    return ShapExplanation(
        student_id=student_id,
        eligible=explanation["eligible"],
        probability=explanation["probability"],
        feature_contributions=explanation["feature_contributions"],
        base_value=explanation["base_value"],
        interpretation=explanation["interpretation"],
    )
