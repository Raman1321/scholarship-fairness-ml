"""Fairness report route: GET /fairness-report."""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.security import get_current_user
from app.db.database import get_db
from app.db.models import FairnessReport
from app.ml.fairness import async_compute_fairness
from app.schemas.schemas import FairnessReportOut

router = APIRouter(tags=["Fairness"])


@router.get("/fairness-report", response_model=FairnessReportOut)
async def get_fairness_report(
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(get_current_user),
):
    try:
        report = await async_compute_fairness()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    # Persist to DB
    db_report = FairnessReport(
        model_version=report["model_version"],
        metrics=report["metrics"],
        mitigation_metrics=None,
    )
    db.add(db_report)
    await db.flush()
    await db.commit()

    logger.info(
        f"Fairness report generated: fair={report['overall_fair']} "
        f"n={report['n_samples']}"
    )

    return FairnessReportOut(
        model_version=report["model_version"],
        protected_attribute=report["protected_attribute"],
        n_samples=report["n_samples"],
        metrics=report["metrics"],
        overall_fair=report["overall_fair"],
        mitigation_applied=report.get("mitigation_applied", False),
        mitigation_metrics=report.get("mitigation_metrics"),
        generated_at=datetime.now(timezone.utc),
    )
