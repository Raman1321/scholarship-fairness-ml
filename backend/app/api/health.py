"""Health check routes."""
import os
from fastapi import APIRouter
from loguru import logger

from app.core.config import get_settings

settings = get_settings()
router = APIRouter(tags=["Health"])


@router.get("/health")
async def health():
    """Liveness probe."""
    return {"status": "healthy", "app": settings.APP_NAME}


@router.get("/ready")
async def readiness():
    """Readiness probe — checks model exists."""
    model_ready = os.path.exists(settings.model_path)
    return {
        "status": "ready" if model_ready else "not_ready",
        "model_loaded": model_ready,
        "model_path": settings.model_path,
    }
