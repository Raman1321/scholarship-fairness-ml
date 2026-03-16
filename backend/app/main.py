"""FastAPI application entry point."""
from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from loguru import logger

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.core.rate_limit import limiter
from app.db.database import init_db

settings = get_settings()
setup_logging(settings.LOG_LEVEL)

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB tables, check model exists."""
    logger.info(f"Starting {settings.APP_NAME} ({settings.APP_ENV})")

    # Ensure storage dirs exist
    os.makedirs(settings.MODEL_DIR, exist_ok=True)
    os.makedirs(settings.REPORTS_DIR, exist_ok=True)

    # Initialize database
    try:
        await init_db()
    except Exception as e:
        logger.warning(f"DB init skipped (no database available): {e}")

    # Check if model exists, warn if not
    if not os.path.exists(settings.model_path):
        logger.warning("No trained model found! Call POST /v1/retrain to train the model.")

    yield

    logger.info("Shutting down application")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Scholarship Fairness ML API",
    description=(
        "Production-grade ML system for scholarship eligibility prediction "
        "with fairness monitoring, SHAP explainability, and bias detection."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Rate limiting middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# ── Middleware ─────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request ID injection
@app.middleware("http")
async def add_request_id(request: Request, call_next) -> Response:
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    with logger.contextualize(request_id=request_id):
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# ── Routes ─────────────────────────────────────────────────────────────────────

from app.api.health import router as health_router
from app.api.auth import router as auth_router
from app.api.students import router as students_router
from app.api.predictions import router as predictions_router
from app.api.explanations import router as explanations_router
from app.api.fairness import router as fairness_router
from app.api.retrain import router as retrain_router

app.include_router(health_router, prefix="/v1")
app.include_router(auth_router, prefix="/v1")
app.include_router(students_router, prefix="/v1")
app.include_router(predictions_router, prefix="/v1")
app.include_router(explanations_router, prefix="/v1")
app.include_router(fairness_router, prefix="/v1")
app.include_router(retrain_router, prefix="/v1")


@app.get("/", tags=["Root"])
async def root():
    return {
        "app": settings.APP_NAME,
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/v1/health",
        "api_base": "/api",
    }
