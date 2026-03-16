"""Inference service for scholarship eligibility prediction."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

from app.ml.trainer import load_model, FEATURE_NAMES

# Module-level cache
_cached_pipeline = None
_cached_metadata: Optional[Dict] = None


def _get_pipeline():
    global _cached_pipeline, _cached_metadata
    if _cached_pipeline is None:
        _cached_pipeline, _cached_metadata = load_model()
        logger.info("Model loaded and cached")
    return _cached_pipeline, _cached_metadata


def invalidate_cache():
    """Call this after retraining to force model reload."""
    global _cached_pipeline, _cached_metadata
    _cached_pipeline = None
    _cached_metadata = None


def predict(
    sgpa: float,
    jee_score: int,
    marks_12: float,
    attendance: float,
    gender: str,
) -> Dict[str, Any]:
    """Run inference on a single student record."""
    pipeline, metadata = _get_pipeline()

    gender_encoded = 0 if gender.lower() == "female" else 1

    input_df = pd.DataFrame([{
        "sgpa": sgpa,
        "jee_score": jee_score,
        "marks_12": marks_12,
        "attendance": attendance,
        "gender": gender_encoded,
    }], columns=FEATURE_NAMES)

    prob = float(pipeline.predict_proba(input_df)[0, 1])
    eligible = prob >= 0.5

    if prob >= 0.75:
        confidence = "High"
    elif prob >= 0.55:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "eligible": eligible,
        "probability": round(prob, 4),
        "confidence": confidence,
        "model_version": metadata.get("model_version", "v1.0"),
    }


async def async_predict(
    sgpa: float,
    jee_score: int,
    marks_12: float,
    attendance: float,
    gender: str,
) -> Dict[str, Any]:
    return await asyncio.to_thread(predict, sgpa, jee_score, marks_12, attendance, gender)
