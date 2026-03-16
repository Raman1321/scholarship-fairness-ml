"""SHAP-based explainability service."""
from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available — explanations will be limited")

from app.ml.trainer import load_model, FEATURE_NAMES

# Cache explainer per model load
_cached_explainer = None
_cached_model_version: Optional[str] = None

FEATURE_DISPLAY = {
    "sgpa": "SGPA",
    "jee_score": "JEE Score",
    "marks_12": "Class 12th Marks",
    "attendance": "Attendance",
    "gender": "Gender",
}


def _get_explainer():
    global _cached_explainer, _cached_model_version
    pipeline, metadata = load_model()
    version = metadata.get("model_version", "v1.0")

    if _cached_explainer is None or _cached_model_version != version:
        classifier = pipeline.named_steps["classifier"]
        preprocessor = pipeline.named_steps["preprocessor"]

        # Use background dataset for SHAP
        from app.ml.data_generator import generate_dataset
        bg = generate_dataset(n_samples=200)
        X_bg = bg[FEATURE_NAMES]
        X_bg_transformed = preprocessor.transform(X_bg)

        try:
            _cached_explainer = shap.TreeExplainer(classifier, X_bg_transformed)
        except Exception:
            _cached_explainer = shap.KernelExplainer(
                classifier.predict_proba, X_bg_transformed[:50]
            )
        _cached_model_version = version
        logger.info(f"SHAP explainer initialized for {version}")

    return _cached_explainer, pipeline


def explain_prediction(
    sgpa: float,
    jee_score: int,
    marks_12: float,
    attendance: float,
    gender: str,
) -> Dict[str, Any]:
    """Compute SHAP local explanation for a single prediction."""
    if not SHAP_AVAILABLE:
        return _fallback_explanation(sgpa, jee_score, marks_12, attendance, gender)

    gender_encoded = 0 if gender.lower() == "female" else 1
    input_data = {
        "sgpa": sgpa,
        "jee_score": jee_score,
        "marks_12": marks_12,
        "attendance": attendance,
        "gender": gender_encoded,
    }
    input_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)

    try:
        explainer, pipeline = _get_explainer()
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]

        X_transformed = preprocessor.transform(input_df)
        shap_vals = explainer.shap_values(X_transformed)

        # For binary classification, take class-1 SHAP values
        if isinstance(shap_vals, list):
            vals = shap_vals[1][0]
        else:
            vals = shap_vals[0] if shap_vals.ndim > 1 else shap_vals

        # Map back to feature names (handle passthrough re-ordering)
        feature_contribs = {}
        for i, name in enumerate(FEATURE_NAMES):
            if i < len(vals):
                feature_contribs[FEATURE_DISPLAY[name]] = round(float(vals[i]), 4)

        base_value = float(explainer.expected_value)
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(base_value[1] if len(base_value) > 1 else base_value[0])

        prob = float(pipeline.predict_proba(input_df)[0, 1])
        eligible = prob >= 0.5

        top_positive = max(feature_contribs, key=lambda k: feature_contribs[k])
        top_negative = min(feature_contribs, key=lambda k: feature_contribs[k])
        interp = (
            f"This prediction was most positively influenced by {top_positive} "
            f"and most negatively by {top_negative}."
        )

        return {
            "eligible": eligible,
            "probability": round(prob, 4),
            "feature_contributions": feature_contribs,
            "base_value": round(base_value, 4),
            "interpretation": interp,
        }

    except Exception as e:
        logger.warning(f"SHAP failed ({e}), using fallback")
        return _fallback_explanation(sgpa, jee_score, marks_12, attendance, gender)


def _fallback_explanation(sgpa, jee_score, marks_12, attendance, gender) -> Dict[str, Any]:
    """Simple linear attribution fallback."""
    weights = {"SGPA": 0.35, "JEE Score": 0.30, "Class 12th Marks": 0.25, "Attendance": 0.10}
    norm = {
        "SGPA": sgpa / 10,
        "JEE Score": jee_score / 360,
        "Class 12th Marks": marks_12 / 100,
        "Attendance": attendance / 100,
    }
    contribs = {k: round((norm[k] - 0.5) * w, 4) for k, w in weights.items()}
    contribs["Gender"] = 0.0
    prob = sum([sgpa * 0.035, jee_score * 0.00083, marks_12 * 0.0025, attendance * 0.001]) / 1
    prob = max(0.1, min(0.95, prob))
    return {
        "eligible": prob >= 0.5,
        "probability": round(prob, 4),
        "feature_contributions": contribs,
        "base_value": 0.0,
        "interpretation": "Approximate attribution based on feature weights.",
    }


async def async_explain(
    sgpa: float,
    jee_score: int,
    marks_12: float,
    attendance: float,
    gender: str,
) -> Dict[str, Any]:
    return await asyncio.to_thread(explain_prediction, sgpa, jee_score, marks_12, attendance, gender)
