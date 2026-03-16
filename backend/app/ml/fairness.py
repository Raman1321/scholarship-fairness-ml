"""Fairness evaluation using fairlearn metrics."""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from loguru import logger

try:
    from fairlearn.metrics import (
        demographic_parity_difference,
        equalized_odds_difference,
        MetricFrame,
    )
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False
    logger.warning("fairlearn not available — fairness metrics will be limited")


from app.core.config import get_settings
from app.ml.trainer import load_model

settings = get_settings()

THRESHOLD = 0.10  # acceptable bias threshold


def _compute_fairness(test_data_path: str) -> Dict[str, Any]:
    """Compute fairness metrics from saved test data."""
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data not found: {test_data_path}")

    df = pd.read_csv(test_data_path)
    y_true = df["eligible"].values
    y_pred = df["predicted"].values
    sensitive = df["gender"].values  # 0=female, 1=male

    metrics = []

    if FAIRLEARN_AVAILABLE:
        # Demographic Parity Difference
        dpd = float(demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive))
        metrics.append({
            "name": "Demographic Parity Difference",
            "value": round(abs(dpd), 4),
            "threshold": THRESHOLD,
            "passed": abs(dpd) <= THRESHOLD,
            "description": "Selection rate difference between genders. Closer to 0 is fairer.",
        })

        # Equalized Odds Difference
        eod = float(equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive))
        metrics.append({
            "name": "Equalized Odds Difference",
            "value": round(abs(eod), 4),
            "threshold": THRESHOLD,
            "passed": abs(eod) <= THRESHOLD,
            "description": (
                "Max difference in TPR and FPR across groups. Closer to 0 is fairer."
            ),
        })

        # Equal Opportunity (TPR parity) via MetricFrame
        from sklearn.metrics import recall_score
        mf = MetricFrame(
            metrics=recall_score,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive,
        )
        tpr_diff = float(mf.difference())
        metrics.append({
            "name": "Equal Opportunity Difference",
            "value": round(abs(tpr_diff), 4),
            "threshold": THRESHOLD,
            "passed": abs(tpr_diff) <= THRESHOLD,
            "description": "Difference in true positive rates across gender groups.",
        })

        # Group-level selection rates
        select_rates = {}
        for g in [0, 1]:
            mask = sensitive == g
            label = "female" if g == 0 else "male"
            select_rates[label] = round(float(y_pred[mask].mean()), 4)

    else:
        # Fallback manual metrics
        female_mask = sensitive == 0
        male_mask = sensitive == 1
        dpd_val = abs(y_pred[female_mask].mean() - y_pred[male_mask].mean())
        metrics.append({
            "name": "Demographic Parity Difference",
            "value": round(float(dpd_val), 4),
            "threshold": THRESHOLD,
            "passed": float(dpd_val) <= THRESHOLD,
            "description": "Selection rate difference between genders.",
        })
        select_rates = {
            "female": round(float(y_pred[female_mask].mean()), 4),
            "male": round(float(y_pred[male_mask].mean()), 4),
        }

    overall_fair = all(m["passed"] for m in metrics)

    return {
        "metrics": metrics,
        "overall_fair": overall_fair,
        "selection_rates": select_rates,
        "n_samples": int(len(y_true)),
        "n_female": int((sensitive == 0).sum()),
        "n_male": int((sensitive == 1).sum()),
    }


def compute_fairness_report() -> Dict[str, Any]:
    """Full fairness evaluation pipeline."""
    _, metadata = load_model()
    test_data_path = metadata.get("test_data_path", "")
    result = _compute_fairness(test_data_path)

    result["model_version"] = metadata.get("model_version", "v1.0")
    result["protected_attribute"] = "gender"
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    result["mitigation_applied"] = False

    logger.info(
        f"Fairness report: overall_fair={result['overall_fair']}, "
        f"metrics={[m['name'] + '=' + str(m['value']) for m in result['metrics']]}"
    )
    return result


async def async_compute_fairness() -> Dict[str, Any]:
    return await asyncio.to_thread(compute_fairness_report)
