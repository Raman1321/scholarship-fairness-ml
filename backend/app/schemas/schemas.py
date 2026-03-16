"""Pydantic v2 schemas for all API requests and responses."""
from __future__ import annotations

from datetime import datetime
from tkinter import INSERT
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, ConfigDict


# ──────────────────────────────────────────────────────────────────────────────
# Auth
# ──────────────────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=80)
    password: str = Field(..., min_length=6)
    role: str = Field(default="analyst", pattern="^(admin|analyst|viewer)$")


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: Any
    username: str
    role: str
    is_active: bool
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str


class LoginRequest(BaseModel):
    username: str
    password: str


# ──────────────────────────────────────────────────────────────────────────────
# Student
# ──────────────────────────────────────────────────────────────────────────────

class StudentCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=150)
    sgpa: float = Field(..., ge=0.0, le=10.0, description="SGPA on 0–10 scale")
    jee_score: int = Field(..., ge=0, le=360, description="JEE score 0–360")
    marks_12: float = Field(..., ge=0.0, le=100.0, description="Class 12th percentage")
    attendance: float = Field(..., ge=0.0, le=100.0, description="Attendance percentage")
    gender: str = Field(..., pattern="^(male|female|other)$")

    @field_validator("gender")
    @classmethod
    def lowercase_gender(cls, v: str) -> str:
        return v.lower()

class StudentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    sgpa: float
    jee_score: int
    marks_12: float
    attendance: float
    gender: str
    created_at: datetime


# ──────────────────────────────────────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    sgpa: float = Field(..., ge=0.0, le=10.0)
    jee_score: int = Field(..., ge=0, le=360)
    marks_12: float = Field(..., ge=0.0, le=100.0)
    attendance: float = Field(..., ge=0.0, le=100.0)
    gender: str = Field(..., pattern="^(male|female|other)$")
    student_id: Optional[int] = Field(None, description="Link to an existing student record")

    @field_validator("gender")
    @classmethod
    def lowercase_gender(cls, v: str) -> str:
        return v.lower()


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    student_id: Optional[int]
    eligible: bool
    probability: float
    confidence: str           # "High / Medium / Low"
    model_version: str
    shap_values: Optional[Dict[str, float]] = None
    message: str


class PredictionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())
    id: int
    student_id: int
    eligible: bool
    probability: float
    model_version: str
    created_at: datetime


# ──────────────────────────────────────────────────────────────────────────────
# SHAP Explanation
# ──────────────────────────────────────────────────────────────────────────────

class ShapExplanation(BaseModel):
    student_id: int
    eligible: bool
    probability: float
    feature_contributions: Dict[str, float]
    base_value: float
    interpretation: str


# ──────────────────────────────────────────────────────────────────────────────
# Fairness Report
# ──────────────────────────────────────────────────────────────────────────────

class FairnessMetric(BaseModel):
    name: str
    value: float
    threshold: float
    passed: bool
    description: str


class FairnessReportOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    protected_attribute: str
    n_samples: int
    metrics: List[FairnessMetric]
    overall_fair: bool
    mitigation_applied: bool
    mitigation_metrics: Optional[List[FairnessMetric]] = None
    generated_at: datetime


# ──────────────────────────────────────────────────────────────────────────────
# Retrain
# ──────────────────────────────────────────────────────────────────────────────

class RetrainResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_version: str
    accuracy: float
    auc_roc: float
    f1_score: float
    cross_val_mean: float
    training_samples: int
    message: str


# ──────────────────────────────────────────────────────────────────────────────
# Generic API wrapper
# ──────────────────────────────────────────────────────────────────────────────

class APIResponse(BaseModel):
    status: str = "success"
    data: Any = None
    message: str = ""


