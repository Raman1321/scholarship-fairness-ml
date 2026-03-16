from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Integer, String, JSON, Enum as SAEnum,
)
from sqlalchemy.orm import DeclarativeBase, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    # Use String(36) for UUID — compatible with both SQLite and PostgreSQL
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(80), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(SAEnum("admin", "analyst", "viewer", name="user_role"), nullable=False, default="analyst")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    audit_logs = relationship("AuditLog", back_populates="user", cascade="all, delete-orphan")


class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(150), nullable=False)
    sgpa = Column(Float, nullable=False)
    jee_score = Column(Integer, nullable=False)
    marks_12 = Column(Float, nullable=False)
    attendance = Column(Float, nullable=False)
    gender = Column(String(20), nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    predictions = relationship("Prediction", back_populates="student", cascade="all, delete-orphan")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(Integer, ForeignKey("students.id", ondelete="CASCADE"), nullable=False)
    eligible = Column(Boolean, nullable=False)
    probability = Column(Float, nullable=False)
    model_version = Column(String(50), nullable=False, default="v1.0")
    shap_values = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    student = relationship("Student", back_populates="predictions")


class FairnessReport(Base):
    __tablename__ = "fairness_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_version = Column(String(50), nullable=False)
    metrics = Column(JSON, nullable=False)
    mitigation_metrics = Column(JSON, nullable=True)
    report_path = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    action = Column(String(100), nullable=False)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    details = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    user = relationship("User", back_populates="audit_logs")
