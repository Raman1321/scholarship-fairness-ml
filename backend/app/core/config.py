from __future__ import annotations

import os
from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )

    # App
    APP_NAME: str = "ScholarshipFairnessML"
    APP_ENV: str = "production"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # Security
    SECRET_KEY: str  # Required from environment
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # Database
    DATABASE_URL: str

    # CORS
    ALLOWED_ORIGINS: str = "http://localhost,http://localhost:3000,http://127.0.0.1"

    @property
    def origins_list(self) -> List[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",")]

    # Storage paths
    MODEL_DIR: str = "storage/models"
    REPORTS_DIR: str = "storage/reports"

    @property
    def model_path(self) -> str:
        return os.path.join(self.MODEL_DIR, "model_pipeline.joblib")

    @property
    def metadata_path(self) -> str:
        return os.path.join(self.MODEL_DIR, "model_metadata.json")


@lru_cache
def get_settings() -> Settings:
    return Settings()
