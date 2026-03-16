@echo off
SETLOCAL ENABLEDELAYEDEXPANSION
title Scholarship Fairness ML System

echo ================================================
echo   Scholarship Fairness ML System
echo   Production-Grade ML + FastAPI + SQLite
echo ================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.11+
    pause & exit /b 1
)

:: Check if we're in a venv — activate or create one
if not defined VIRTUAL_ENV (
    if exist "venv\Scripts\activate.bat" (
        echo [INFO] Activating existing virtual environment...
        call venv\Scripts\activate.bat
    ) else (
        echo [INFO] Creating virtual environment...
        python -m venv venv
        call venv\Scripts\activate.bat
    )
) else (
    echo [INFO] Using existing virtual environment...
)

:: Install dependencies
echo [INFO] Installing/verifying dependencies...
python -m pip install -q --upgrade pip
pip install -q -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies. Check requirements.txt.
    pause & exit /b 1
)

:: Create storage dirs
if not exist "storage\models" mkdir storage\models
if not exist "storage\reports" mkdir storage\reports
if not exist "storage\logs" mkdir storage\logs

:: Auto-train model if no model file exists
if not exist "storage\models\model_pipeline.joblib" (
    echo.
    echo [INFO] No trained model found. Running initial training (this may take 1-2 minutes)...
    python -c "from app.ml.trainer import train_model; train_model(n_samples=2000); print('[INFO] Model trained successfully!')"
    if errorlevel 1 (
        echo [WARN] Auto-training failed. You can train via POST /v1/retrain after server starts.
    )
)

echo.
echo [INFO] Starting FastAPI server...
echo [INFO] API Docs:    http://localhost:8000/docs
echo [INFO] Health:      http://localhost:8000/v1/health
echo [INFO] Ready:       http://localhost:8000/v1/ready
echo [INFO] Frontend:    Open frontend\index.html in your browser
echo [INFO] Press Ctrl+C to stop
echo.

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-level info

ENDLOCAL
