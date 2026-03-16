#!/bin/bash
set -e

echo "================================================"
echo "  Scholarship Fairness ML System"
echo "  Production-Grade ML + FastAPI + PostgreSQL"
echo "================================================"
echo

# Create venv if not exists
if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

# Install dependencies
echo "[INFO] Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Create storage dirs
mkdir -p storage/models storage/reports storage/logs

echo
echo "[INFO] Starting server..."
echo "[INFO] API Docs: http://localhost:8000/docs"
echo "[INFO] Health:   http://localhost:8000/v1/health"
echo

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-level info
