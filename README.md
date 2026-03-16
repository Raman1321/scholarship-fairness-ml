# Scholarship Fairness ML System

> **Production-grade ML pipeline for scholarship eligibility prediction with fairness monitoring, SHAP explainability, and bias detection.**

## 🏗️ Architecture

```
Student → Web UI (Frontend)
        ↓
FastAPI Backend (Python 3.11)
        ↓
Data Validation (Pydantic v2)
        ↓
PostgreSQL / SQLite (SQLAlchemy async)
        ↓
ML Pipeline (XGBoost + cross-validation)
        ↓
Fairness Evaluation (fairlearn)
        ↓
SHAP Explanation (TreeExplainer)
        ↓
Return Result → UI
```

## 🚀 Quick Start (Local — No Docker Needed)

```powershell
# 1. Go to backend directory
cd c:\stitch\production_ready\backend

# 2. Create virtual environment and install
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# SQLite is used automatically when no PostgreSQL is configured
# 3. Start the server
$env:DATABASE_URL="sqlite+aiosqlite:///./scholarship.db"
$env:SECRET_KEY="local-dev-secret-change-me"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**OR** just double-click `run.bat`.

## 🐳 Docker (Full Stack with PostgreSQL)

```powershell
cd c:\stitch\production_ready
docker-compose up --build
```

| Service   | URL                              |
|-----------|----------------------------------|
| Frontend  | http://localhost                 |
| API Docs  | http://localhost:8000/docs       |
| Health    | http://localhost:8000/v1/health  |
| Readiness | http://localhost:8000/v1/ready   |

## 📡 API Endpoints

| Method | Endpoint                  | Auth | Description                        |
|--------|---------------------------|------|------------------------------------|
| POST   | `/auth/register`          | No   | Create account                     |
| POST   | `/auth/login`             | No   | Get JWT token                      |
| POST   | `/students/`              | JWT  | Add student record                 |
| GET    | `/students/`              | JWT  | List all students                  |
| POST   | `/predict`                | JWT  | Predict eligibility + SHAP         |
| GET    | `/explain/{student_id}`   | JWT  | Local SHAP explanation             |
| GET    | `/fairness-report`        | JWT  | Full fairness metrics              |
| POST   | `/retrain`                | JWT  | Retrain model (admin/analyst)      |
| GET    | `/health`                 | No   | Liveness probe                     |
| GET    | `/ready`                  | No   | Readiness probe                    |
| GET    | `/predictions`            | JWT  | Prediction history                 |

## 🔧 Testing the API

```powershell
# 1. Register + Login
$body = '{"username":"admin","password":"admin123","role":"admin"}'
Invoke-RestMethod -Uri "http://localhost:8000/auth/register" -Method Post -Body $body -ContentType "application/json"
$login = Invoke-RestMethod -Uri "http://localhost:8000/auth/login" -Method Post -Body $body -ContentType "application/json"
$token = $login.access_token
$headers = @{Authorization = "Bearer $token"}

# 2. Add a student
$student = '{"name":"Priya Sharma","sgpa":8.5,"jee_score":250,"marks_12":88.0,"attendance":92.0,"gender":"female"}'
Invoke-RestMethod -Uri "http://localhost:8000/students/" -Method Post -Body $student -ContentType "application/json" -Headers $headers

# 3. Predict
$predict = '{"sgpa":8.5,"jee_score":250,"marks_12":88.0,"attendance":92.0,"gender":"female","student_id":1}'
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $predict -ContentType "application/json" -Headers $headers

# 4. SHAP explanation
Invoke-RestMethod -Uri "http://localhost:8000/explain/1" -Headers $headers

# 5. Fairness report
Invoke-RestMethod -Uri "http://localhost:8000/fairness-report" -Headers $headers

# 6. Retrain
Invoke-RestMethod -Uri "http://localhost:8000/retrain?n_samples=2000" -Method Post -Headers $headers
```

## 📊 ML Features

| Feature    | Description              | Weight |
|------------|--------------------------|--------|
| SGPA       | Semester GPA (0–10)      | 35%    |
| JEE Score  | Entrance score (0–360)   | 30%    |
| Class 12th | Percentage (0–100)       | 25%    |
| Attendance | Attendance % (0–100)     | 10%    |
| Gender     | Protected attribute      | —      |

## ⚖️ Fairness Metrics

| Metric                        | Threshold | Description                              |
|-------------------------------|-----------|------------------------------------------|
| Demographic Parity Difference | ≤ 0.10    | Selection rate equality across genders   |
| Equalized Odds Difference     | ≤ 0.10    | TPR + FPR parity across groups           |
| Equal Opportunity Difference  | ≤ 0.10    | True positive rate parity                |

## 📁 Project Structure

```
production_ready/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app + lifespan
│   │   ├── core/
│   │   │   ├── config.py           # Pydantic-settings
│   │   │   ├── logging.py          # loguru
│   │   │   └── security.py         # JWT + bcrypt
│   │   ├── db/
│   │   │   ├── database.py         # Async SQLAlchemy
│   │   │   └── models.py           # ORM models
│   │   ├── schemas/schemas.py      # Pydantic v2 schemas
│   │   ├── api/
│   │   │   ├── auth.py             # Authentication
│   │   │   ├── students.py         # Student CRUD
│   │   │   ├── predictions.py      # Prediction + SHAP
│   │   │   ├── explanations.py     # SHAP endpoint
│   │   │   ├── fairness.py         # Fairness report
│   │   │   ├── retrain.py          # Model retraining
│   │   │   └── health.py           # Health checks
│   │   └── ml/
│   │       ├── data_generator.py   # Synthetic dataset
│   │       ├── trainer.py          # XGBoost training
│   │       ├── predictor.py        # Inference + cache
│   │       ├── fairness.py         # fairlearn metrics
│   │       └── explainability.py   # SHAP explanations
│   ├── run.bat / run.sh            # Local dev launchers
│   ├── Dockerfile                  # Multi-stage build
│   └── requirements.txt
├── frontend/
│   └── index.html                  # Rich SPA dashboard
├── nginx/default.conf              # Reverse proxy
├── docker-compose.yml              # Full stack
└── .github/workflows/ci.yml        # CI pipeline
```

## 🔒 Security

- JWT Bearer tokens (60-min expiry by default)
- bcrypt password hashing
- CORS configured via environment  
- Non-root Docker user
- Structured request-ID tracing
