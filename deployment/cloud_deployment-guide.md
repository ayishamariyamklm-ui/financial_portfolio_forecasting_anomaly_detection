# Cloud Deployment Guide

## Financial Portfolio Forecasting & Anomaly Detection

This guide explains how to deploy the **Financial Portfolio Forecasting & Anomaly Detection** project to cloud platforms.

The project contains:

* FastAPI backend
* Streamlit dashboard
* Forecasting models
* Anomaly detection models
* Monitoring files
* Evaluation reports

---

## 1. Project Deployment Overview

Cloud deployment allows the project to be accessed through public URLs instead of running only on a local computer. This makes the project stronger for portfolio presentation because recruiters can view the API documentation, dashboard, model outputs, and monitoring results online.

---

### 1.1 Main Applications

| Application      | File                       | Framework | Purpose                                     |
| ---------------- | -------------------------- | --------- | ------------------------------------------- |
| API Backend      | `api/main.py`              | FastAPI   | Forecasting and anomaly detection endpoints |
| Dashboard        | `dashboard/app.py`         | Streamlit | Visual dashboard for project outputs        |
| Inference Script | `src/inference/predict.py` | Python    | Run local forecast and anomaly inference    |
| API Runner       | `scripts/run_api.py`       | Uvicorn   | Local API startup script                    |
| Dashboard Runner | `scripts/run_dashboard.py` | Streamlit | Local dashboard startup script              |

---

## 2. Important Deployment Files

| File                        | Purpose                             |
| --------------------------- | ----------------------------------- |
| `deployment/render.yaml`    | Render Blueprint configuration      |
| `railway.json`              | Railway deployment configuration    |
| `Dockerfile`                | Container deployment configuration  |
| `docker-compose.yml`        | Local multi-service container setup |
| `requirements.txt`          | Python dependencies                 |
| `.env.example`              | Environment variable template       |
| `cloud_deployment_guide.md` | Cloud deployment instructions       |
| `production_notes.md`       | Production readiness notes          |

---

## 3. Pre-Deployment Checklist

Before deploying, make sure the following files and folders exist:

```text
api/main.py
dashboard/app.py
requirements.txt
Dockerfile
data/raw/financial_portfolio_data.csv
models/
outputs/
reports/
monitoring/
```

---

### 3.1 Run Tests Locally

Before deploying, run all tests:

```bash
pytest tests -v
```

---

### 3.2 Run API Locally

Start the FastAPI backend locally:

```bash
python scripts/run_api.py --reload
```

Open the API documentation:

```text
http://127.0.0.1:8000/docs
```

---

### 3.3 Run Dashboard Locally

Start the Streamlit dashboard locally:

```bash
python scripts/run_dashboard.py
```

Open the dashboard:

```text
http://localhost:8501
```

---

## 4. Recommended Local Run Order

Use this order before cloud deployment:

```bash
python scripts/run_data_pipeline.py
python scripts/run_all_models.py
python scripts/evaluate_models.py
python src/inference/predict.py
python scripts/run_api.py --reload
python scripts/run_dashboard.py
```

If the API and dashboard work locally, then deploy to cloud.

---

## 5. Environment Variables

Use these variables in cloud platforms such as Render or Railway.

| Variable         | Example Value                                       | Purpose                |
| ---------------- | --------------------------------------------------- | ---------------------- |
| `PYTHON_VERSION` | `3.11.9`                                            | Python runtime         |
| `ENVIRONMENT`    | `production`                                        | Deployment environment |
| `PROJECT_NAME`   | `financial-portfolio-forecasting-anomaly-detection` | Project name           |
| `LOG_LEVEL`      | `INFO`                                              | Logging level          |
| `DATA_PATH`      | `data/raw/financial_portfolio_data.csv`             | Dataset path           |
| `MODEL_DIR`      | `models`                                            | Model artifact folder  |
| `OUTPUT_DIR`     | `outputs`                                           | Output folder          |
| `REPORTS_DIR`    | `reports`                                           | Reports folder         |
| `MONITORING_DIR` | `monitoring`                                        | Monitoring folder      |

---

## 6. Deploy FastAPI on Render

Render can be used to deploy the FastAPI backend as a web service.

---

### 6.1 Render Service Type

Use the following service type:

```text
Web Service
```

---

### 6.2 Build Command

Use this build command:

```bash
pip install --upgrade pip && pip install -r requirements.txt
```

---

### 6.3 Start Command

Use this start command:

```bash
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

---

### 6.4 Health Check Path

Use this health check path:

```text
/health/
```

---

### 6.5 Render Blueprint

Render Blueprint configuration is included in:

```text
deployment/render.yaml
```

For Render Blueprint deployment, copy it to the repository root:

```text
render.yaml
```

Render uses Blueprint YAML files to define services, databases, and environment settings. The standard filename is usually:

```text
render.yaml
```

---

### 6.6 API URLs After Deployment

After deployment, Render gives a public URL similar to:

```text
https://your-service-name.onrender.com
```

Useful API endpoints:

```text
/
 /docs
 /health/
 /health/full
 /models/status
 /forecast/default
 /anomaly/default
```

---

## 7. Deploy Streamlit Dashboard on Render

Render can also be used to deploy the Streamlit dashboard.

---

### 7.1 Build Command

Use this build command:

```bash
pip install --upgrade pip && pip install -r requirements.txt
```

---

### 7.2 Start Command

Use this start command:

```bash
streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false
```

---

### 7.3 Dashboard URL

After deployment, Render gives a URL similar to:

```text
https://financial-portfolio-dashboard.onrender.com
```

---

## 8. Deploy FastAPI on Railway

Railway is another beginner-friendly platform for deploying FastAPI applications.

---

### 8.1 Railway Config File

Railway configuration is included in:

```text
railway.json
```

---

### 8.2 Start Command

Use this start command:

```bash
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

---

### 8.3 Railway Deployment Steps

1. Push the project to GitHub.
2. Open Railway.
3. Create a new project.
4. Choose **Deploy from GitHub repo**.
5. Select your repository.
6. Railway reads `railway.json`.
7. Add environment variables if needed.
8. Deploy the service.

---

### 8.4 Railway URLs

After deployment, Railway gives a generated public URL.

Open the API documentation:

```text
https://your-railway-app.up.railway.app/docs
```

---

## 9. Deploy Using Docker

Docker allows the project to run inside a container. This helps create a production-like environment locally.

---

### 9.1 Build Docker Image

Build the Docker image:

```bash
docker build -t financial-portfolio-api .
```

---

### 9.2 Run Docker Container

Run the Docker container:

```bash
docker run -p 8000:8000 financial-portfolio-api
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

### 9.3 Docker Compose

Run the project using Docker Compose:

```bash
docker compose up --build
```

---

## 10. Important API Endpoints

| Endpoint            | Method | Purpose                                  |
| ------------------- | ------ | ---------------------------------------- |
| `/`                 | GET    | Root endpoint                            |
| `/health/`          | GET    | Health check                             |
| `/health/full`      | GET    | Full health report                       |
| `/models/status`    | GET    | Model artifact status                    |
| `/forecast`         | POST   | Forecast from JSON records               |
| `/anomaly`          | POST   | Detect anomalies from JSON records       |
| `/predict`          | POST   | Combined forecast and anomaly prediction |
| `/forecast/default` | GET    | Forecast using default dataset           |
| `/anomaly/default`  | GET    | Detect anomalies using default dataset   |
| `/upload/predict`   | POST   | Upload CSV and run prediction            |

---

## 11. Example API Test

After deploying, test the root endpoint:

```bash
curl https://your-api-url.com/
```

Test the health endpoint:

```bash
curl https://your-api-url.com/health/
```

Test the full health endpoint:

```bash
curl https://your-api-url.com/health/full
```

---

## 12. Common Deployment Issues

This section explains common deployment errors and how to fix them.

---

### Issue 1: App Cannot Find Dataset

Error:

```text
FileNotFoundError: financial_portfolio_data.csv not found
```

Fix:

Make sure this file exists:

```text
data/raw/financial_portfolio_data.csv
```

If the file is missing, run:

```bash
python scripts/run_data_pipeline.py
```

Or manually place the dataset inside:

```text
data/raw/
```

---

### Issue 2: App Cannot Find Model File

Error:

```text
Saved XGBoost model not found
```

Fix:

Train models before deployment:

```bash
python scripts/run_all_models.py
```

Or commit small model artifacts if allowed by the deployment platform and GitHub file size limits.

Expected model folder:

```text
models/
```

---

### Issue 3: Package Install Error

Possible reason:

Some packages in `requirements.txt` are missing, outdated, or incompatible.

Fix:

Update pip:

```bash
pip install --upgrade pip
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

---

### Issue 4: FastAPI App Path Error

Correct app path:

```text
api.main:app
```

Correct start command:

```bash
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

---

### Issue 5: Streamlit App Path Error

Correct dashboard path:

```text
dashboard/app.py
```

Correct start command:

```bash
streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

---

## 13. Free Deployment Recommendation

For a beginner portfolio deployment, use the following setup:

| Platform | Best For                    | Recommendation                |
| -------- | --------------------------- | ----------------------------- |
| Render   | FastAPI and Streamlit       | Good for portfolio demo       |
| Railway  | FastAPI backend             | Good for quick API deployment |
| Docker   | Local production simulation | Good for technical proof      |
| GitHub   | Code hosting                | Required for recruiters       |

---

### 13.1 Recommended Portfolio Setup

For a job-ready portfolio project:

1. Deploy FastAPI on Render or Railway.
2. Deploy Streamlit dashboard on Render.
3. Add deployed links to `README.md`.
4. Add screenshots to `docs/` or `README.md`.
5. Include API documentation URL.
6. Include dashboard URL.
7. Add deployment screenshots.
8. Mention deployment tools in the portfolio summary.

---

## 14. Final Deployment Checklist

| Task                           | Status  |
| ------------------------------ | ------- |
| Code pushed to GitHub          | Pending |
| `requirements.txt` updated     | Pending |
| API runs locally               | Pending |
| Dashboard runs locally         | Pending |
| Tests pass                     | Pending |
| Dataset available              | Pending |
| Model artifacts available      | Pending |
| Render config added            | Pending |
| Railway config added           | Pending |
| Environment variables added    | Pending |
| Cloud deployment tested        | Pending |
| README updated with live links | Pending |

---
