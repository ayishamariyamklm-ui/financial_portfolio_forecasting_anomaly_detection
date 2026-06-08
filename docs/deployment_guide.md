# Deployment Guide

## Financial Portfolio Forecasting & Anomaly Detection

This guide explains how to run and deploy the **Financial Portfolio Forecasting & Anomaly Detection** project locally and on cloud platforms.

---

## 1. Deployment Overview

The project has two main deployable applications:

| App       | Framework | File               | Purpose                                   |
| --------- | --------- | ------------------ | ----------------------------------------- |
| API       | FastAPI   | `api/main.py`      | Forecasting and anomaly detection backend |
| Dashboard | Streamlit | `dashboard/app.py` | Visual dashboard for project results      |

The API is used to serve forecasting and anomaly detection results through endpoints, while the dashboard provides a visual interface for exploring project outputs.

---

## 2. Local Setup

Follow these steps to run the project locally before deployment.

---

### 2.1 Clone Repository

```bash
git clone https://github.com/your-username/financial-portfolio-forecasting-anomaly-detection.git
cd financial-portfolio-forecasting-anomaly-detection
```

---

### 2.2 Create Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Mac/Linux

```bash
python -m venv venv
source venv/bin/activate
```

---

### 2.3 Install Requirements

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Required Files Before Running

Make sure the dataset exists:

```text
data/raw/financial_portfolio_data.csv
```

Make sure these files exist:

```text
api/main.py
dashboard/app.py
requirements.txt
```

Recommended folders before running:

```text
data/
models/
outputs/
reports/
monitoring/
```

---

## 4. Local Run Order

Recommended full run order:

```bash
python scripts/run_data_pipeline.py
python scripts/train_all_models.py
python scripts/evaluate_models.py
python scripts/run_api.py --reload
python scripts/run_dashboard.py
```

This order ensures that the data pipeline runs first, models are trained, evaluation files are generated, and then the API and dashboard are started.

---

## 5. Run API Locally

Using script:

```bash
python scripts/run_api.py --reload
```

Using Uvicorn directly:

```bash
uvicorn api.main:app --reload
```

Open API docs:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health/full
```

---

## 6. Run Dashboard Locally

Using script:

```bash
python scripts/run_dashboard.py
```

Using Streamlit directly:

```bash
streamlit run dashboard/app.py
```

Open dashboard:

```text
http://localhost:8501
```

---

## 7. Docker Deployment

Docker can be used to run the project inside a container.

---

### 7.1 Build Image

```bash
docker build -t financial-portfolio-api .
```

---

### 7.2 Run Container

```bash
docker run -p 8000:8000 financial-portfolio-api
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

### 7.3 Docker Compose

```bash
docker-compose up --build
```

Or, if your Docker version uses the newer command:

```bash
docker compose up --build
```

---

## 8. Render Deployment

Render can deploy both the FastAPI backend and the Streamlit dashboard.

Render deployment configuration:

```text
deployment/render.yaml
```

For Render Blueprint deployment, copy this file to the project root:

```text
render.yaml
```

---

### 8.1 FastAPI Render Settings

Build command:

```bash
pip install --upgrade pip && pip install -r requirements.txt
```

Start command:

```bash
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

Health check path:

```text
/health/
```

---

### 8.2 Streamlit Render Settings

Build command:

```bash
pip install --upgrade pip && pip install -r requirements.txt
```

Start command:

```bash
streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false
```

---

## 9. Railway Deployment

Railway can be used to deploy the FastAPI backend.

Railway config file:

```text
deployment/railway.json
```

If Railway expects the config file at the root, copy it to:

```text
railway.json
```

Start command:

```bash
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

---

### 9.1 Railway Deployment Steps

1. Push project to GitHub.
2. Create a Railway project.
3. Select your GitHub repository.
4. Add environment variables if needed.
5. Confirm the start command.
6. Deploy the service.
7. Open the generated Railway URL.
8. Test the API documentation endpoint.

API documentation URL format:

```text
https://your-railway-app.up.railway.app/docs
```

---

## 10. Environment Variables

Recommended environment variables:

```env
PYTHON_VERSION=3.11.9
ENVIRONMENT=production
LOG_LEVEL=INFO
DATA_PATH=data/raw/financial_portfolio_data.csv
MODEL_DIR=models
OUTPUT_DIR=outputs
REPORTS_DIR=reports
MONITORING_DIR=monitoring
```

Do not commit:

```text
.env
secrets.json
credentials.json
```

If authentication is added later, also keep API keys and tokens only in environment variables.

---

## 11. API Endpoints After Deployment

| Endpoint            | Purpose                             |
| ------------------- | ----------------------------------- |
| `/`                 | Root                                |
| `/docs`             | API documentation                   |
| `/health/`          | Basic health                        |
| `/health/full`      | Full health                         |
| `/models/status`    | Model artifact status               |
| `/forecast/default` | Forecast default dataset            |
| `/anomaly/default`  | Detect anomalies on default dataset |

Example deployed API URL:

```text
https://your-api-url.com/docs
```

---

## 12. Testing Before Deployment

Run tests:

```bash
pytest tests -v
```

Run API:

```bash
python scripts/run_api.py --reload
```

Run dashboard:

```bash
python scripts/run_dashboard.py
```

Before sharing the deployed project link, confirm:

* API docs open successfully.
* Health endpoint works.
* Dashboard loads correctly.
* Dataset is available.
* Model artifacts are available.
* No critical errors appear in logs.

---

## 13. Common Deployment Errors

This section explains common deployment errors and fixes.

---

### Error: `ModuleNotFoundError`

Possible reason:

Required packages are not installed.

Fix:

```bash
pip install -r requirements.txt
```

---

### Error: `api.main:app` Not Found

Check that this file exists:

```text
api/main.py
```

The file must contain:

```python
app = FastAPI()
```

Correct start command:

```bash
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

---

### Error: Dataset Missing

Expected file:

```text
data/raw/financial_portfolio_data.csv
```

Fix:

Make sure the dataset exists in the correct folder:

```text
data/raw/
```

Or run:

```bash
python scripts/run_data_pipeline.py
```

---

### Error: Model File Missing

Possible reason:

Models were not trained before deployment.

Fix:

```bash
python scripts/train_all_models.py
```

Expected model folder:

```text
models/
```

---

### Error: Port Issue

Cloud platforms usually provide the port using `$PORT`.

Use:

```bash
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

For Streamlit:

```bash
streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

---