# Production Notes

## Financial Portfolio Forecasting & Anomaly Detection

This document explains production-readiness notes, limitations, risks, and recommended improvements for the **Financial Portfolio Forecasting & Anomaly Detection** project.

---

## 1. Production Purpose

This project demonstrates an end-to-end machine learning system for:

* Financial asset price forecasting
* Anomaly detection
* API-based inference
* Dashboard-based monitoring
* Model evaluation
* Monitoring and logging

It is suitable for:

* Data Science portfolio
* Internship submission
* ML engineering practice
* API and dashboard demonstration

It is not yet a full enterprise-grade trading or investment system.

---

## 2. Current Production-Ready Features

| Feature                      | Status    |
| ---------------------------- | --------- |
| Modular project structure    | Available |
| Data cleaning pipeline       | Available |
| Feature engineering pipeline | Available |
| Forecasting models           | Available |
| Anomaly detection models     | Available |
| Model evaluation scripts     | Available |
| FastAPI backend              | Available |
| Streamlit dashboard          | Available |
| Docker support               | Available |
| Render deployment config     | Available |
| Railway deployment config    | Available |
| Monitoring config            | Available |
| Test files                   | Available |

---

## 3. Important Limitations

This section explains the current limitations of the project and recommended improvements for making it more production-ready.

---

### 3.1 Dataset Limitation

The current project uses a CSV file:

```text
data/raw/financial_portfolio_data.csv
```

#### Limitations

* Static dataset
* No live market data feed
* No automatic daily data refresh
* No external financial API integration in production mode

#### Recommended Improvement

* Add live data collection from a reliable financial data API.
* Store updated data in a database.
* Schedule daily data refresh.
* Add validation checks before saving new data.
* Track dataset version and update date.

---

### 3.2 Model Limitation

The project includes multiple forecasting models:

* ARIMA
* XGBoost
* Prophet
* LSTM

#### Limitations

* Models depend on available historical data.
* Performance may degrade when market behavior changes.
* Forecasting financial prices is uncertain.
* LSTM and Prophet may require heavier cloud resources.
* Models may not include external market indicators.
* Model performance may vary across different assets.

#### Recommended Improvement

* Add scheduled retraining.
* Track model drift.
* Store model versions.
* Add model registry support.
* Compare model performance regularly.
* Add baseline models for comparison.
* Add asset-specific model evaluation.

---

### 3.3 Anomaly Detection Limitation

Anomaly detection methods:

* IQR
* Z-score
* Isolation Forest

#### Limitations

* Anomalies are statistical signals, not guaranteed fraud or risk events.
* High volatility can create false positives.
* Low volatility periods can hide real issues.
* Business review is still required.
* Thresholds may not work equally for all assets.

#### Recommended Improvement

* Add explainability for anomaly reasons.
* Add asset-specific thresholds.
* Add alert rules for high-severity anomalies.
* Add historical anomaly trend analysis.
* Add manual review status for detected anomalies.
* Add anomaly severity scoring.

---

### 3.4 Deployment Limitation

The current deployment setup is simple and suitable for portfolio demonstration.

#### Limitations

* No authentication
* No user management
* No database backend
* No cloud object storage for models
* No CI/CD production pipeline
* Limited persistence on free hosting platforms
* Large model files may not deploy smoothly on free cloud services

#### Recommended Improvement

* Add authentication.
* Use a managed database.
* Store model files in cloud storage.
* Add GitHub Actions for tests and deployment.
* Add production monitoring.
* Add API rate limiting.
* Add structured logging.
* Add cloud-based error tracking.

---

## 4. Security Notes

Security is important before moving the project from portfolio/demo level to production level.

---

### 4.1 API Security

The current API is open by default.

#### Production Risk

Anyone with the URL can call prediction endpoints.

#### Recommended Fixes

* Add API key authentication.
* Add JWT authentication if users are involved.
* Add rate limiting.
* Restrict CORS origins.
* Validate file upload size.
* Reject unsupported file types.
* Add request logging.
* Add input schema validation.
* Add error handling for invalid requests.

---

### 4.2 Environment Variables

Do not hardcode secrets inside Python files.

Use `.env` locally and cloud environment variables in production.

#### Recommended Variables

```env
ENVIRONMENT=production
LOG_LEVEL=INFO
API_KEY=your-secure-api-key
DATA_PATH=data/raw/financial_portfolio_data.csv
MODEL_DIR=models
OUTPUT_DIR=outputs
REPORTS_DIR=reports
MONITORING_DIR=monitoring
```

#### Never Commit These Files

```text
.env
secrets.json
private_key.json
credentials.json
```

---

### 4.3 File Upload Security

The API supports CSV upload.

#### Production Risks

* Large files can slow down the server.
* Invalid files can crash processing.
* Malicious file names can cause issues.
* Unsupported file formats can create errors.
* Poorly formatted CSV files can break prediction logic.

#### Recommended Fixes

* Limit upload file size.
* Accept only `.csv` files.
* Validate MIME type.
* Store uploaded files temporarily.
* Delete temporary files after processing.
* Add try-except handling.
* Validate required columns before prediction.
* Return clear error messages to the user.

---

## 5. Performance Notes

Performance optimization is important for API speed, dashboard loading, and model inference.

---

### 5.1 API Performance

FastAPI app path:

```text
api.main:app
```

Development start command:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Production start command:

```bash
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

For heavier production use, consider:

```bash
gunicorn api.main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
```

#### API Performance Tips

* Load model artifacts once during application startup.
* Avoid retraining inside API endpoints.
* Keep request payload size reasonable.
* Use async endpoints where useful.
* Add request timeout handling.
* Log slow requests.
* Return clear errors for invalid input.

---

### 5.2 Dashboard Performance

Streamlit start command:

```bash
streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

#### Performance Tips

* Cache loaded CSV files.
* Avoid loading very large files repeatedly.
* Use filtered charts.
* Use precomputed evaluation outputs.
* Avoid training models inside the dashboard.
* Use pagination or filters for large tables.
* Keep chart rendering simple and fast.

---

### 5.3 Model Inference Performance

For better inference speed:

* Load models once if possible.
* Avoid retraining during API requests.
* Use saved model artifacts.
* Keep forecasting horizon reasonable.
* Limit uploaded CSV size.
* Use optimized data preprocessing.
* Avoid unnecessary recalculation.

Recommended maximum forecast steps:

```text
365
```

Recommended default forecast steps:

```text
30
```

---

## 6. Monitoring Notes

Monitoring helps track the health of the project after deployment.

### Monitoring Files

```text
monitoring/monitoring_config.yaml
monitoring/performance_dashboard.md
monitoring/logs/app.log
```

---

### Recommended Monitoring Areas

| Area              | What to Monitor                                |
| ----------------- | ---------------------------------------------- |
| Data Quality      | Missing values, invalid dates, negative prices |
| Forecasting       | MAE, RMSE, MAPE, R2 Score                      |
| Anomaly Detection | Anomaly count, anomaly rate, severity          |
| Drift             | Price distribution and return distribution     |
| API               | Response time, error rate, uptime              |
| Dashboard         | Load time and chart rendering                  |
| Logs              | Errors, warnings, failed steps                 |

---

## 7. Logging Notes

Logging helps identify failures, warnings, and successful project steps.

---

### Recommended Log Files

```text
monitoring/logs/app.log
monitoring/logs/training.log
monitoring/logs/inference.log
monitoring/logs/api.log
monitoring/logs/dashboard.log
monitoring/logs/errors.log
```

---

### Recommended Log Levels

| Level    | Use                      |
| -------- | ------------------------ |
| INFO     | Successful normal events |
| WARNING  | Non-critical issues      |
| ERROR    | Failed operation         |
| CRITICAL | Serious failure          |

---

### Recommended Logging Practices

* Log pipeline start and end.
* Log model training results.
* Log API request errors.
* Log missing files clearly.
* Log dashboard loading issues.
* Log critical failures separately.
* Avoid logging secrets or API keys.

---

## 8. Testing Notes

Testing is important to make sure the project works correctly before deployment.

---

### Run All Tests

```bash
pytest tests -v
```

---

### Run Selected Tests

```bash
pytest tests/test_data_cleaning.py -v
pytest tests/test_feature_engineering.py -v
pytest tests/test_frecast_service.py -v
pytest tests/test_anomaly_service.py -v
pytest tests/test_api.py -v
```

> Note: If your actual file name is `test_forecast_service.py`, use that corrected spelling instead of `test_frecast_service.py`.

---

### Required Testing Packages

```text
pytest
httpx
fastapi
```

---

### Recommended Future Tests

* API authentication tests
* File upload tests
* Model artifact loading tests
* Dashboard smoke tests
* Data drift tests
* Performance tests
* Invalid input tests
* Missing file tests
* Health endpoint tests

---

## 9. Deployment Notes

This section summarizes the deployment commands for Render, Railway, and Docker.

---

### 9.1 Render

FastAPI start command:

```bash
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

Dashboard start command:

```bash
streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false
```

Health check:

```text
/health/
```

---

### 9.2 Railway

Railway start command:

```bash
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

Config file:

```text
railway.json
```

---

### 9.3 Docker

Build Docker image:

```bash
docker build -t financial-portfolio-api .
```

Run Docker container:

```bash
docker run -p 8000:8000 financial-portfolio-api
```

Open API documentation:

```text
http://127.0.0.1:8000/docs
```

---

## 10. Model Artifact Notes

Model artifacts are saved trained models used by the API, dashboard, and inference scripts.

---

