# API Documentation

## Financial Portfolio Forecasting & Anomaly Detection

This document explains the FastAPI backend used for forecasting asset prices and detecting anomalies in the **Financial Portfolio Forecasting & Anomaly Detection** project.

---

## 1. API Overview

The API provides endpoints for:

* Health checks
* Forecasting future asset prices
* Detecting anomalies
* Running combined prediction
* Uploading CSV files
* Checking model artifact status
* Returning latest API summary

Main API file:

```text
api/main.py
```

API app path:

```text
api.main:app
```

Run command:

```bash
uvicorn api.main:app --reload
```

API docs:

```text
http://127.0.0.1:8000/docs
```

---

## 2. API Folder Structure

```text
api/
├── __init__.py
├── main.py
├── schemas.py
└── routes/
    ├── __init__.py
    ├── forecast_routes.py
    ├── anomaly_routes.py
    └── health_routes.py
```

---

## 3. Base URL

Local base URL:

```text
http://127.0.0.1:8000
```

Production base URL example:

```text
https://your-api-url.com
```

---

## 4. Main Endpoints

| Endpoint            | Method | Purpose                                 |
| ------------------- | ------ | --------------------------------------- |
| `/`                 | GET    | Root API endpoint                       |
| `/health/`          | GET    | Basic health check                      |
| `/health/full`      | GET    | Full health report                      |
| `/models/status`    | GET    | Model artifact status                   |
| `/forecast`         | POST   | Forecast from JSON records              |
| `/anomaly`          | POST   | Detect anomalies from JSON records      |
| `/predict`          | POST   | Combined forecast and anomaly detection |
| `/forecast/default` | GET    | Forecast using default dataset          |
| `/anomaly/default`  | GET    | Detect anomalies using default dataset  |
| `/upload/predict`   | POST   | Upload CSV and run prediction           |
| `/summary/latest`   | GET    | Return latest API summary               |

---

## 5. Request Schema

This section explains the expected JSON request format for the API endpoints.

---

### 5.1 Price Record

Example price record:

```json
{
  "date": "2024-01-01",
  "asset": "Stock A",
  "price": 120.5
}
```

| Field   | Type   | Required | Description          |
| ------- | ------ | -------- | -------------------- |
| `date`  | String | Yes      | Date of price record |
| `asset` | String | Yes      | Asset name           |
| `price` | Float  | Yes      | Asset price          |

---

### 5.2 Forecast Request

Example forecast request:

```json
{
  "records": [
    {
      "date": "2024-01-01",
      "asset": "Stock A",
      "price": 120.5
    },
    {
      "date": "2024-01-02",
      "asset": "Stock A",
      "price": 121.0
    }
  ],
  "model_name": "xgboost",
  "forecast_steps": 30,
  "save_output": true,
  "return_records": true
}
```

| Field            | Type    | Required | Description                                     |
| ---------------- | ------- | -------- | ----------------------------------------------- |
| `records`        | List    | Yes      | Historical price records                        |
| `model_name`     | String  | No       | `xgboost`, `arima`, `prophet`, `lstm`, or `all` |
| `forecast_steps` | Integer | No       | Number of future steps                          |
| `save_output`    | Boolean | No       | Save CSV output                                 |
| `return_records` | Boolean | No       | Return forecast records                         |

---

### 5.3 Anomaly Request

Example anomaly request:

```json
{
  "records": [
    {
      "date": "2024-01-01",
      "asset": "Stock A",
      "price": 120.5
    }
  ],
  "save_output": true,
  "return_records": true,
  "only_anomalies": false
}
```

| Field            | Type    | Required | Description                |
| ---------------- | ------- | -------- | -------------------------- |
| `records`        | List    | Yes      | Historical price records   |
| `save_output`    | Boolean | No       | Save output                |
| `return_records` | Boolean | No       | Return records in response |
| `only_anomalies` | Boolean | No       | Return only anomaly rows   |

---

### 5.4 Combined Prediction Request

Example combined prediction request:

```json
{
  "records": [
    {
      "date": "2024-01-01",
      "asset": "Stock A",
      "price": 120.5
    }
  ],
  "model_name": "xgboost",
  "forecast_steps": 30,
  "run_forecasting": true,
  "run_anomaly_detection": true,
  "save_output": true,
  "return_records": true,
  "only_anomalies": false
}
```

| Field                   | Type    | Required | Description                         |
| ----------------------- | ------- | -------- | ----------------------------------- |
| `records`               | List    | Yes      | Historical price records            |
| `model_name`            | String  | No       | Forecasting model name              |
| `forecast_steps`        | Integer | No       | Number of future forecast steps     |
| `run_forecasting`       | Boolean | No       | Whether to run forecasting          |
| `run_anomaly_detection` | Boolean | No       | Whether to run anomaly detection    |
| `save_output`           | Boolean | No       | Whether to save output files        |
| `return_records`        | Boolean | No       | Whether to return output records    |
| `only_anomalies`        | Boolean | No       | Whether to return only anomaly rows |

---

## 6. Endpoint Details

---

### 6.1 Root Endpoint

Endpoint:

```text
GET /
```

Purpose:

Returns API status and project information.

Example response:

```json
{
  "status": "success",
  "message": "Financial Portfolio Forecasting & Anomaly Detection API is running.",
  "timestamp": "2026-06-08 10:00:00"
}
```

---

### 6.2 Health Check

Endpoint:

```text
GET /health/
```

Purpose:

Checks basic project health.

Example response:

```json
{
  "status": "success",
  "message": "API is healthy.",
  "timestamp": "2026-06-08 10:00:00"
}
```

---

### 6.3 Full Health Check

Endpoint:

```text
GET /health/full
```

Purpose:

Returns a complete health report for the project.

The full health check may include:

* Project information
* System information
* Dataset status
* Model artifact status
* Output artifact status
* Performance summary
* Monitoring file status

---

### 6.4 Model Status

Endpoint:

```text
GET /models/status
```

Purpose:

Checks whether trained model artifacts exist.

Expected model files:

```text
models/forecasting/arima_model.pkl
models/forecasting/prophet_model.pkl
models/forecasting/xgboost_model.pkl
models/forecasting/lstm_model.keras
models/anomaly/isolation_forest_model.pkl
models/scalers/price_scaler.pkl
```

---

### 6.5 Forecast Endpoint

Endpoint:

```text
POST /forecast
```

Purpose:

Generates future price forecasts from JSON records.

Example curl:

```bash
curl -X POST "http://127.0.0.1:8000/forecast" \
-H "Content-Type: application/json" \
-d '{
  "records": [
    {"date": "2024-01-01", "asset": "Stock A", "price": 100},
    {"date": "2024-01-02", "asset": "Stock A", "price": 101},
    {"date": "2024-01-03", "asset": "Stock A", "price": 102}
  ],
  "model_name": "xgboost",
  "forecast_steps": 30,
  "save_output": true,
  "return_records": true
}'
```

Expected output file:

```text
outputs/api_forecast_results.csv
```

---

### 6.6 Anomaly Endpoint

Endpoint:

```text
POST /anomaly
```

Purpose:

Detects anomalies from JSON records.

Example curl:

```bash
curl -X POST "http://127.0.0.1:8000/anomaly" \
-H "Content-Type: application/json" \
-d '{
  "records": [
    {"date": "2024-01-01", "asset": "Stock A", "price": 100},
    {"date": "2024-01-02", "asset": "Stock A", "price": 500},
    {"date": "2024-01-03", "asset": "Stock A", "price": 102}
  ],
  "save_output": true,
  "return_records": true,
  "only_anomalies": false
}'
```

Expected output file:

```text
outputs/api_anomaly_results.csv
```

---

### 6.7 Combined Predict Endpoint

Endpoint:

```text
POST /predict
```

Purpose:

Runs both forecasting and anomaly detection.

This endpoint is useful when the user wants one API call to generate both forecast results and anomaly detection results.

Example curl:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "records": [
    {"date": "2024-01-01", "asset": "Stock A", "price": 100},
    {"date": "2024-01-02", "asset": "Stock A", "price": 101},
    {"date": "2024-01-03", "asset": "Stock A", "price": 102}
  ],
  "model_name": "xgboost",
  "forecast_steps": 30,
  "run_forecasting": true,
  "run_anomaly_detection": true,
  "save_output": true,
  "return_records": true,
  "only_anomalies": false
}'
```

Expected output files:

```text
outputs/api_forecast_results.csv
outputs/api_anomaly_results.csv
reports/tables/api_summary.json
```

---

### 6.8 Default Forecast Endpoint

Endpoint:

```text
GET /forecast/default
```

Purpose:

Runs forecasting using the default dataset.

Default dataset path:

```text
data/raw/financial_portfolio_data.csv
```

---

### 6.9 Default Anomaly Endpoint

Endpoint:

```text
GET /anomaly/default
```

Purpose:

Runs anomaly detection using the default CSV dataset.

Default dataset path:

```text
data/raw/financial_portfolio_data.csv
```

---

### 6.10 Upload Predict Endpoint

Endpoint:

```text
POST /upload/predict
```

Purpose:

Uploads a CSV file and runs forecast or anomaly detection.

CSV must contain:

```text
date, asset, price
```

or:

```text
Date, Asset, Price
```

Example curl:

```bash
curl -X POST "http://127.0.0.1:8000/upload/predict" \
-F "file=@data/sample/sample_prediction_input.csv"
```

---

### 6.11 Latest Summary Endpoint

Endpoint:

```text
GET /summary/latest
```

Purpose:

Returns the latest API summary if the summary file exists.

Expected summary files may include:

```text
reports/tables/api_summary.json
reports/api_summary.md
```

---

## 7. Response Format

---

### 7.1 Standard Success Response

```json
{
  "status": "success",
  "message": "Request completed successfully.",
  "timestamp": "2026-06-08 10:00:00",
  "data": {}
}
```

---

### 7.2 Standard Error Response

```json
{
  "status": "failed",
  "error_type": "ValueError",
  "error_message": "Missing required columns.",
  "timestamp": "2026-06-08 10:00:00"
}
```

---

### 7.3 Forecast Response Fields

Forecast responses may include:

| Field              | Description                     |
| ------------------ | ------------------------------- |
| `status`           | Request status                  |
| `message`          | Response message                |
| `timestamp`        | Response timestamp              |
| `model_name`       | Forecasting model used          |
| `forecast_steps`   | Number of future forecast steps |
| `records_count`    | Number of input records         |
| `forecast_records` | Forecast output records         |
| `output_file`      | Saved output file path          |

---

### 7.4 Anomaly Response Fields

Anomaly responses may include:

| Field             | Description                     |
| ----------------- | ------------------------------- |
| `status`          | Request status                  |
| `message`         | Response message                |
| `timestamp`       | Response timestamp              |
| `records_count`   | Number of input records         |
| `total_anomalies` | Total detected anomalies        |
| `anomaly_rate`    | Percentage of anomalous records |
| `anomaly_records` | Anomaly output records          |
| `output_file`     | Saved output file path          |

---

## 8. Output Files

API outputs:

```text
outputs/api_forecast_results.csv
outputs/api_anomaly_results.csv
reports/tables/api_summary.json
reports/api_summary.md
```

Forecast route outputs:

```text
outputs/api_forecast_results.csv
reports/tables/api_forecast_summary.json
reports/api_forecast_summary.md
```

Anomaly route outputs:

```text
outputs/api_anomaly_results.csv
reports/tables/api_anomaly_summary.json
reports/api_anomaly_summary.md
```

---

## 9. Running API Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run API:

```bash
python scripts/run_api.py --reload
```

Or run directly with Uvicorn:

```bash
uvicorn api.main:app --reload
```

Open API docs:

```text
http://127.0.0.1:8000/docs
```

Open ReDoc:

```text
http://127.0.0.1:8000/redoc
```

---

## 10. Testing API

Run API tests:

```bash
pytest tests/test_api.py -v
```

Run all tests:

```bash
pytest tests -v
```

---

## 11. Deployment API Command

Production command:

```bash
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

Render start command:

```bash
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

Railway start command:

```bash
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

---

## 12. Common API Issues and Fixes

---

### Issue 1: API Not Starting

Possible reason:

* FastAPI is not installed.
* Uvicorn is not installed.
* App path is incorrect.

Fix:

```bash
pip install fastapi uvicorn
uvicorn api.main:app --reload
```

---

### Issue 2: Module Import Error

Possible reason:

Python cannot find the project modules.

Fix:

Run the command from the project root folder:

```bash
uvicorn api.main:app --reload
```

Or use:

```bash
python scripts/run_api.py --reload
```

---

### Issue 3: Dataset Missing

Expected file:

```text
data/raw/financial_portfolio_data.csv
```

Fix:

Place the dataset inside:

```text
data/raw/
```

Or run:

```bash
python scripts/run_data_pipeline.py
```

---

### Issue 4: Model Artifacts Missing

Expected model files:

```text
models/forecasting/arima_model.pkl
models/forecasting/prophet_model.pkl
models/forecasting/xgboost_model.pkl
models/forecasting/lstm_model.keras
models/anomaly/isolation_forest_model.pkl
models/scalers/price_scaler.pkl
```

Fix:

Train models:

```bash
python scripts/train_all_models.py
```

---

### Issue 5: Invalid JSON Input

Possible reason:

* Missing required fields
* Invalid date format
* Non-numeric price
* Empty records list

Fix:

Use this minimum valid record format:

```json
{
  "records": [
    {
      "date": "2024-01-01",
      "asset": "Stock A",
      "price": 120.5
    }
  ]
}
```

---


