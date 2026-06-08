# Project Architecture

## Financial Portfolio Forecasting & Anomaly Detection

This document explains the system architecture of the **Financial Portfolio Forecasting & Anomaly Detection** project.

---

## 1. Project Overview

This project is an end-to-end machine learning system that performs:

* Financial asset price data loading
* Data cleaning and preprocessing
* Time series feature engineering
* Forecasting using multiple models
* Anomaly detection using statistical and machine learning methods
* Model evaluation
* FastAPI deployment
* Streamlit dashboard visualization
* Monitoring and logging

The project is designed as a portfolio-ready machine learning application.

---

## 2. High-Level Architecture

The project follows a complete machine learning workflow from raw data to prediction outputs, evaluation, API deployment, dashboard visualization, and monitoring.

```text
Raw CSV Data
    |
    v
Data Processing Layer
    |
    v
Feature Engineering Layer
    |
    v
Model Training Layer
    |
    +----------------------+
    |                      |
    v                      v
Forecasting Models     Anomaly Detection Model
    |                      |
    v                      v
Prediction Outputs     Anomaly Outputs
    |                      |
    +----------+-----------+
               |
               v
Evaluation Layer
               |
               v
API + Dashboard + Reports
```

---

## 3. Main Components

| Component   | Folder        | Purpose                                                                     |
| ----------- | ------------- | --------------------------------------------------------------------------- |
| Data        | `data/`       | Stores raw, processed, and sample data                                      |
| Notebooks   | `notebooks/`  | EDA, forecasting experiments, anomaly experiments, evaluation               |
| Source Code | `src/`        | Modular Python code for processing, models, training, inference, evaluation |
| API         | `api/`        | FastAPI backend for predictions                                             |
| Dashboard   | `dashboard/`  | Streamlit dashboard for visualization                                       |
| Models      | `models/`     | Saved trained model artifacts                                               |
| Reports     | `reports/`    | Figures, metrics, and final reports                                         |
| Tests       | `tests/`      | Unit and API tests                                                          |
| Scripts     | `scripts/`    | Run pipeline, train models, evaluate models, run API/dashboard              |
| Config      | `config/`     | YAML configuration files                                                    |
| Deployment  | `deployment/` | Render, Railway, and production deployment files                            |
| Monitoring  | `monitoring/` | Logs and monitoring configuration                                           |
| Docs        | `docs/`       | Project documentation                                                       |

---

## 4. Folder Structure

```text
financial-portfolio-forecasting-anomaly-detection/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
│
├── notebooks/
│
├── src/
│   ├── data_processing/
│   ├── models/
│   ├── training/
│   ├── inference/
│   ├── evaluation/
│   ├── monitoring/
│   └── utils/
│
├── api/
│   ├── main.py
│   ├── schemas.py
│   └── routes/
│
├── dashboard/
│   ├── app.py
│   └── pages/
│
├── models/
│   ├── forecasting/
│   ├── anomaly/
│   └── scalers/
│
├── reports/
│   ├── figures/
│   ├── metrics/
│   └── final_project_report.md
│
├── tests/
├── scripts/
├── config/
├── deployment/
├── monitoring/
└── docs/
```

---

## 5. Data Flow

The data flow explains how raw financial data moves through the system and becomes model-ready data, predictions, reports, API responses, and dashboard visuals.

---

### Step 1: Raw Data Input

Raw data is stored in:

```text
data/raw/financial_portfolio_data.csv
```

Expected columns:

* `Date`
* `Asset`
* `Price`

The project standardizes them to:

* `date`
* `asset`
* `price`

---

### Step 2: Data Processing

Data processing modules are stored in:

```text
src/data_processing/
```

Files:

| File                     | Purpose                                                  |
| ------------------------ | -------------------------------------------------------- |
| `data_loader.py`         | Loads raw CSV data                                       |
| `data_cleaning.py`       | Cleans missing values, invalid dates, and invalid prices |
| `feature_engineering.py` | Creates time series features                             |
| `train_test_split.py`    | Creates time-based train/test split                      |

Processed outputs:

```text
data/processed/cleaned_financial_data.csv
data/processed/daily_asset_prices.csv
data/processed/train_data.csv
data/processed/test_data.csv
data/processed/anomaly_results.csv
```

---

### Step 3: Feature Engineering

Feature engineering creates useful time series features from the cleaned financial price data.

Created feature groups include:

* Calendar features
* Lag features
* Rolling mean features
* Rolling standard deviation features
* Daily return
* Volatility features

Example features:

```text
year
month
quarter
day_of_week
daily_return
price_lag_1
price_lag_7
price_rolling_mean_7
price_rolling_std_7
```

---

## 6. Model Architecture

The model architecture contains two major parts:

1. Forecasting models
2. Anomaly detection models

---

### 6.1 Forecasting Models

Forecasting model files:

```text
src/models/arima_model.py
src/models/prophet_model.py
src/models/xgboost_model.py
src/models/lstm_model.py
```

Saved artifacts:

```text
models/forecasting/arima_model.pkl
models/forecasting/prophet_model.pkl
models/forecasting/xgboost_model.pkl
models/forecasting/lstm_model.keras
models/scalers/price_scaler.pkl
```

Models used:

| Model   | Type                         | Purpose                            |
| ------- | ---------------------------- | ---------------------------------- |
| ARIMA   | Statistical time series      | Baseline time series forecasting   |
| Prophet | Additive time series model   | Trend and seasonality forecasting  |
| XGBoost | Machine learning regression  | Feature-based price prediction     |
| LSTM    | Deep learning sequence model | Sequential time series forecasting |

---

### 6.2 Anomaly Detection Model

Anomaly detection file:

```text
src/models/anomaly_detector.py
```

Saved artifact:

```text
models/anomaly/isolation_forest_model.pkl
```

Methods used:

| Method           | Purpose                                         |
| ---------------- | ----------------------------------------------- |
| IQR              | Detects price outliers using quartiles          |
| Z-Score          | Detects values far from the mean                |
| Isolation Forest | Detects unusual records using multiple features |

---

## 7. Training Architecture

Training scripts:

```text
src/training/train_forecasting_models.py
src/training/train_lstm_model.py
src/training/train_anomaly_model.py
scripts/train_all_models.py
```

Training workflow:

```text
Load processed data
    |
    v
Create features
    |
    v
Train forecasting models
    |
    v
Train anomaly model
    |
    v
Save model artifacts
    |
    v
Save prediction outputs
```

Training outputs include:

* Forecasting model artifacts
* Anomaly detection model artifacts
* Scaler artifacts
* Prediction output files
* Training logs
* Evaluation-ready files

---

## 8. Evaluation Architecture

Evaluation modules:

```text
src/evaluation/metrics.py
src/evaluation/model_comparison.py
src/evaluation/visualization.py
scripts/evaluate_models.py
```

Evaluation outputs:

```text
reports/metrics/forecasting_metrics.json
reports/metrics/anomaly_metrics.json
reports/metrics/model_comparison.csv
reports/figures/model_comparison.png
reports/figures/forecast_vs_actual.png
reports/figures/anomaly_detection_plot.png
```

Forecasting metrics:

* MAE
* MSE
* RMSE
* MAPE
* SMAPE
* R2 Score

Anomaly evaluation:

* Total anomalies
* Anomaly rate
* Method-wise anomaly count
* Asset-wise anomaly count
* Severity distribution

---

## 9. API Architecture

The API layer provides prediction, anomaly detection, health check, and model status endpoints using FastAPI.

FastAPI files:

```text
api/main.py
api/schemas.py
api/routes/forecast_routes.py
api/routes/anomaly_routes.py
api/routes/health_routes.py
```

Main API app:

```text
api.main:app
```

API run command:

```bash
uvicorn api.main:app --reload
```

Important endpoints:

| Endpoint            | Method | Purpose                                    |
| ------------------- | ------ | ------------------------------------------ |
| `/`                 | GET    | Root endpoint                              |
| `/health/`          | GET    | Health check                               |
| `/health/full`      | GET    | Full project health                        |
| `/forecast`         | POST   | Forecast from JSON records                 |
| `/anomaly`          | POST   | Detect anomalies from JSON records         |
| `/predict`          | POST   | Combined forecasting and anomaly detection |
| `/forecast/default` | GET    | Forecast using default dataset             |
| `/anomaly/default`  | GET    | Detect anomalies using default dataset     |
| `/upload/predict`   | POST   | Upload CSV and run prediction              |

---

## 10. Dashboard Architecture

The dashboard layer provides a visual interface for exploring the project outputs, forecasts, anomalies, and model performance.

Dashboard files:

```text
dashboard/app.py
dashboard/pages/1_Portfolio_Overview.py
dashboard/pages/2_Price_Forecasting.py
dashboard/pages/3_Anomaly_Detection.py
dashboard/pages/4_Model_Performance.py
```

Dashboard run command:

```bash
streamlit run dashboard/app.py
```

Dashboard pages:

| Page               | Purpose                                        |
| ------------------ | ---------------------------------------------- |
| Portfolio Overview | Shows dataset and project summary              |
| Price Forecasting  | Shows forecasting results and model comparison |
| Anomaly Detection  | Shows anomaly detection outputs                |
| Model Performance  | Shows model metrics and artifact health        |

---

## 11. Monitoring Architecture

The monitoring layer tracks the project health, data quality, model performance, API availability, dashboard status, and logs.

Monitoring files:

```text
monitoring/monitoring_config.yaml
monitoring/performance_dashboard.md
monitoring/logs/app.log
```

Monitoring tracks:

* Data quality
* Model performance
* Drift detection
* API health
* Dashboard health
* Runtime performance
* Error logs

Recommended monitoring outputs:

```text
monitoring/performance_metrics.csv
monitoring/performance_summary.json
monitoring/drift_report.csv
monitoring/monitoring_summary.json
monitoring/reports/daily_monitoring_report.md
```

---

## 12. Deployment Architecture

The deployment layer contains files for cloud deployment, local container testing, and production notes.

Deployment files:

```text
deployment/render.yaml
deployment/railway.json
deployment/cloud_deployment_guide.md
deployment/production_notes.md
Dockerfile
docker-compose.yml
```

Supported deployment options:

| Platform | Purpose                              |
| -------- | ------------------------------------ |
| Render   | Deploy FastAPI and Streamlit         |
| Railway  | Deploy FastAPI                       |
| Docker   | Containerized local/cloud deployment |
| GitHub   | Portfolio source control             |

---

## 13. End-to-End Run Order

Recommended run order:

```bash
python scripts/run_data_pipeline.py
python scripts/train_all_models.py
python scripts/evaluate_models.py
python scripts/run_api.py --reload
python scripts/run_dashboard.py
```

---
