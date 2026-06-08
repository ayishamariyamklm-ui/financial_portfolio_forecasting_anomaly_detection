# Performance Monitoring Dashboard

## Financial Portfolio Forecasting & Anomaly Detection

This document provides a monitoring dashboard template for tracking the performance, health, and reliability of the **Financial Portfolio Forecasting & Anomaly Detection** project.

Project: Financial Portfolio Forecasting & Anomaly Detection  
Module: Monitoring  
File: `monitoring/performance_dashboard.md`  
Purpose: Track model, data, API, dashboard, and system performance.

---

## 1. Monitoring Dashboard Overview

The monitoring dashboard helps track:

- Data quality status
- Forecasting model performance
- Anomaly detection behavior
- Data drift indicators
- API health
- Dashboard health
- Runtime performance
- Error logs
- Output file availability
- Business risk indicators

---

## 2. Key Monitoring Goals

| Goal | Description |
|---|---|
| Data Reliability | Ensure input data is clean, valid, and complete |
| Forecast Accuracy | Track MAE, RMSE, MAPE, SMAPE, and R2 Score |
| Anomaly Monitoring | Track anomaly count, anomaly rate, and severity |
| Drift Detection | Identify changes in price and return distributions |
| API Stability | Monitor API availability and response time |
| Dashboard Stability | Ensure Streamlit dashboard loads correctly |
| Runtime Tracking | Track execution time for pipeline and model workflows |
| Error Visibility | Capture warnings, errors, and failed steps |

---

## 3. Project Health Summary

| Component | Status | Expected Output |
|---|---|---|
| Raw Dataset | Pending Check | `data/raw/financial_portfolio_data.csv` |
| Processed Dataset | Pending Check | `data/processed/cleaned_financial_data.csv` |
| Forecast Models | Pending Check | `models/` |
| Anomaly Detector | Pending Check | `models/anomaly_detection_model.pkl` |
| Forecast Outputs | Pending Check | `outputs/` |
| Anomaly Outputs | Pending Check | `outputs/anomaly_detection_results.csv` |
| Evaluation Reports | Pending Check | `reports/` |
| API Service | Pending Check | `http://127.0.0.1:8000/docs` |
| Dashboard Service | Pending Check | `http://localhost:8501` |
| Logs | Pending Check | `monitoring/logs/` |

---

## 4. Data Quality Dashboard

### 4.1 Data Quality Checks

| Check | Threshold | Status | Action |
|---|---:|---|---|
| Required columns exist | 100% required | Pending | Validate `date`, `asset`, `price` |
| Missing values | Less than 5% | Pending | Clean or impute missing values |
| Duplicate rows | Less than 2% | Pending | Remove duplicates |
| Negative prices | 0 records | Pending | Remove invalid price records |
| Invalid dates | 0 records | Pending | Convert and validate dates |
| Minimum records | At least 10 | Pending | Add more data if too small |
| Minimum assets | At least 1 | Pending | Check asset column |

### 4.2 Required Columns

The dataset must contain:

```text
date
asset
price
```

Original CSV can contain:

```text
Date
Asset
Price
```

The pipeline should standardize these to lowercase.

### 4.3 Dataset Summary Metrics

| Metric              |         Value |
| ------------------- | ------------: |
| Total Records       | To be updated |
| Total Assets        | To be updated |
| Start Date          | To be updated |
| End Date            | To be updated |
| Average Price       | To be updated |
| Minimum Price       | To be updated |
| Maximum Price       | To be updated |
| Missing Value Count | To be updated |
| Duplicate Count     | To be updated |

---

## 5. Forecasting Model Performance Dashboard

### 5.1 Forecasting Models Monitored

| Model   | Artifact Path               | Status  |
| ------- | --------------------------- | ------- |
| ARIMA   | `models/arima_models.pkl`   | Pending |
| XGBoost | `models/xgboost_model.pkl`  | Pending |
| Prophet | `models/prophet_models.pkl` | Pending |
| LSTM    | `models/lstm_models/`       | Pending |


### 5.2 Forecasting Metrics

| Model   | MAE | MSE | RMSE | MAPE | SMAPE | R2 Score | Rank |
| ------- | --: | --: | ---: | ---: | ----: | -------: | ---: |
| ARIMA   | TBD | TBD |  TBD |  TBD |   TBD |      TBD |  TBD |
| XGBoost | TBD | TBD |  TBD |  TBD |   TBD |      TBD |  TBD |
| Prophet | TBD | TBD |  TBD |  TBD |   TBD |      TBD |  TBD |
| LSTM    | TBD | TBD |  TBD |  TBD |   TBD |      TBD |  TBD |


### 5.3 Metric Interpretation

| Metric   | Meaning                                    | Better Direction |
| -------- | ------------------------------------------ | ---------------- |
| MAE      | Average absolute forecast error            | Lower            |
| MSE      | Average squared forecast error             | Lower            |
| RMSE     | Error metric that penalizes large mistakes | Lower            |
| MAPE     | Average percentage forecast error          | Lower            |
| SMAPE    | Symmetric percentage forecast error        | Lower            |
| R2 Score | Model explanation power                    | Higher           |


### 5.4 Forecasting Thresholds

| Metric   | Warning Threshold | Critical Threshold |
| -------- | ----------------: | -----------------: |
| RMSE     |   Greater than 50 |   Greater than 100 |
| MAE      |   Greater than 30 |    Greater than 60 |
| MAPE     |  Greater than 25% |   Greater than 50% |
| R2 Score |    Less than 0.50 |     Less than 0.20 |


### 5.5 Best Model Selection

Primary selection metric:

`RMSE`

Best model rule:

`Lowest RMSE wins`

Current best model:

`To be updated after running scripts/evaluate_models.py`

Expected output file:

``reports/tables/model_comparison.csv`
`reports/tables/best_forecasting_model.json`
`
---

## 6. Anomaly Detection Monitoring Dashboard

### 6.1 Anomaly Detection Methods

| Method           | Purpose                                      | Status  |
| ---------------- | -------------------------------------------- | ------- |
| IQR              | Detects extreme price values using quartiles | Pending |
| Z-Score          | Detects values far from average              | Pending |
| Isolation Forest | Detects unusual patterns using ML            | Pending |

### 6.2 Anomaly Summary

| Metric                  |         Value |
| ----------------------- | ------------: |
| Total Records Checked   | To be updated |
| Total Normal Records    | To be updated |
| Total Anomalies         | To be updated |
| Anomaly Rate %          | To be updated |
| High Severity Anomalies | To be updated |
| Assets with Anomalies   | To be updated |


### 6.3 Method-wise Anomaly Count

| Method           | Anomaly Count |
| ---------------- | ------------: |
| IQR              |           TBD |
| Z-Score          |           TBD |
| Isolation Forest |           TBD |


### 6.4 Severity Distribution

| Severity | Meaning                   | Record Count |
| -------- | ------------------------- | -----------: |
| Normal   | Not detected as anomaly   |          TBD |
| Low      | Detected by one method    |          TBD |
| Medium   | Detected by two methods   |          TBD |
| High     | Detected by three methods |          TBD |


### 6.5 Asset-wise Anomaly Monitoring

| Asset   | Total Records | Anomaly Count | Anomaly Rate % | Risk Level |
| ------- | ------------: | ------------: | -------------: | ---------- |
| Asset 1 |           TBD |           TBD |            TBD | TBD        |
| Asset 2 |           TBD |           TBD |            TBD | TBD        |
| Asset 3 |           TBD |           TBD |            TBD | TBD        |


### 6.6 Anomaly Thresholds

| Indicator           |               Warning Threshold |              Critical Threshold |
| ------------------- | ------------------------------: | ------------------------------: |
| Anomaly Rate        |                   More than 20% |                   More than 35% |
| High Severity Count |                     More than 5 |                    More than 10 |
| Anomalies per Asset |                    More than 50 |                   More than 100 |
| Sudden Price Jump   | More than 3 standard deviations | More than 5 standard deviations |

Expected output files:

outputs/anomaly_detection_results.csv
outputs/inference_anomaly_results.csv
outputs/api_anomaly_results.csv
reports/tables/anomaly_metrics.json
reports/tables/04_asset_anomaly_summary.csv

--- 

## 7. Data Drift Monitoring Dashboard

### 7.1 Drift Monitoring Purpose

Data drift means the new data distribution is different from the training data distribution.

This project should monitor drift in:

- Price
- Daily return
- Rolling mean
- Rolling standard deviation
- Volatility
- Asset distribution

### 7.2 Drift Checks

| Feature            | Drift Test |      Threshold | Status  |
| ------------------ | ---------- | -------------: | ------- |
| Price              | KS Test    | p-value < 0.05 | Pending |
| Daily Return       | KS Test    | p-value < 0.05 | Pending |
| Rolling Mean       | Mean Shift |          > 20% | Pending |
| Rolling Std        | Std Shift  |          > 30% | Pending |
| Asset Distribution | Chi-square | p-value < 0.05 | Pending |

### 7.3 Drift Severity

| Drift Score | Severity | Action                      |
| ----------: | -------- | --------------------------- |
| 0.00 - 0.10 | Low      | Continue monitoring         |
| 0.11 - 0.30 | Medium   | Review feature distribution |
| 0.31 - 0.50 | High     | Consider retraining         |
|  Above 0.50 | Critical | Retrain model urgently      |

Expected output file:

monitoring/drift_report.csv

---

## 8. Runtime Performance Dashboard

### 8.1 Execution Time Monitoring

| Step                | Expected Time | Warning Threshold | Status  |
| ------------------- | ------------: | ----------------: | ------- |
| Data Loading        |      < 10 sec |          > 30 sec | Pending |
| Data Cleaning       |      < 20 sec |          > 60 sec | Pending |
| Feature Engineering |      < 30 sec |         > 120 sec | Pending |
| ARIMA Training      |      < 10 min |          > 30 min | Pending |
| XGBoost Training    |       < 5 min |          > 20 min | Pending |
| Prophet Training    |      < 10 min |          > 30 min | Pending |
| LSTM Training       |      < 20 min |          > 60 min | Pending |
| Anomaly Training    |       < 5 min |          > 15 min | Pending |
| Model Evaluation    |       < 2 min |          > 10 min | Pending |
| API Response        |      < 10 sec |          > 10 sec | Pending |
| Dashboard Load      |      < 15 sec |          > 15 sec | Pending |

### 8.2 Performance Tracking Fields

| Field                  | Description                                           |
| ---------------------- | ----------------------------------------------------- |
| step_name              | Name of workflow step                                 |
| category               | Data, training, evaluation, inference, API, dashboard |
| start_time             | Step start timestamp                                  |
| end_time               | Step end timestamp                                    |
| execution_time_seconds | Total step runtime                                    |
| status                 | Success or failed                                     |
| error_message          | Error details if failed                               |
| record_count           | Number of records processed                           |

Expected output files:

monitoring/performance_metrics.csv
monitoring/performance_summary.json
reports/tables/performance_summary.json

---

## 9. API Monitoring Dashboard

### 9.1 API Endpoints

| Endpoint            | Purpose                                  | Status  |
| ------------------- | ---------------------------------------- | ------- |
| `/`                 | Root endpoint                            | Pending |
| `/health/`          | Basic health check                       | Pending |
| `/health/full`      | Full health check                        | Pending |
| `/models/status`    | Model artifact status                    | Pending |
| `/forecast`         | Forecast from JSON records               | Pending |
| `/anomaly`          | Detect anomalies from JSON records       | Pending |
| `/predict`          | Combined forecast and anomaly prediction | Pending |
| `/forecast/default` | Forecast using default dataset           | Pending |
| `/anomaly/default`  | Detect anomalies using default dataset   | Pending |
| `/upload/predict`   | Upload CSV and run prediction            | Pending |


### 9.2 API Health Metrics

| Metric                  |   Target |
| ----------------------- | -------: |
| API availability        |     99%+ |
| Average response time   | < 10 sec |
| Error rate              |     < 5% |
| Successful request rate |    > 95% |


### 9.3 API Run Command

python scripts/run_api.py --reload

API docs:

http://127.0.0.1:8000/docs

Full health endpoint:

http://127.0.0.1:8000/health/full

---

````markdown
## 10. Dashboard Monitoring

Dashboard monitoring is used to verify whether the Streamlit dashboard is working correctly, loading required project files, displaying charts, and showing model outputs without critical UI errors.

---

### 10.1 Dashboard Pages

| Page | File | Status |
|---|---|---|
| Main App | `dashboard/app.py` | Pending |
| Portfolio Overview | `dashboard/pages/1_portfolio_overview.py` | Pending |
| Price Forecasting | `dashboard/pages/2_price_forecasting.py` | Pending |
| Anomaly Detection | `dashboard/pages/3_anomaly_detection.py` | Pending |
| Model Performance | `dashboard/pages/4_model_performance.py` | Pending |

---

### 10.2 Dashboard Health Metrics

| Metric | Target |
|---|---|
| Dashboard load time | `< 15 sec` |
| Chart render time | `< 10 sec` |
| Required output files available | Yes |
| Dataset loaded successfully | Yes |
| No critical UI error | Yes |

---

### 10.3 Dashboard Run Command

Use the following command to start the dashboard:

```bash
python scripts/run_dashboard.py
````

Dashboard URL:

```text
http://localhost:8501
```

---

## 11. Logging Dashboard

The logging dashboard helps track project execution, model training, inference, API requests, dashboard errors, and alert messages. Logs are useful for debugging, monitoring, and portfolio-level production readiness.

---

### 11.1 Log Files

| Log File                        | Purpose                  |
| ------------------------------- | ------------------------ |
| `monitoring/logs/app.log`       | General application logs |
| `monitoring/logs/training.log`  | Model training logs      |
| `monitoring/logs/inference.log` | Prediction pipeline logs |
| `monitoring/logs/api.log`       | API request logs         |
| `monitoring/logs/dashboard.log` | Dashboard logs           |
| `monitoring/logs/errors.log`    | Error logs               |
| `monitoring/logs/alerts.log`    | Alert logs               |

---

### 11.2 Log Levels

| Level    | Meaning                                    |
| -------- | ------------------------------------------ |
| INFO     | Normal successful event                    |
| WARNING  | Something needs attention                  |
| ERROR    | Failed operation                           |
| CRITICAL | Serious failure requiring immediate action |

---

### 11.3 Log Review Checklist

| Check                        | Status  |
| ---------------------------- | ------- |
| App log exists               | Pending |
| Training log exists          | Pending |
| Inference log exists         | Pending |
| API log exists               | Pending |
| Error log reviewed           | Pending |
| No unresolved critical error | Pending |

---

## 12. Alert Dashboard

The alert dashboard defines monitoring rules that notify the user when something important goes wrong in the project. These alerts are useful for detecting data issues, model performance problems, API failures, dashboard failures, and anomaly-related risks.

---

### 12.1 Alert Rules

| Alert                    | Trigger                             | Severity |
| ------------------------ | ----------------------------------- | -------- |
| Missing dataset          | Raw CSV not found                   | Critical |
| Missing required columns | `date`, `asset`, or `price` missing | Critical |
| High missing values      | Missing values `> 5%`               | Warning  |
| Negative prices          | Negative price count `> 0`          | Error    |
| High RMSE                | RMSE `> threshold`                  | Warning  |
| Low R2 Score             | R2 `< threshold`                    | Warning  |
| High anomaly rate        | Anomaly rate `> 20%`                | Warning  |
| High severity anomalies  | High severity count `> 5`           | Warning  |
| Drift detected           | Drift score `> 0.30`                | Warning  |
| API unavailable          | Health endpoint fails               | Critical |
| Dashboard unavailable    | Streamlit app not reachable         | Warning  |

---

### 12.2 Alert Channels

| Channel  | Enabled |
| -------- | ------- |
| Console  | Yes     |
| Log File | Yes     |
| Email    | No      |
| Slack    | No      |

---

## 13. Monitoring Output Files

The monitoring system generates different files for checking performance, drift, alerts, logs, and reports.

| File                                              | Description                    |
| ------------------------------------------------- | ------------------------------ |
| `monitoring/performance_metrics.csv`              | Step-level performance records |
| `monitoring/performance_summary.json`             | Performance summary            |
| `monitoring/drift_report.csv`                     | Data drift report              |
| `monitoring/monitoring_summary.json`              | Full monitoring summary        |
| `monitoring/logs/app.log`                         | Application log                |
| `monitoring/logs/errors.log`                      | Error log                      |
| `monitoring/reports/daily_monitoring_report.md`   | Daily monitoring report        |
| `monitoring/reports/model_performance_report.md`  | Model performance report       |
| `monitoring/reports/anomaly_monitoring_report.md` | Anomaly monitoring report      |

---

## 14. Monitoring Run Order

Use the following run order for complete monitoring:

```bash
python scripts/run_data_pipeline.py
python scripts/run_all_models.py
python scripts/evaluate_models.py
python src/inference/predict.py
python scripts/run_api.py --reload
python scripts/run_dashboard.py
```

---

### 14.1 Run Tests

Use the following command to run all tests:

```bash
pytest tests -v
```

---

### 14.2 API Health Check

API health endpoint:

```text
http://127.0.0.1:8000/health/full
```

---

### 14.3 Dashboard URL

Dashboard URL:

```text
http://localhost:8501
```

---

## 15. Manual Monitoring Checklist

Manual monitoring helps verify the project regularly. This is useful when presenting the project as a job-ready portfolio project.

---

### 15.1 Daily Checklist

| Task                                     | Status  |
| ---------------------------------------- | ------- |
| Check latest dataset availability        | Pending |
| Check missing values and invalid records | Pending |
| Review model prediction outputs          | Pending |
| Review anomaly count and severity        | Pending |
| Check API health endpoint                | Pending |
| Check dashboard loading                  | Pending |
| Review error logs                        | Pending |

---

### 15.2 Weekly Checklist

| Task                                 | Status  |
| ------------------------------------ | ------- |
| Review forecasting model performance | Pending |
| Compare model RMSE and MAE           | Pending |
| Check data drift report              | Pending |
| Review anomaly trends by asset       | Pending |
| Update documentation if needed       | Pending |
| Push updated reports to GitHub       | Pending |

---

### 15.3 Monthly Checklist

| Task                                     | Status  |
| ---------------------------------------- | ------- |
| Retrain models if needed                 | Pending |
| Review feature engineering logic         | Pending |
| Review model artifact size and freshness | Pending |
| Improve dashboard visuals                | Pending |
| Update README and portfolio summary      | Pending |

---

## 16. Troubleshooting Guide

This troubleshooting guide explains common errors and their fixes during monitoring, API execution, model evaluation, and dashboard usage.

---

### Problem: Dataset Missing

Expected file:

```text
data/raw/financial_portfolio_data.csv
```

Fix:

```bash
python scripts/run_data_pipeline.py
```

Or manually place the CSV file inside:

```text
data/raw/
```

---

### Problem: Forecast Model Missing

Expected files:

```text
models/xgboost_model.pkl
models/arima_models.pkl
models/prophet_models.pkl
models/lstm_models/
```

Fix:

```bash
python scripts/run_all_models.py
```

---

### Problem: Anomaly Model Missing

Expected file:

```text
models/anomaly_detection_model.pkl
```

Fix:

```bash
python src/training/train_anomaly_model.py
```

---

### Problem: Evaluation Files Missing

Expected files:

```text
reports/tables/model_comparison.csv
reports/tables/evaluation_summary.json
```

Fix:

```bash
python scripts/evaluate_models.py
```

---

### Problem: API Not Starting

Possible reason:

Required API packages are not installed.

Fix:

```bash
pip install fastapi uvicorn
python scripts/run_api.py --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

### Problem: Dashboard Not Starting

Possible reason:

Required dashboard packages are not installed.

Fix:

```bash
pip install streamlit plotly
python scripts/run_dashboard.py
```

Open:

```text
http://localhost:8501
```

---

## 17. Portfolio Explanation

This monitoring dashboard shows that the project is not just a notebook-based machine learning project. It includes production-style monitoring components such as:

* Data quality checks
* Model performance tracking
* Data drift monitoring
* Anomaly monitoring
* API health checks
* Dashboard health checks
* Runtime performance tracking
* Error logging
* Alert rules
* Reproducible run commands

This makes the project stronger for a Data Analyst, Data Scientist, or AI/ML portfolio.

The monitoring system shows that the project follows a more professional machine learning workflow. Instead of only training models in notebooks, the project also checks whether the data, models, API, dashboard, reports, logs, and alerts are working correctly.

This helps demonstrate:

* Production readiness
* Monitoring awareness
* Model evaluation knowledge
* Dashboard deployment understanding
* API testing understanding
* Real-world machine learning project structure
* Strong portfolio documentation

---

## 18. Final Monitoring Summary

The monitoring system helps answer the following important questions:

* Is the dataset available and valid?
* Are the trained model artifacts available?
* Which forecasting model performs best?
* Are prediction errors acceptable?
* Are anomalies within normal range?
* Has data drift occurred?
* Is the API healthy?
* Is the dashboard healthy?
* Are there unresolved errors?
* Is the project ready for portfolio presentation?

Overall, this monitoring setup improves the reliability, maintainability, and presentation quality of the financial portfolio forecasting and anomaly detection project.

It proves that the project includes not only machine learning model development, but also proper monitoring, logging, alerting, dashboard health checks, API health checks, troubleshooting instructions, and portfolio-ready documentation.

```
```
