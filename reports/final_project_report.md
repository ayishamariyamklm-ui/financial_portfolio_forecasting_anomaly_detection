# Final Project Report

## Financial Portfolio Forecasting & Anomaly Detection

**Project Type:** End-to-End Machine Learning Portfolio Project
**Domain:** Finance / Time Series / Risk Monitoring
**Prepared By:** Ayisha Mariyam
**Project Folder:** `financial-portfolio-forecasting-anomaly-detection`

---

## Table of Contents

1. Project Overview
2. Business Problem
3. Project Objectives
4. Dataset Description
5. Project Architecture
6. Data Processing
7. Exploratory Data Analysis
8. Feature Engineering
9. Forecasting Models
10. Anomaly Detection
11. Model Evaluation
12. API Deployment
13. Dashboard Visualization
14. Monitoring and Logging
15. Testing
16. Deployment Readiness
17. Business Impact
18. Limitations
19. Future Improvements
20. Final Conclusion

---

## 1. Project Overview

The **Financial Portfolio Forecasting & Anomaly Detection** project is an end-to-end machine learning system designed to analyze financial asset prices, forecast future price movements, and detect unusual asset price behavior.

This project demonstrates a production-style workflow that includes:

* Data loading
* Data cleaning
* Feature engineering
* Time series analysis
* Forecasting model training
* Anomaly detection
* Model comparison
* API-based prediction service
* Streamlit dashboard
* Monitoring and logging
* Testing
* Deployment configuration
* Documentation

The project is designed to be suitable for a professional **Data Science**, **Data Analytics**, or **AI/ML Engineering** portfolio.

---

## 2. Business Problem

Financial portfolios usually contain multiple assets whose prices change over time. These price movements may be normal market behavior, but some movements may be unusual and require further review.

Businesses, analysts, and portfolio managers need answers to questions such as:

* How are asset prices moving over time?
* Which assets show increasing or decreasing trends?
* Which forecasting model gives better future price estimates?
* Which price movements are unusual?
* Which assets require closer monitoring?
* Can predictions be served through an API?
* Can business users view insights through a dashboard?

Manual analysis can be time-consuming and inconsistent. This project solves the problem by creating an automated machine learning system for forecasting and anomaly detection.

---

## 3. Project Objectives

The main objectives of this project are:

1. Load and clean financial asset price data.
2. Perform exploratory data analysis on asset price trends.
3. Create time series features for machine learning.
4. Train and compare multiple forecasting models.
5. Detect anomalies using statistical and machine learning methods.
6. Evaluate model performance using standard metrics.
7. Build an API service for prediction and anomaly detection.
8. Build a dashboard for business-friendly visualization.
9. Add monitoring and logging support.
10. Prepare the project for GitHub portfolio and cloud deployment.

---

## 4. Dataset Description

### 4.1 Raw Dataset

Raw dataset path:

```text
data/raw/financial_portfolio_data.csv
```

Expected columns:

| Column  | Description                    |
| ------- | ------------------------------ |
| `Date`  | Date of the asset price record |
| `Asset` | Name of the financial asset    |
| `Price` | Asset price on that date       |

After cleaning, columns are standardized as:

| Standard Column | Description                    |
| --------------- | ------------------------------ |
| `date`          | Date of the asset price record |
| `asset`         | Name of the financial asset    |
| `price`         | Asset price                    |

---

### 4.2 Processed Dataset Files

Processed files are stored in:

```text
data/processed/
```

| File                         | Purpose                      |
| ---------------------------- | ---------------------------- |
| `cleaned_financial_data.csv` | Cleaned raw data             |
| `daily_asset_prices.csv`     | Daily asset-level price data |
| `train_data.csv`             | Training dataset             |
| `test_data.csv`              | Testing dataset              |
| `anomaly_results.csv`        | Anomaly detection output     |

---

### 4.3 Sample Input File

Sample prediction input:

```text
data/sample/sample_prediction_input.csv
```

This file can be used for API testing and inference.

---

## 5. Project Architecture

The project follows a modular machine learning architecture.

```text
Raw CSV Data
    |
    v
Data Processing
    |
    v
Feature Engineering
    |
    v
Model Training
    |
    +-------------------------+
    |                         |
    v                         v
Forecasting Models       Anomaly Detection
    |                         |
    v                         v
Forecast Outputs         Anomaly Outputs
    |                         |
    +-----------+-------------+
                |
                v
Evaluation and Reports
                |
                v
API + Dashboard + Monitoring
```

---

### 5.1 Main Project Structure

```text
financial-portfolio-forecasting-anomaly-detection/
│
├── data/
├── notebooks/
├── src/
├── api/
├── dashboard/
├── models/
├── reports/
├── tests/
├── scripts/
├── config/
├── deployment/
├── monitoring/
└── docs/
```

---

## 6. Data Processing

Data processing is handled inside:

```text
src/data_processing/
```

Main files:

| File                     | Purpose                      |
| ------------------------ | ---------------------------- |
| `data_loader.py`         | Loads raw CSV data           |
| `data_cleaning.py`       | Cleans and validates data    |
| `feature_engineering.py` | Creates time series features |
| `train_test_split.py`    | Creates train/test split     |

---

### 6.1 Data Cleaning Steps

The project applies the following cleaning steps:

1. Load raw CSV file.
2. Standardize column names.
3. Convert `date` to datetime.
4. Convert `price` to numeric.
5. Remove missing values.
6. Remove negative prices.
7. Sort data by asset and date.
8. Save cleaned output.

---

### 6.2 Data Quality Rules

| Rule             | Expected Condition                  |
| ---------------- | ----------------------------------- |
| Required columns | `date`, `asset`, `price` must exist |
| Date format      | Must be convertible to datetime     |
| Price format     | Must be numeric                     |
| Missing values   | Removed from required columns       |
| Negative prices  | Removed                             |
| Empty dataset    | Should raise an error               |

---

## 7. Exploratory Data Analysis

EDA is performed using the notebooks:

```text
notebooks/01_data_understanding.ipynb
notebooks/02_time_series_eda.ipynb
```

The EDA includes:

* Dataset shape analysis
* Missing value analysis
* Duplicate row check
* Asset count
* Date range
* Price distribution
* Asset-wise summaries
* Price trend visualization
* Daily return analysis
* Rolling average analysis
* Volatility analysis

---

### 7.1 EDA Output Figures

Expected figures:

```text
reports/figures/asset_price_trends.png
reports/figures/forecast_vs_actual.png
reports/figures/anomaly_detection_plot.png
reports/figures/model_comparison.png
```

---

## 8. Feature Engineering

Feature engineering creates additional columns for forecasting and anomaly detection.

---

### 8.1 Calendar Features

| Feature        | Description       |
| -------------- | ----------------- |
| `year`         | Year from date    |
| `month`        | Month from date   |
| `quarter`      | Quarter from date |
| `day`          | Day of month      |
| `day_of_week`  | Day of week       |
| `week_of_year` | ISO week number   |

---

### 8.2 Lag Features

| Feature        | Description           |
| -------------- | --------------------- |
| `price_lag_1`  | Previous 1-day price  |
| `price_lag_3`  | Previous 3-day price  |
| `price_lag_7`  | Previous 7-day price  |
| `price_lag_14` | Previous 14-day price |

Lag features help machine learning models understand recent price history.

---

### 8.3 Rolling Features

| Feature                 | Description                       |
| ----------------------- | --------------------------------- |
| `price_rolling_mean_7`  | 7-day rolling average             |
| `price_rolling_mean_14` | 14-day rolling average            |
| `price_rolling_mean_30` | 30-day rolling average            |
| `price_rolling_std_7`   | 7-day rolling standard deviation  |
| `price_rolling_std_14`  | 14-day rolling standard deviation |
| `price_rolling_std_30`  | 30-day rolling standard deviation |

---

### 8.4 Return and Volatility Features

| Feature                        | Description                |
| ------------------------------ | -------------------------- |
| `daily_return`                 | Percentage change in price |
| `return_rolling_volatility_7`  | 7-day return volatility    |
| `return_rolling_volatility_14` | 14-day return volatility   |
| `return_rolling_volatility_30` | 30-day return volatility   |

---

## 9. Forecasting Models

Forecasting model code is stored in:

```text
src/models/
```

Model files:

| File               | Model   |
| ------------------ | ------- |
| `arima_model.py`   | ARIMA   |
| `prophet_model.py` | Prophet |
| `xgboost_model.py` | XGBoost |
| `lstm_model.py`    | LSTM    |

---

### 9.1 ARIMA Model

ARIMA is a statistical time series forecasting model.

Strengths:

* Good classical baseline
* Interpretable
* Useful for simple trends

Limitations:

* May not capture complex non-linear behavior
* Usually needs one model per asset
* Requires sufficient historical data

---

### 9.2 Prophet Model

Prophet is a forecasting model that handles trend and seasonality.

Strengths:

* Business-friendly
* Handles trend patterns
* Useful for time series with seasonality

Limitations:

* Can be heavier than simple models
* May require enough historical records

---

### 9.3 XGBoost Model

XGBoost is a machine learning regression model.

Strengths:

* Strong performance
* Handles non-linear patterns
* Works well with engineered features

Limitations:

* Requires feature engineering
* Does not naturally understand time order unless lag features are created

---

### 9.4 LSTM Model

LSTM is a deep learning sequence model.

Strengths:

* Learns sequential patterns
* Can model complex temporal relationships
* Useful for advanced time series forecasting

Limitations:

* Needs more data
* Needs scaling
* More computationally expensive
* Can overfit on small datasets

---

## 10. Anomaly Detection

Anomaly detection is handled by:

```text
src/models/anomaly_detector.py
```

Saved anomaly model:

```text
models/anomaly/isolation_forest_model.pkl
```

The project uses three anomaly detection methods:

1. IQR
2. Z-score
3. Isolation Forest

---

### 10.1 IQR Method

IQR detects price values outside the normal quartile range.

Formula:

```text
IQR = Q3 - Q1
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

A record is an anomaly if:

```text
price < lower_bound
or
price > upper_bound
```

---

### 10.2 Z-Score Method

Z-score measures how far a value is from the mean.

Formula:

```text
z_score = (price - mean) / standard_deviation
```

A record is an anomaly if:

```text
absolute z_score > 3
```

---

### 10.3 Isolation Forest

Isolation Forest is a machine learning anomaly detection algorithm.

It detects unusual records using features such as:

* Price
* Daily return
* Lag features
* Rolling mean
* Rolling standard deviation
* Volatility

---

### 10.4 Final Anomaly Flag

The final anomaly flag is created by combining methods:

```text
is_anomaly = True if any method detects anomaly
```

Severity levels:

| Severity | Meaning                   |
| -------- | ------------------------- |
| `normal` | No anomaly detected       |
| `low`    | Detected by one method    |
| `medium` | Detected by two methods   |
| `high`   | Detected by three methods |

---

## 11. Model Evaluation

Evaluation code is stored in:

```text
src/evaluation/
```

Files:

| File                  | Purpose                       |
| --------------------- | ----------------------------- |
| `metrics.py`          | Calculates evaluation metrics |
| `model_comparison.py` | Compares models               |
| `visualization.py`    | Creates evaluation charts     |

Evaluation script:

```text
scripts/evaluate_models.py
```

---

### 11.1 Forecasting Metrics

| Metric   | Meaning                        | Better Direction |
| -------- | ------------------------------ | ---------------- |
| MAE      | Mean Absolute Error            | Lower            |
| MSE      | Mean Squared Error             | Lower            |
| RMSE     | Root Mean Squared Error        | Lower            |
| MAPE     | Mean Absolute Percentage Error | Lower            |
| SMAPE    | Symmetric MAPE                 | Lower            |
| R2 Score | Model explanation power        | Higher           |

Primary model selection metric:

```text
RMSE
```

Reason:

RMSE penalizes large prediction errors, which is useful in financial forecasting.

---

### 11.2 Model Comparison Output

Expected file:

```text
reports/metrics/model_comparison.csv
```

Template:

| Model   |           MAE |           MSE |          RMSE |          MAPE |         SMAPE |      R2 Score |          Rank |
| ------- | ------------: | ------------: | ------------: | ------------: | ------------: | ------------: | ------------: |
| ARIMA   | To be updated | To be updated | To be updated | To be updated | To be updated | To be updated | To be updated |
| Prophet | To be updated | To be updated | To be updated | To be updated | To be updated | To be updated | To be updated |
| XGBoost | To be updated | To be updated | To be updated | To be updated | To be updated | To be updated | To be updated |
| LSTM    | To be updated | To be updated | To be updated | To be updated | To be updated | To be updated | To be updated |

---

### 11.3 Best Model Selection

Best model rule:

```text
Lowest RMSE wins
```

Best model:

```text
To be updated after running scripts/evaluate_models.py
```

---

### 11.4 Anomaly Metrics

Expected file:

```text
reports/metrics/anomaly_metrics.json
```

Metrics:

| Metric                | Description                   |
| --------------------- | ----------------------------- |
| Total Records         | Number of records checked     |
| Normal Records        | Number of non-anomaly records |
| Total Anomalies       | Number of anomaly records     |
| Anomaly Rate          | Percentage of anomaly records |
| Method Summary        | Count by method               |
| Asset Summary         | Count by asset                |
| Severity Distribution | Count by severity level       |

---

## 12. API Deployment

The API is built using FastAPI.

Main API file:

```text
api/main.py
```

API app path:

```text
api.main:app
```

Run locally:

```bash
uvicorn api.main:app --reload
```

or:

```bash
python scripts/run_api.py --reload
```

API docs:

```text
http://127.0.0.1:8000/docs
```

---

### 12.1 API Endpoints

| Endpoint            | Method | Purpose                                     |
| ------------------- | ------ | ------------------------------------------- |
| `/`                 | GET    | Root endpoint                               |
| `/health/`          | GET    | Basic health check                          |
| `/health/full`      | GET    | Full health check                           |
| `/models/status`    | GET    | Model artifact status                       |
| `/forecast`         | POST   | Forecast using JSON records                 |
| `/anomaly`          | POST   | Detect anomalies using JSON records         |
| `/predict`          | POST   | Run forecast and anomaly detection together |
| `/forecast/default` | GET    | Forecast using default dataset              |
| `/anomaly/default`  | GET    | Detect anomalies using default dataset      |
| `/upload/predict`   | POST   | Upload CSV and run prediction               |

---

### 12.2 API Input Example

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
  "save_output": true,
  "return_records": true
}
```

---

## 13. Dashboard Visualization

The dashboard is built using Streamlit.

Main dashboard file:

```text
dashboard/app.py
```

Dashboard pages:

```text
dashboard/pages/1_Portfolio_Overview.py
dashboard/pages/2_Price_Forecasting.py
dashboard/pages/3_Anomaly_Detection.py
dashboard/pages/4_Model_Performance.py
```

Run dashboard:

```bash
streamlit run dashboard/app.py
```

or:

```bash
python scripts/run_dashboard.py
```

Dashboard URL:

```text
http://localhost:8501
```

---

### 13.1 Dashboard Pages

| Page               | Purpose                                  |
| ------------------ | ---------------------------------------- |
| Portfolio Overview | Shows project and dataset summary        |
| Price Forecasting  | Shows forecasting outputs and comparison |
| Anomaly Detection  | Shows anomaly results                    |
| Model Performance  | Shows metrics and model artifact health  |

---

## 14. Monitoring and Logging

Monitoring files:

```text
monitoring/monitoring_config.yaml
monitoring/performance_dashboard.md
monitoring/logs/app.log
```

Monitoring tracks:

* Data quality
* Forecasting performance
* Anomaly rate
* Data drift
* API health
* Dashboard health
* Runtime performance
* Error logs

---

### 14.1 Log Files

| Log File                        | Purpose                  |
| ------------------------------- | ------------------------ |
| `monitoring/logs/app.log`       | General application logs |
| `monitoring/logs/training.log`  | Model training logs      |
| `monitoring/logs/inference.log` | Inference logs           |
| `monitoring/logs/api.log`       | API logs                 |
| `monitoring/logs/dashboard.log` | Dashboard logs           |
| `monitoring/logs/errors.log`    | Error logs               |

---

## 15. Testing

Test files:

```text
tests/test_data_cleaning.py
tests/test_feature_engineering.py
tests/test_forecast_service.py
tests/test_anomaly_service.py
tests/test_api.py
```

Run all tests:

```bash
pytest tests -v
```

Run individual tests:

```bash
pytest tests/test_data_cleaning.py -v
pytest tests/test_feature_engineering.py -v
pytest tests/test_forecast_service.py -v
pytest tests/test_anomaly_service.py -v
pytest tests/test_api.py -v
```

Testing coverage includes:

* Data cleaning
* Feature engineering
* Forecast service
* Anomaly service
* API endpoints
* Schema validation

---

## 16. Deployment Readiness

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

| Platform | Purpose                  |
| -------- | ------------------------ |
| Render   | Deploy API and dashboard |
| Railway  | Deploy API               |
| Docker   | Containerized deployment |
| GitHub   | Portfolio repository     |

---

### 16.1 Render Start Commands

FastAPI:

```bash
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

Streamlit:

```bash
streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

---

### 16.2 Railway Start Command

```bash
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

---

### 16.3 Docker Commands

Build:

```bash
docker build -t financial-portfolio-api .
```

Run:

```bash
docker run -p 8000:8000 financial-portfolio-api
```

---

## 17. Business Impact

This project provides business value by converting raw financial data into useful insights.

---

### 17.1 Forecasting Impact

Forecasting helps users:

* Estimate future asset prices
* Monitor expected asset direction
* Compare assets
* Support planning
* Identify assets needing attention

---

### 17.2 Anomaly Detection Impact

Anomaly detection helps users:

* Detect sudden price spikes
* Detect sudden price drops
* Identify suspicious records
* Find possible data quality problems
* Prioritize records for review

---

### 17.3 Dashboard Impact

The dashboard helps business users:

* View trends without coding
* Understand model results visually
* Review anomaly records
* Download outputs
* Present insights to stakeholders

---

### 17.4 API Impact

The API helps developers:

* Integrate predictions into other systems
* Automate forecasting workflows
* Run anomaly detection from external apps
* Use the model as a service

---

## 18. Limitations

Current limitations:

* Static CSV-based dataset
* No live market data integration
* No database storage
* No API authentication
* No scheduled retraining
* No production model registry
* Forecasts are estimates and may be wrong
* Anomaly detection can produce false positives

Important disclaimer:

```text
This project is for educational and portfolio purposes only. It does not provide financial advice.
```

---

## 19. Future Improvements

Recommended future improvements:

| Improvement                | Benefit                          |
| -------------------------- | -------------------------------- |
| Live financial data API    | More current insights            |
| Database integration       | Better data storage              |
| Model retraining scheduler | Keeps models updated             |
| API authentication         | More secure deployment           |
| Rate limiting              | Prevents abuse                   |
| Model versioning           | Better production tracking       |
| Drift detection automation | Better model monitoring          |
| Email alerts               | Faster anomaly response          |
| PDF report generation      | Better stakeholder reporting     |
| CI/CD with GitHub Actions  | Automated testing and deployment |

---

## 20. Final Conclusion

The **Financial Portfolio Forecasting & Anomaly Detection** project demonstrates a complete machine learning workflow from raw data to deployment.

The project includes:

* Data cleaning
* Feature engineering
* Time series EDA
* Multiple forecasting models
* Anomaly detection
* Model evaluation
* API deployment
* Dashboard visualization
* Monitoring configuration
* Testing
* Documentation
* Cloud deployment files

This makes the project suitable for a professional portfolio and demonstrates practical skills in:

* Python
* Pandas
* Machine learning
* Time series forecasting
* Anomaly detection
* FastAPI
* Streamlit
* Testing
* Deployment
* Monitoring
* Documentation

Final project status:

```text
Portfolio-ready after running all scripts, generating outputs, and updating model metrics.
```

Recommended final run order:

```bash
python scripts/run_data_pipeline.py
python scripts/train_all_models.py
python scripts/evaluate_models.py
python src/inference/predict.py
pytest tests -v
python scripts/run_api.py --reload
python scripts/run_dashboard.py
```

---
