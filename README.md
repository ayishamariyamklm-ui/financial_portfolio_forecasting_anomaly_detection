# Financial Portfolio Forecasting & Anomaly Detection

An end-to-end machine learning project for **financial asset price forecasting**, **anomaly detection**, **model evaluation**, **API deployment**, **dashboard visualization**, and **monitoring**.

---

## Project Overview

Financial portfolios contain multiple assets whose prices change over time. Analysts and stakeholders need to understand asset price trends, forecast future values, detect unusual movements, and monitor model performance.

This project solves that problem by building a complete machine learning system that can:

* Clean and process financial portfolio data
* Perform time series exploratory data analysis
* Forecast future asset prices
* Detect unusual price movements
* Compare forecasting models
* Serve predictions through FastAPI
* Visualize insights through Streamlit
* Track logs, monitoring, and performance
* Prepare the project for cloud deployment

---

## Business Problem

Financial asset prices can move normally over time, but sudden spikes, drops, or abnormal patterns may indicate risk, volatility, data issues, or market events.

Manual monitoring is time-consuming and inconsistent.

This project helps answer:

* How are asset prices changing over time?
* Which assets may increase or decrease?
* Which price movements are unusual?
* Which asset records should be reviewed?
* Which forecasting model performs best?
* Can predictions be served through an API?
* Can results be viewed through a business-friendly dashboard?

---

## Project Solution

This project provides:

1. **Asset Price Forecasting**

   * Predicts future asset prices using time series and machine learning models.

2. **Anomaly Detection**

   * Detects unusual asset price movements using IQR, Z-score, and Isolation Forest.

3. **Model Comparison**

   * Compares multiple models using MAE, MSE, RMSE, MAPE, SMAPE, and R2 Score.

4. **API Service**

   * Provides FastAPI endpoints for forecasting and anomaly detection.

5. **Dashboard**

   * Provides Streamlit dashboard for portfolio overview, forecasting, anomaly detection, and model performance.

6. **Monitoring**

   * Tracks logs, performance, drift configuration, and system health.

---

## Key Features

* Modular Python project structure
* Data cleaning pipeline
* Feature engineering pipeline
* Time series EDA notebooks
* ARIMA forecasting model
* Prophet forecasting model
* XGBoost forecasting model
* LSTM forecasting model
* IQR anomaly detection
* Z-score anomaly detection
* Isolation Forest anomaly detection
* Model evaluation and comparison
* FastAPI backend
* Streamlit dashboard
* Pytest test suite
* Docker support
* Render deployment configuration
* Railway deployment configuration
* Monitoring configuration
* Complete documentation

---

## Tech Stack

| Category         | Tools                       |
| ---------------- | --------------------------- |
| Programming      | Python                      |
| Data Processing  | Pandas, NumPy               |
| Visualization    | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn, XGBoost       |
| Time Series      | ARIMA, Prophet              |
| Deep Learning    | TensorFlow / Keras          |
| API              | FastAPI, Uvicorn            |
| Dashboard        | Streamlit                   |
| Testing          | Pytest, HTTPX               |
| Deployment       | Docker, Render, Railway     |
| Monitoring       | Logging, YAML config        |
| Documentation    | Markdown                    |

---

## GitHub Project Structure

```text
financial-portfolio-forecasting-anomaly-detection/
│
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── .env.example
├── Dockerfile
├── docker-compose.yml
│
├── data/
│   ├── raw/
│   │   └── financial_portfolio_data.csv
│   │
│   ├── processed/
│   │   ├── cleaned_financial_data.csv
│   │   ├── daily_asset_prices.csv
│   │   ├── train_data.csv
│   │   ├── test_data.csv
│   │   └── anomaly_results.csv
│   │
│   └── sample/
│       └── sample_prediction_input.csv
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_time_series_eda.ipynb
│   ├── 03_forecasting_experiments.ipynb
│   ├── 04_anomaly_detection_experiments.ipynb
│   └── 05_model_evaluation.ipynb
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
│   ├── pages/
│   └── assets/
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

## Dataset

Raw dataset path:

```text
data/raw/financial_portfolio_data.csv
```

Expected columns:

| Column  | Description                |
| ------- | -------------------------- |
| `Date`  | Date of asset price record |
| `Asset` | Asset name                 |
| `Price` | Asset price                |

After cleaning, columns are standardized to:

```text
date
asset
price
```

---

## Data Processing

The data processing pipeline performs:

* Column name standardization
* Date conversion
* Price conversion
* Missing value removal
* Negative price removal
* Asset-date sorting
* Train-test split
* Processed data saving

Main files:

```text
src/data_processing/data_loader.py
src/data_processing/data_cleaning.py
src/data_processing/feature_engineering.py
src/data_processing/train_test_split.py
```

Run data pipeline:

```bash
python scripts/run_data_pipeline.py
```

Processed outputs:

```text
data/processed/cleaned_financial_data.csv
data/processed/daily_asset_prices.csv
data/processed/train_data.csv
data/processed/test_data.csv
data/processed/anomaly_results.csv
```

---

## Feature Engineering

The project creates time series features such as:

| Feature Type      | Examples                                        |
| ----------------- | ----------------------------------------------- |
| Calendar Features | `year`, `month`, `quarter`, `day_of_week`       |
| Lag Features      | `price_lag_1`, `price_lag_7`, `price_lag_14`    |
| Rolling Mean      | `price_rolling_mean_7`, `price_rolling_mean_30` |
| Rolling Std       | `price_rolling_std_7`, `price_rolling_std_30`   |
| Returns           | `daily_return`                                  |
| Volatility        | `return_rolling_volatility_7`                   |

These features are used for forecasting and anomaly detection.

---

## Notebooks

| Notebook                                 | Purpose                                    |
| ---------------------------------------- | ------------------------------------------ |
| `01_data_understanding.ipynb`            | Dataset inspection and basic understanding |
| `02_time_series_eda.ipynb`               | Time series EDA and visual analysis        |
| `03_forecasting_experiments.ipynb`       | Forecasting model experiments              |
| `04_anomaly_detection_experiments.ipynb` | Anomaly detection experiments              |
| `05_model_evaluation.ipynb`              | Final model evaluation and comparison      |

---

## Forecasting Models

The project trains and compares the following models:

| Model   | Type                         | Purpose                            |
| ------- | ---------------------------- | ---------------------------------- |
| ARIMA   | Statistical time series      | Classical forecasting baseline     |
| Prophet | Time series model            | Trend and seasonality forecasting  |
| XGBoost | Machine learning regression  | Feature-based forecasting          |
| LSTM    | Deep learning sequence model | Sequential time series forecasting |

Model code:

```text
src/models/arima_model.py
src/models/prophet_model.py
src/models/xgboost_model.py
src/models/lstm_model.py
```

Saved model artifacts:

```text
models/forecasting/arima_model.pkl
models/forecasting/prophet_model.pkl
models/forecasting/xgboost_model.pkl
models/forecasting/lstm_model.keras
models/scalers/price_scaler.pkl
```

Train forecasting models:

```bash
python src/training/train_forecasting_models.py
```

Train LSTM model:

```bash
python src/training/train_lstm_model.py
```

Train all models:

```bash
python scripts/train_all_models.py
```

---

## Anomaly Detection

The project detects unusual price movements using:

| Method           | Purpose                                |
| ---------------- | -------------------------------------- |
| IQR              | Detects extreme values using quartiles |
| Z-score          | Detects values far from average        |
| Isolation Forest | Detects unusual multi-feature patterns |

Anomaly model file:

```text
src/models/anomaly_detector.py
```

Saved anomaly model:

```text
models/anomaly/isolation_forest_model.pkl
```

Train anomaly model:

```bash
python src/training/train_anomaly_model.py
```

---

## Model Evaluation

Evaluation includes:

* Forecasting metrics
* Anomaly metrics
* Model comparison
* Best model selection
* Visualization outputs

Evaluation metrics:

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

Run evaluation:

```bash
python scripts/evaluate_models.py
```

Expected outputs:

```text
reports/metrics/forecasting_metrics.json
reports/metrics/anomaly_metrics.json
reports/metrics/model_comparison.csv
reports/figures/asset_price_trends.png
reports/figures/forecast_vs_actual.png
reports/figures/anomaly_detection_plot.png
reports/figures/model_comparison.png
reports/final_project_report.md
```

---

## API Service

The API is built using FastAPI.

Main API file:

```text
api/main.py
```

API app path:

```text
api.main:app
```

Run API locally:

```bash
python scripts/run_api.py --reload
```

Or:

```bash
uvicorn api.main:app --reload
```

Open API docs:

```text
http://127.0.0.1:8000/docs
```

---

## API Endpoints

| Endpoint            | Method | Purpose                                    |
| ------------------- | ------ | ------------------------------------------ |
| `/`                 | GET    | Root endpoint                              |
| `/health/`          | GET    | Basic health check                         |
| `/health/full`      | GET    | Full system health                         |
| `/models/status`    | GET    | Model artifact status                      |
| `/forecast`         | POST   | Forecast using JSON records                |
| `/anomaly`          | POST   | Detect anomalies using JSON records        |
| `/predict`          | POST   | Combined forecasting and anomaly detection |
| `/forecast/default` | GET    | Forecast using default dataset             |
| `/anomaly/default`  | GET    | Detect anomalies using default dataset     |
| `/upload/predict`   | POST   | Upload CSV and run prediction              |
| `/summary/latest`   | GET    | Return latest API summary                  |

---

## Example Forecast API Request

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
      "price": 121.3
    }
  ],
  "model_name": "xgboost",
  "forecast_steps": 30,
  "save_output": true,
  "return_records": true
}
```

---

## Example Anomaly API Request

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
      "price": 500.0
    }
  ],
  "save_output": true,
  "return_records": true,
  "only_anomalies": false
}
```

---

## Dashboard

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
python scripts/run_dashboard.py
```

Or:

```bash
streamlit run dashboard/app.py
```

Open dashboard:

```text
http://localhost:8501
```

---

## Dashboard Pages

| Page               | Purpose                                                         |
| ------------------ | --------------------------------------------------------------- |
| Portfolio Overview | Shows project summary, asset trends, and dataset overview       |
| Price Forecasting  | Shows forecasting outputs and model comparison                  |
| Anomaly Detection  | Shows anomaly records, severity, and asset-wise anomaly summary |
| Model Performance  | Shows model metrics and artifact health                         |

---

## Monitoring

Monitoring files:

```text
monitoring/monitoring_config.yaml
monitoring/performance_dashboard.md
monitoring/logs/app.log
```

Monitoring covers:

* Data quality
* Forecasting performance
* Anomaly rate
* Drift detection configuration
* API health
* Dashboard health
* Runtime performance
* Logs and alerts

---

## Testing

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

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-username/financial-portfolio-forecasting-anomaly-detection.git
cd financial-portfolio-forecasting-anomaly-detection
```

### 2. Create Virtual Environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Mac/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Requirements

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Complete Run Order

Use this order to run the full project:

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

## Docker Usage

Build Docker image:

```bash
docker build -t financial-portfolio-api .
```

Run Docker container:

```bash
docker run -p 8000:8000 financial-portfolio-api
```

Open API docs:

```text
http://127.0.0.1:8000/docs
```

Run with Docker Compose:

```bash
docker-compose up --build
```

---

## Cloud Deployment

Deployment files:

```text
deployment/render.yaml
deployment/railway.json
deployment/cloud_deployment_guide.md
deployment/production_notes.md
```

### Render FastAPI Start Command

```bash
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

### Render Streamlit Start Command

```bash
streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

### Railway Start Command

```bash
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

---

## Documentation

Project documentation is stored in:

```text
docs/
```

| File                   | Purpose                          |
| ---------------------- | -------------------------------- |
| `architecture.md`      | Project architecture             |
| `data_dictionary.md`   | Dataset and feature descriptions |
| `methodology.md`       | Project methodology              |
| `api_documentation.md` | API usage and endpoints          |
| `model_report.md`      | Model details and evaluation     |
| `deployment_guide.md`  | Deployment instructions          |
| `troubleshooting.md`   | Common errors and fixes          |
| `business_impact.md`   | Business value explanation       |

---

## Business Impact

This project helps stakeholders:

* Understand portfolio asset trends
* Forecast future asset prices
* Detect unusual price movements
* Review risky or suspicious records
* Compare model performance
* Use predictions through API endpoints
* View results through an interactive dashboard

Potential users:

| User                 | Use Case                                   |
| -------------------- | ------------------------------------------ |
| Data Analyst         | Analyze asset trends and anomalies         |
| Data Scientist       | Train and compare forecasting models       |
| Risk Analyst         | Monitor unusual price movements            |
| Portfolio Manager    | Review forecast and anomaly summaries      |
| Business Stakeholder | View dashboard insights                    |
| Developer            | Integrate API predictions into another app |

---

## Project Outputs

| Output Type      | Location                                                              |
| ---------------- | --------------------------------------------------------------------- |
| Cleaned Data     | `data/processed/`                                                     |
| Forecast Results | `outputs/`                                                            |
| Anomaly Results  | `outputs/`                                                            |
| Model Artifacts  | `models/`                                                             |
| Metrics          | `reports/metrics/`                                                    |
| Figures          | `reports/figures/`                                                    |
| Final Report     | `reports/final_project_report.md`                                     |
| API Output       | `outputs/api_forecast_results.csv`, `outputs/api_anomaly_results.csv` |
| Dashboard        | `dashboard/app.py`                                                    |

---

## Screenshots

Add dashboard screenshots here after running the app.

Example:

```markdown
![Dashboard Preview](dashboard/assets/dashboard_preview.png)
```

Expected screenshot file:

```text
dashboard/assets/dashboard_preview.png
```

---

## Troubleshooting

### Dataset Missing

Make sure this file exists:

```text
data/raw/financial_portfolio_data.csv
```

### API Not Starting

Install requirements:

```bash
pip install -r requirements.txt
```

Run API:

```bash
uvicorn api.main:app --reload
```

### Dashboard Not Starting

Install dashboard packages:

```bash
pip install streamlit plotly
```

Run dashboard:

```bash
streamlit run dashboard/app.py
```

### Model Files Missing

Train models:

```bash
python scripts/train_all_models.py
```

### Tests Failing Due to Missing `httpx`

Install:

```bash
pip install httpx
```

---

## Limitations

Important limitations:

* Forecasts are estimates, not guarantees.
* This project does not provide financial advice.
* The dataset is CSV-based and static.
* No live financial data API is included by default.
* No database is included by default.
* API authentication is not enabled by default.
* Model performance may degrade over time.
* Anomaly detection may produce false positives.

Disclaimer:

```text
This project is for educational and portfolio purposes only. It does not provide financial advice.
```

---

## Future Improvements

Recommended future improvements:

* Add live financial market data integration
* Add PostgreSQL database
* Add model versioning
* Add scheduled retraining
* Add API authentication
* Add rate limiting
* Add cloud model storage
* Add GitHub Actions CI/CD
* Add automated drift monitoring
* Add email alerts for high-severity anomalies
* Add PDF report generation
* Add SHAP explainability

---

## Portfolio Summary

This project demonstrates practical skills in:

* Python programming
* Data cleaning
* Time series analysis
* Feature engineering
* Machine learning
* Deep learning
* Anomaly detection
* Model evaluation
* FastAPI development
* Streamlit dashboarding
* Testing with Pytest
* Docker deployment
* Cloud deployment preparation
* Monitoring and documentation

---

## Final Project Statement

**Financial Portfolio Forecasting & Anomaly Detection** is a complete machine learning portfolio project that transforms raw financial asset data into forecasting insights, anomaly alerts, API predictions, dashboard visualizations, and monitoring reports.

It is designed to demonstrate end-to-end project development from raw data to deployment-ready machine learning application.

---
