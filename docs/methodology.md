# Methodology

## Financial Portfolio Forecasting & Anomaly Detection

This document explains the methodology used to build the **Financial Portfolio Forecasting & Anomaly Detection** project from data ingestion to deployment.

---

## 1. Objective

The objective of this project is to build an end-to-end financial portfolio analytics system that can:

1. Analyze historical asset prices
2. Forecast future asset prices
3. Detect unusual price movements
4. Compare multiple forecasting models
5. Expose predictions through an API
6. Visualize results in a dashboard

The project is designed to demonstrate practical skills in data preprocessing, time series forecasting, anomaly detection, model evaluation, API development, dashboard development, monitoring, and deployment.

---

## 2. Business Problem

Financial asset prices change over time due to market behavior, economic conditions, and asset-specific events.

This project helps answer:

* How are asset prices changing over time?
* Which forecasting model performs best?
* Which assets show unusual movements?
* Which records should be reviewed as anomalies?
* How can model outputs be served through an API and dashboard?

The system supports better portfolio monitoring by combining forecasting, anomaly detection, model evaluation, and visual reporting.

---

## 3. Dataset

Raw dataset path:

```text
data/raw/financial_portfolio_data.csv
```

Required columns:

| Column  | Description          |
| ------- | -------------------- |
| `date`  | Date of price record |
| `asset` | Asset name           |
| `price` | Asset price          |

The raw dataset contains historical price records for different financial assets. Each row represents the price of one asset on a specific date.

---

## 4. Data Cleaning Methodology

Data cleaning is performed to ensure the dataset is usable for time series analysis and machine learning.

### Steps

1. Load raw CSV file.
2. Standardize column names.
3. Convert `date` column to datetime.
4. Convert `price` column to numeric.
5. Remove missing values.
6. Remove negative prices.
7. Sort data by `asset` and `date`.
8. Save cleaned data.

Output:

```text
data/processed/cleaned_financial_data.csv
```

### Purpose of Data Cleaning

Data cleaning ensures that:

* Dates are valid.
* Prices are numeric.
* Missing required values are removed.
* Invalid negative prices are removed.
* Records are sorted correctly for time series analysis.
* The dataset is ready for feature engineering and modeling.

---

## 5. Exploratory Data Analysis

Exploratory Data Analysis, also called EDA, is performed to understand the dataset before model training.

EDA is performed in:

```text
notebooks/01_data_understanding.ipynb
notebooks/02_time_series_eda.ipynb
```

EDA includes:

* Dataset shape inspection
* Missing value check
* Duplicate check
* Asset count
* Date range
* Price distribution
* Asset-wise price summary
* Price trend visualization
* Daily return analysis
* Rolling average analysis
* Volatility analysis

### Purpose of EDA

EDA helps identify:

* Data quality issues
* Price trends
* Asset-level differences
* Volatility patterns
* Missing values
* Outliers
* Model feature opportunities

---

## 6. Feature Engineering Methodology

Feature engineering is required because machine learning models need useful numerical inputs. Since financial data is time-based, the project creates calendar, lag, rolling, return, and volatility features.

---

### 6.1 Calendar Features

Calendar features help models understand time-based patterns.

Features created:

* Year
* Month
* Quarter
* Day
* Day of week
* Week of year
* Month start flag
* Month end flag

Example columns:

```text
year
month
quarter
day
day_of_week
week_of_year
is_month_start
is_month_end
```

---

### 6.2 Lag Features

Lag features help the model learn from previous prices.

Features created:

* Previous 1-day price
* Previous 3-day price
* Previous 7-day price
* Previous 14-day price

Example columns:

```text
price_lag_1
price_lag_3
price_lag_7
price_lag_14
```

Lag features are calculated separately for each asset to avoid mixing price history between different assets.

---

### 6.3 Rolling Features

Rolling features help capture recent trends and volatility.

Features created:

* 7-day rolling mean
* 14-day rolling mean
* 30-day rolling mean
* 7-day rolling standard deviation
* 14-day rolling standard deviation
* 30-day rolling standard deviation

Example columns:

```text
price_rolling_mean_7
price_rolling_mean_14
price_rolling_mean_30
price_rolling_std_7
price_rolling_std_14
price_rolling_std_30
```

---

### 6.4 Return Features

Return features measure price movement from one record to the next.

Features created:

* Daily return
* Rolling volatility

Example columns:

```text
daily_return
return_rolling_volatility_7
return_rolling_volatility_14
return_rolling_volatility_30
```

Formula for daily return:

```text
daily_return = (current_price - previous_price) / previous_price
```

---

## 7. Train-Test Split Methodology

Since this is a time series project, random splitting is not used.

Correct approach:

```text
Older records → Training data
Newer records → Testing data
```

Default split:

| Dataset       | Percentage |
| ------------- | ---------- |
| Training data | 80%        |
| Testing data  | 20%        |

### Why Time-Based Split Is Important

Time-based splitting is important because it:

* Prevents data leakage.
* Simulates real-world forecasting.
* Ensures the model predicts future data, not shuffled historical data.
* Gives a more realistic evaluation of forecasting performance.

Output files:

```text
data/processed/train_data.csv
data/processed/test_data.csv
```

---

## 8. Forecasting Methodology

Forecasting is used to predict future asset prices based on historical trends and engineered features.

Forecasting experiments are performed in:

```text
notebooks/03_forecasting_experiments.ipynb
```

Forecasting model files:

```text
src/models/arima_model.py
src/models/prophet_model.py
src/models/xgboost_model.py
src/models/lstm_model.py
```

The project compares both traditional time series models and machine learning models.

---

## 9. Forecasting Models

The project uses multiple models to compare forecasting performance.

---

### 9.1 Baseline Models

Baseline models are simple forecasting methods used for comparison.

Examples:

* Previous price baseline
* Rolling mean baseline

Purpose:

* Gives a simple benchmark.
* Helps compare advanced models against simple methods.
* Shows whether complex models are actually improving predictions.

---

### 9.2 ARIMA

ARIMA is a statistical time series forecasting model.

Used for:

* Single asset time series forecasting
* Trend-based forecasting
* Classical time series baseline

Strengths:

* Good for simple time series.
* Interpretable.
* Works without many engineered features.
* Useful as a traditional forecasting benchmark.

Limitations:

* May not perform well on highly non-linear data.
* Requires enough historical records.
* Usually trained per asset.
* May require parameter tuning.

---

### 9.3 Prophet

Prophet is a time series forecasting model designed for trend and seasonality.

Used for:

* Forecasting asset prices with trend
* Handling seasonal patterns
* Business-friendly time series forecasting

Strengths:

* Handles trend and seasonality.
* Easy to use.
* Good for business forecasting.
* Can create future forecasts with confidence intervals.

Limitations:

* May be heavy for deployment.
* Requires enough historical data.
* May not perform well when financial price movements are highly volatile.
* Can require tuning for better results.

---

### 9.4 XGBoost

XGBoost is a machine learning regression model.

Used for:

* Feature-based price forecasting
* Learning non-linear relationships
* Predicting price using lag, rolling, return, and calendar features

Strengths:

* Strong predictive performance.
* Works well with engineered features.
* Faster than deep learning models.
* Handles non-linear patterns well.

Limitations:

* Requires feature engineering.
* Does not naturally understand time order unless lag features are created.
* Needs careful validation to avoid leakage.
* Requires tuning for best performance.

---

### 9.5 LSTM

LSTM is a deep learning sequence model.

Used for:

* Learning sequential patterns
* Forecasting based on price sequences
* Capturing time dependencies

Strengths:

* Can model complex time dependencies.
* Useful for long-term sequence learning.
* Can learn patterns from historical windows.

Limitations:

* Requires more data.
* Requires more computing resources.
* Needs scaling.
* Can overfit on small datasets.
* More difficult to explain than traditional models.

---

## 10. Forecasting Evaluation

Forecasting models are evaluated using multiple error metrics.

| Metric   | Meaning                    | Better Direction |
| -------- | -------------------------- | ---------------- |
| MAE      | Average absolute error     | Lower            |
| MSE      | Average squared error      | Lower            |
| RMSE     | Penalizes large errors     | Lower            |
| MAPE     | Percentage error           | Lower            |
| SMAPE    | Symmetric percentage error | Lower            |
| R2 Score | Explanation power          | Higher           |

Primary selection metric:

```text
RMSE
```

Reason:

* RMSE penalizes large prediction mistakes.
* It is useful for financial forecasting where large errors matter.
* It helps compare models based on prediction accuracy.

Evaluation outputs:

```text
reports/metrics/forecasting_metrics.json
reports/metrics/model_comparison.csv
reports/figures/model_comparison.png
reports/figures/forecast_vs_actual.png
```

---

## 11. Anomaly Detection Methodology

Anomaly detection is used to identify unusual price records or abnormal asset movements.

Anomaly detection experiments are performed in:

```text
notebooks/04_anomaly_detection_experiments.ipynb
```

Anomaly model file:

```text
src/models/anomaly_detector.py
```

Anomaly detection methods:

* IQR
* Z-score
* Isolation Forest

---

## 12. IQR Anomaly Detection

IQR detects values outside the normal quartile range.

Formula:

```text
IQR = Q3 - Q1

Lower Bound = Q1 - 1.5 × IQR

Upper Bound = Q3 + 1.5 × IQR
```

A value is an anomaly if:

```text
price < lower_bound
```

or

```text
price > upper_bound
```

Strengths:

* Simple.
* Easy to explain.
* Good for extreme values.
* Works without model training.

Limitations:

* May not detect complex anomalies.
* Sensitive to skewed distributions.
* Uses only statistical boundaries.
* May not capture time-based abnormal behavior.

---

## 13. Z-Score Anomaly Detection

Z-score measures how far a value is from the mean.

Formula:

```text
z_score = (price - mean) / standard_deviation
```

Rule:

```text
absolute z-score > 3
```

Strengths:

* Easy to interpret.
* Good for normally distributed data.
* Simple to implement.
* Useful for detecting extreme deviations.

Limitations:

* Less effective for skewed data.
* Sensitive to outliers.
* Assumes the mean and standard deviation are meaningful.
* May not capture complex anomaly patterns.

---

## 14. Isolation Forest

Isolation Forest is a machine learning anomaly detection algorithm.

It detects unusual records based on multiple features.

Features used:

* Price
* Daily return
* Lag features
* Rolling mean
* Rolling standard deviation
* Volatility
* Calendar features

Strengths:

* Detects complex anomalies.
* Uses multiple features.
* Works well for unsupervised anomaly detection.
* Can identify unusual combinations of feature values.

Limitations:

* Less explainable than IQR or Z-score.
* Requires contamination parameter tuning.
* Results may vary depending on feature quality.
* Requires careful interpretation.

---

## 15. Combined Anomaly Flag

The final anomaly flag combines the results of multiple anomaly detection methods.

Final anomaly logic:

```text
is_anomaly = True if any method detects anomaly
```

Severity mapping:

| Method Count | Severity |
| ------------ | -------- |
| 0            | normal   |
| 1            | low      |
| 2            | medium   |
| 3            | high     |

This helps prioritize which anomalies need review first.

Example:

* If only IQR detects an anomaly, severity is `low`.
* If IQR and Z-score detect an anomaly, severity is `medium`.
* If IQR, Z-score, and Isolation Forest detect an anomaly, severity is `high`.

---

## 16. Model Evaluation Methodology

Final evaluation is performed in:

```text
notebooks/05_model_evaluation.ipynb
scripts/evaluate_models.py
```

Evaluation includes:

* Forecasting model comparison
* Best model selection
* Prediction result review
* Anomaly output review
* Asset-wise anomaly summary
* Final project report generation

Final evaluation outputs:

```text
reports/metrics/model_comparison.csv
reports/metrics/forecasting_metrics.json
reports/metrics/anomaly_metrics.json
reports/figures/model_comparison.png
reports/figures/forecast_vs_actual.png
reports/figures/anomaly_detection_plot.png
```

---

## 17. API Methodology

The API is built using FastAPI.

Files:

```text
api/main.py
api/schemas.py
api/routes/
```

API responsibilities:

* Accept JSON records.
* Validate input.
* Run forecasting.
* Run anomaly detection.
* Return JSON response.
* Save outputs.
* Provide health checks.
* Provide model artifact status.

Important API endpoints:

| Endpoint            | Method | Purpose                                 |
| ------------------- | ------ | --------------------------------------- |
| `/`                 | GET    | Root endpoint                           |
| `/health/`          | GET    | Health check                            |
| `/health/full`      | GET    | Full project health                     |
| `/forecast`         | POST   | Forecast from JSON records              |
| `/anomaly`          | POST   | Detect anomalies from JSON records      |
| `/predict`          | POST   | Combined forecast and anomaly detection |
| `/forecast/default` | GET    | Forecast using default dataset          |
| `/anomaly/default`  | GET    | Detect anomalies using default dataset  |
| `/upload/predict`   | POST   | Upload CSV and run prediction           |

---

## 18. Dashboard Methodology

The dashboard is built using Streamlit.

Files:

```text
dashboard/app.py
dashboard/pages/
```

Dashboard responsibilities:

* Show dataset summary.
* Show price trends.
* Show forecast results.
* Show anomaly results.
* Show model performance.
* Show downloadable outputs.
* Show project explanation.
* Show model artifact health.

Dashboard pages:

| Page               | Purpose                                        |
| ------------------ | ---------------------------------------------- |
| Portfolio Overview | Shows dataset and project summary              |
| Price Forecasting  | Shows forecasting results and model comparison |
| Anomaly Detection  | Shows anomaly detection outputs                |
| Model Performance  | Shows model metrics and artifact health        |

---

## 19. Monitoring Methodology

Monitoring tracks the health and reliability of the project.

Monitoring tracks:

* Data quality
* Model performance
* Anomaly rate
* Drift
* API health
* Dashboard health
* Logs

Monitoring files:

```text
monitoring/monitoring_config.yaml
monitoring/performance_dashboard.md
monitoring/logs/app.log
```

Recommended monitoring outputs:

```text
monitoring/performance_metrics.csv
monitoring/performance_summary.json
monitoring/drift_report.csv
monitoring/monitoring_summary.json
monitoring/reports/daily_monitoring_report.md
```

Monitoring helps answer:

* Is the dataset valid?
* Are model artifacts available?
* Are forecasting errors acceptable?
* Are anomaly rates normal?
* Has data drift occurred?
* Is the API healthy?
* Is the dashboard healthy?
* Are there unresolved errors?

---

## 20. Final Workflow

Recommended complete workflow:

```bash
python scripts/run_data_pipeline.py
python scripts/train_all_models.py
python scripts/evaluate_models.py
python scripts/run_api.py --reload
python scripts/run_dashboard.py
```

---

