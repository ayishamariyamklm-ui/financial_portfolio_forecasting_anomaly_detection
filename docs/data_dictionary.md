# Data Dictionary

## Financial Portfolio Forecasting & Anomaly Detection

This document describes the dataset, processed files, feature columns, model outputs, and anomaly detection outputs used in the **Financial Portfolio Forecasting & Anomaly Detection** project.

---

## 1. Raw Dataset

Raw file:

```text
data/raw/financial_portfolio_data.csv
```

The raw dataset contains financial asset price records. Each row represents the price of one asset on a specific date.

---

### 1.1 Expected Original Columns

| Column  | Type        | Description                   | Example      |
| ------- | ----------- | ----------------------------- | ------------ |
| `Date`  | Date/String | Date of asset price record    | `2024-01-01` |
| `Asset` | String      | Name of financial asset       | `Stock A`    |
| `Price` | Numeric     | Asset price on the given date | `120.50`     |

---

### 1.2 Standardized Columns

After standardization, the column names become lowercase and consistent across the project.

| Standard Column | Original Column | Description          |
| --------------- | --------------- | -------------------- |
| `date`          | `Date`          | Date of price record |
| `asset`         | `Asset`         | Asset name           |
| `price`         | `Price`         | Asset price          |

---

## 2. Required Columns

The entire pipeline requires these columns:

```text
date
asset
price
```

If any of these columns are missing, the pipeline should raise an error.

---

## 3. Processed Data Files

Processed data files are saved inside the `data/processed/` folder. These files are created after cleaning, feature engineering, train-test splitting, and anomaly detection.

---

### 3.1 `cleaned_financial_data.csv`

Path:

```text
data/processed/cleaned_financial_data.csv
```

Purpose:

Stores the cleaned version of the raw dataset.

| Column  | Type     | Description                      |
| ------- | -------- | -------------------------------- |
| `date`  | Datetime | Cleaned date column              |
| `asset` | String   | Cleaned asset name               |
| `price` | Float    | Cleaned non-negative asset price |

Cleaning rules:

* Convert `date` to datetime.
* Convert `price` to numeric.
* Remove missing `date`, `asset`, or `price`.
* Remove negative prices.
* Remove duplicate records if needed.
* Sort by `asset` and `date`.

---

### 3.2 `daily_asset_prices.csv`

Path:

```text
data/processed/daily_asset_prices.csv
```

Purpose:

Stores daily asset-level price records.

| Column  | Type     | Description          |
| ------- | -------- | -------------------- |
| `date`  | Datetime | Date of price record |
| `asset` | String   | Asset name           |
| `price` | Float    | Daily asset price    |

---

### 3.3 `train_data.csv`

Path:

```text
data/processed/train_data.csv
```

Purpose:

Stores training data for model development.

| Column          | Type     | Description                                 |
| --------------- | -------- | ------------------------------------------- |
| `date`          | Datetime | Training record date                        |
| `asset`         | String   | Asset name                                  |
| `price`         | Float    | Target price                                |
| Feature columns | Numeric  | Engineered features used for model training |

---

### 3.4 `test_data.csv`

Path:

```text
data/processed/test_data.csv
```

Purpose:

Stores testing data used for model evaluation.

| Column          | Type     | Description                             |
| --------------- | -------- | --------------------------------------- |
| `date`          | Datetime | Testing record date                     |
| `asset`         | String   | Asset name                              |
| `price`         | Float    | Actual target price                     |
| Feature columns | Numeric  | Engineered features used for prediction |

---

### 3.5 `anomaly_results.csv`

Path:

```text
data/processed/anomaly_results.csv
```

Purpose:

Stores anomaly detection results generated from processed financial data.

Common columns include:

| Column             | Type     | Description                |
| ------------------ | -------- | -------------------------- |
| `date`             | Datetime | Date of asset price record |
| `asset`            | String   | Asset name                 |
| `price`            | Float    | Asset price                |
| `daily_return`     | Float    | Daily return value         |
| `is_anomaly`       | Boolean  | Final anomaly flag         |
| `anomaly_severity` | String   | Severity level of anomaly  |

---

## 4. Feature Engineering Columns

Feature engineering creates additional columns from the original `date`, `asset`, and `price` columns. These features help forecasting and anomaly detection models understand trends, seasonality, momentum, and volatility.

---

### 4.1 Calendar Features

| Column           | Type    | Description                      |
| ---------------- | ------- | -------------------------------- |
| `year`           | Integer | Year extracted from date         |
| `month`          | Integer | Month extracted from date        |
| `quarter`        | Integer | Quarter extracted from date      |
| `day`            | Integer | Day of month                     |
| `day_of_week`    | Integer | Day of week, Monday = 0          |
| `week_of_year`   | Integer | ISO week number                  |
| `is_month_start` | Integer | 1 if date is month start, else 0 |
| `is_month_end`   | Integer | 1 if date is month end, else 0   |

---

### 4.2 Return Features

| Column         | Type  | Description                                                               |
| -------------- | ----- | ------------------------------------------------------------------------- |
| `daily_return` | Float | Percentage change in price compared to previous record for the same asset |

Formula:

```text
daily_return = (current_price - previous_price) / previous_price
```

Example:

```text
If previous_price = 100
If current_price = 110

daily_return = (110 - 100) / 100
daily_return = 0.10
```

This means the asset price increased by 10%.

---

### 4.3 Lag Features

Lag features store previous price values for the same asset.

| Column         | Type  | Description           |
| -------------- | ----- | --------------------- |
| `price_lag_1`  | Float | Previous 1-day price  |
| `price_lag_3`  | Float | Previous 3-day price  |
| `price_lag_7`  | Float | Previous 7-day price  |
| `price_lag_14` | Float | Previous 14-day price |

Lag features are calculated separately for each asset.

---

### 4.4 Rolling Features

Rolling features summarize recent price behavior over a moving time window.

| Column                         | Type  | Description                                  |
| ------------------------------ | ----- | -------------------------------------------- |
| `price_rolling_mean_7`         | Float | 7-day rolling average price                  |
| `price_rolling_mean_14`        | Float | 14-day rolling average price                 |
| `price_rolling_mean_30`        | Float | 30-day rolling average price                 |
| `price_rolling_std_7`          | Float | 7-day rolling price standard deviation       |
| `price_rolling_std_14`         | Float | 14-day rolling price standard deviation      |
| `price_rolling_std_30`         | Float | 30-day rolling price standard deviation      |
| `return_rolling_volatility_7`  | Float | 7-day rolling standard deviation of returns  |
| `return_rolling_volatility_14` | Float | 14-day rolling standard deviation of returns |
| `return_rolling_volatility_30` | Float | 30-day rolling standard deviation of returns |

---

## 5. Forecasting Output Columns

Forecast output files may include:

```text
outputs/xgboost_forecast_results.csv
outputs/arima_forecast_results.csv
outputs/prophet_forecast_results.csv
outputs/lstm_forecast_results.csv
outputs/inference_forecasts.csv
outputs/api_forecast_results.csv
```

---

### 5.1 Common Forecast Columns

| Column             | Type     | Description                        |
| ------------------ | -------- | ---------------------------------- |
| `date`             | Datetime | Future forecast date               |
| `asset`            | String   | Asset being forecasted             |
| `forecasted_price` | Float    | Forecasted future price            |
| `predicted_price`  | Float    | Alternative prediction column name |
| `model_name`       | String   | Model used for forecast            |

---

### 5.2 Optional Forecast Columns

| Column                 | Type  | Description               |
| ---------------------- | ----- | ------------------------- |
| `forecast_lower_bound` | Float | Lower confidence estimate |
| `forecast_upper_bound` | Float | Upper confidence estimate |
| `yhat`                 | Float | Prophet forecast value    |
| `yhat_lower`           | Float | Prophet lower estimate    |
| `yhat_upper`           | Float | Prophet upper estimate    |

---

## 6. Prediction Evaluation Columns

Prediction output file:

```text
outputs/03_forecasting_predictions.csv
```

| Column                         | Type     | Description                            |
| ------------------------------ | -------- | -------------------------------------- |
| `date`                         | Datetime | Test date                              |
| `asset`                        | String   | Asset name                             |
| `actual_price`                 | Float    | Actual test price                      |
| `baseline_previous_price`      | Float    | Baseline forecast using previous price |
| `baseline_rolling_mean`        | Float    | Baseline forecast using rolling mean   |
| `linear_regression_prediction` | Float    | Linear Regression prediction           |
| `random_forest_prediction`     | Float    | Random Forest prediction               |
| `xgboost_prediction`           | Float    | XGBoost prediction                     |
| `best_model_name`              | String   | Best model name                        |

---

## 7. Forecasting Metrics

Metrics files:

```text
reports/metrics/forecasting_metrics.json
reports/metrics/model_comparison.csv
```

| Metric     | Type  | Description                              | Better Direction |
| ---------- | ----- | ---------------------------------------- | ---------------- |
| `mae`      | Float | Mean Absolute Error                      | Lower            |
| `mse`      | Float | Mean Squared Error                       | Lower            |
| `rmse`     | Float | Root Mean Squared Error                  | Lower            |
| `mape`     | Float | Mean Absolute Percentage Error           | Lower            |
| `smape`    | Float | Symmetric Mean Absolute Percentage Error | Lower            |
| `r2_score` | Float | Model explanation power                  | Higher           |

---

## 8. Anomaly Detection Output Columns

Anomaly output files:

```text
outputs/anomaly_detection_results.csv
outputs/api_anomaly_results.csv
outputs/inference_anomaly_results.csv
outputs/04_anomaly_detection_results.csv
```

| Column                        | Type     | Description                               |
| ----------------------------- | -------- | ----------------------------------------- |
| `date`                        | Datetime | Date of record                            |
| `asset`                       | String   | Asset name                                |
| `price`                       | Float    | Asset price                               |
| `daily_return`                | Float    | Daily price return                        |
| `price_z_score`               | Float    | Z-score of price                          |
| `is_anomaly_iqr`              | Boolean  | Whether IQR detected anomaly              |
| `is_anomaly_zscore`           | Boolean  | Whether Z-score detected anomaly          |
| `is_anomaly_isolation_forest` | Boolean  | Whether Isolation Forest detected anomaly |
| `anomaly_method_count`        | Integer  | Number of methods that detected anomaly   |
| `is_anomaly`                  | Boolean  | Final combined anomaly flag               |
| `anomaly_method`              | String   | Detection method names                    |
| `anomaly_severity`            | String   | Anomaly severity                          |
| `anomaly_score`               | Float    | Isolation Forest anomaly score            |

---

## 9. Anomaly Severity Levels

| Severity | Method Count | Meaning                       |
| -------- | ------------ | ----------------------------- |
| `normal` | 0            | Not anomalous                 |
| `low`    | 1            | Detected by one method        |
| `medium` | 2            | Detected by two methods       |
| `high`   | 3            | Detected by all three methods |

---

## 10. Model Artifacts

Model artifacts are trained models saved to disk. These are used by inference scripts, API endpoints, and dashboard pages.

---

### 10.1 Forecasting Models

| File                                   | Description         |
| -------------------------------------- | ------------------- |
| `models/forecasting/arima_model.pkl`   | Saved ARIMA model   |
| `models/forecasting/prophet_model.pkl` | Saved Prophet model |
| `models/forecasting/xgboost_model.pkl` | Saved XGBoost model |
| `models/forecasting/lstm_model.keras`  | Saved LSTM model    |

---

### 10.2 Anomaly Models

| File                                        | Description                          |
| ------------------------------------------- | ------------------------------------ |
| `models/anomaly/isolation_forest_model.pkl` | Saved Isolation Forest anomaly model |

---

### 10.3 Scalers

| File                              | Description                                     |
| --------------------------------- | ----------------------------------------------- |
| `models/scalers/price_scaler.pkl` | Price scaler used for LSTM or normalized models |

---

## 11. API Input Schema

Example API input:

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

### 11.1 API Input Fields

| Field            | Type         | Required | Description                        |
| ---------------- | ------------ | -------- | ---------------------------------- |
| `records`        | List[Object] | Yes      | Input financial records            |
| `date`           | String       | Yes      | Date of asset price                |
| `asset`          | String       | Yes      | Asset name                         |
| `price`          | Float        | Yes      | Asset price                        |
| `model_name`     | String       | No       | Forecasting model name             |
| `forecast_steps` | Integer      | No       | Number of future steps to forecast |
| `save_output`    | Boolean      | No       | Whether to save output file        |
| `return_records` | Boolean      | No       | Whether to return detailed records |

---

## 12. Data Quality Rules

| Rule             | Expected Condition                          |
| ---------------- | ------------------------------------------- |
| Required columns | `date`, `asset`, and `price` must exist     |
| Date validity    | Dates must be convertible to datetime       |
| Price validity   | Prices must be numeric                      |
| Negative prices  | Negative prices should be removed           |
| Missing values   | Missing required values should be removed   |
| Sorting          | Data should be sorted by `asset` and `date` |
| Empty data       | Empty cleaned dataset should raise error    |

---

## 13. Data Validation Checks

Recommended validation checks before model training:

| Check                | Purpose                                     |
| -------------------- | ------------------------------------------- |
| Column check         | Confirms required columns are present       |
| Missing value check  | Detects missing `date`, `asset`, or `price` |
| Duplicate check      | Detects duplicate asset-date records        |
| Date parsing check   | Confirms date column can be converted       |
| Price type check     | Confirms price is numeric                   |
| Negative price check | Detects invalid negative prices             |
| Empty dataset check  | Prevents training on empty data             |
| Asset count check    | Confirms at least one asset exists          |

---

