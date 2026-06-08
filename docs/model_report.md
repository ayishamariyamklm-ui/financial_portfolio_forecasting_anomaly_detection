# Model Report

## Financial Portfolio Forecasting & Anomaly Detection

This report summarizes the models used in the **Financial Portfolio Forecasting & Anomaly Detection** project, including their purpose, evaluation metrics, outputs, strengths, limitations, and expected artifacts.

---

## 1. Model Objective

The project uses two categories of models:

1. Forecasting models
2. Anomaly detection models

---

### 1.1 Forecasting Objective

```text
Predict future asset prices using historical price patterns.
```

The forecasting models use historical asset price data and engineered time series features to estimate future prices.

---

### 1.2 Anomaly Detection Objective

```text
Identify unusual asset price movements that may require review.
```

The anomaly detection methods are used to identify unusual price records, abnormal returns, or suspicious movements in asset prices.

---

## 2. Forecasting Models

Forecasting models included in this project:

| Model   | File                          | Artifact                               |
| ------- | ----------------------------- | -------------------------------------- |
| ARIMA   | `src/models/arima_model.py`   | `models/forecasting/arima_model.pkl`   |
| Prophet | `src/models/prophet_model.py` | `models/forecasting/prophet_model.pkl` |
| XGBoost | `src/models/xgboost_model.py` | `models/forecasting/xgboost_model.pkl` |
| LSTM    | `src/models/lstm_model.py`    | `models/forecasting/lstm_model.keras`  |

---

## 3. Forecasting Model Descriptions

This section explains each forecasting model used in the project.

---

### 3.1 ARIMA

ARIMA is a classical statistical time series model.

ARIMA stands for:

```text
AutoRegressive Integrated Moving Average
```

It is commonly used for forecasting time series data based on past values and past forecast errors.

#### Strengths

* Good baseline for time series forecasting
* Interpretable
* Works well for simple trends
* Useful for single-asset forecasting
* Does not require many engineered features

#### Limitations

* May not capture non-linear patterns
* Usually trained separately per asset
* Requires sufficient historical data
* Can require parameter tuning
* May not perform well during highly volatile market periods

---

### 3.2 Prophet

Prophet is a forecasting model designed for trend and seasonality.

It is useful for business-friendly time series forecasting and can model trend changes and seasonal patterns.

#### Strengths

* Handles trend and seasonality
* Business-friendly
* Good for interpretable time series forecasting
* Can produce future forecasts with uncertainty intervals
* Easier to use than many traditional time series models

#### Limitations

* Can be slower than simple models
* May require enough historical data
* External regressors are not included in the basic version
* May not perform well on highly noisy financial price data
* Can be heavy for free cloud deployment

---

### 3.3 XGBoost

XGBoost is a gradient boosting regression model.

It uses engineered features such as lag values, rolling statistics, calendar features, and return features to predict asset prices.

#### Strengths

* Strong predictive performance
* Handles non-linear relationships
* Works well with engineered features
* Usually faster than deep learning models
* Handles tabular data effectively

#### Limitations

* Requires feature engineering
* Does not naturally understand time order without lag features
* Hyperparameters may need tuning
* Can overfit if validation is not done correctly
* Needs careful time-based train-test splitting to avoid data leakage

---

### 3.4 LSTM

LSTM is a deep learning sequence model.

LSTM stands for:

```text
Long Short-Term Memory
```

It is designed to learn patterns from sequential data.

#### Strengths

* Learns sequential patterns
* Can capture complex temporal relationships
* Useful for advanced time series modeling
* Can learn from historical price windows
* Suitable for sequence-based forecasting problems

#### Limitations

* Requires more data
* Requires scaling
* More computationally expensive
* Can overfit small datasets
* Less interpretable than statistical or tree-based models
* May require GPU or stronger cloud resources for larger datasets

---

## 4. Forecasting Features

The forecasting models use engineered features created from the raw financial price data.

| Feature Type | Examples                                        |
| ------------ | ----------------------------------------------- |
| Calendar     | `year`, `month`, `quarter`, `day_of_week`       |
| Lag          | `price_lag_1`, `price_lag_7`, `price_lag_14`    |
| Rolling Mean | `price_rolling_mean_7`, `price_rolling_mean_30` |
| Rolling Std  | `price_rolling_std_7`, `price_rolling_std_30`   |
| Returns      | `daily_return`                                  |
| Volatility   | `return_rolling_volatility_7`                   |

---

### 4.1 Calendar Features

Calendar features help the model identify time-based patterns.

Examples:

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

### 4.2 Lag Features

Lag features provide previous price values to the model.

Examples:

```text
price_lag_1
price_lag_3
price_lag_7
price_lag_14
```

Lag features are useful because past price values often influence future price movement.

---

### 4.3 Rolling Features

Rolling features summarize recent price behavior.

Examples:

```text
price_rolling_mean_7
price_rolling_mean_14
price_rolling_mean_30
price_rolling_std_7
price_rolling_std_14
price_rolling_std_30
```

Rolling features help capture short-term trends and volatility.

---

### 4.4 Return and Volatility Features

Return and volatility features capture percentage movement and risk-like behavior.

Examples:

```text
daily_return
return_rolling_volatility_7
return_rolling_volatility_14
return_rolling_volatility_30
```

---

## 5. Forecasting Evaluation Metrics

Forecasting models are evaluated using the following metrics:

| Metric   | Meaning                        | Better Direction |
| -------- | ------------------------------ | ---------------- |
| MAE      | Mean Absolute Error            | Lower            |
| MSE      | Mean Squared Error             | Lower            |
| RMSE     | Root Mean Squared Error        | Lower            |
| MAPE     | Mean Absolute Percentage Error | Lower            |
| SMAPE    | Symmetric MAPE                 | Lower            |
| R2 Score | Model explanation power        | Higher           |

---

### 5.1 Primary Model Selection Metric

Primary model selection metric:

```text
RMSE
```

Reason:

RMSE penalizes large forecast errors, which is important in financial forecasting because large prediction mistakes can have a stronger practical impact than small errors.

---

### 5.2 Metric Interpretation

| Metric   | Interpretation                                                  |
| -------- | --------------------------------------------------------------- |
| MAE      | Average absolute difference between actual and predicted prices |
| MSE      | Average squared prediction error                                |
| RMSE     | Square root of MSE, easier to interpret in price units          |
| MAPE     | Percentage-based forecast error                                 |
| SMAPE    | Symmetric percentage error, useful when values vary in scale    |
| R2 Score | Measures how much variation the model explains                  |

---

## 6. Forecasting Model Comparison

Expected comparison file:

```text
reports/metrics/model_comparison.csv
```

Template:

| Model   | MAE       | MSE       | RMSE      | MAPE      | SMAPE     | R2 Score  | Rank      |
| ------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| ARIMA   | To update | To update | To update | To update | To update | To update | To update |
| Prophet | To update | To update | To update | To update | To update | To update | To update |
| XGBoost | To update | To update | To update | To update | To update | To update | To update |
| LSTM    | To update | To update | To update | To update | To update | To update | To update |

---

## 7. Best Forecasting Model

The best model is selected using the lowest RMSE.

Expected file:

```text
reports/metrics/model_comparison.csv
```

Best model after running evaluation:

```text
To be updated after scripts/evaluate_models.py
```

After evaluation, the best model should be documented in the final project report and README file.

---

## 8. Forecasting Outputs

Expected forecast output files:

```text
outputs/arima_forecast_results.csv
outputs/prophet_forecast_results.csv
outputs/xgboost_forecast_results.csv
outputs/lstm_forecast_results.csv
outputs/inference_forecasts.csv
outputs/api_forecast_results.csv
```

Common output columns:

| Column             | Description                        |
| ------------------ | ---------------------------------- |
| `date`             | Forecast date                      |
| `asset`            | Asset name                         |
| `forecasted_price` | Forecasted price                   |
| `predicted_price`  | Alternative prediction column name |
| `model_name`       | Model used                         |

Optional output columns:

| Column                 | Description             |
| ---------------------- | ----------------------- |
| `forecast_lower_bound` | Lower forecast estimate |
| `forecast_upper_bound` | Upper forecast estimate |
| `yhat`                 | Prophet forecast value  |
| `yhat_lower`           | Prophet lower estimate  |
| `yhat_upper`           | Prophet upper estimate  |

---

## 9. Anomaly Detection Models

Anomaly detection methods used in this project:

| Method           | Type             | Purpose                               |
| ---------------- | ---------------- | ------------------------------------- |
| IQR              | Statistical      | Detect price outliers                 |
| Z-Score          | Statistical      | Detect values far from average        |
| Isolation Forest | Machine Learning | Detect unusual multi-feature patterns |

Model file:

```text
src/models/anomaly_detector.py
```

Artifact:

```text
models/anomaly/isolation_forest_model.pkl
```

---

## 10. Anomaly Detection Method Descriptions

This section explains each anomaly detection method.

---

### 10.1 IQR

IQR detects values outside the normal price spread.

IQR stands for:

```text
Interquartile Range
```

Rule:

```text
price < Q1 - 1.5 × IQR
```

or

```text
price > Q3 + 1.5 × IQR
```

#### Strengths

* Easy to explain
* Good for extreme outliers
* Does not require model training
* Useful for quick statistical anomaly detection

#### Limitations

* May not detect complex anomalies
* Sensitive to skewed distributions
* Uses only distribution-based boundaries
* May not capture time-based abnormal behavior

---

### 10.2 Z-Score

Z-score detects values far from the mean.

Rule:

```text
absolute z-score > 3
```

#### Strengths

* Simple
* Easy to interpret
* Useful for normally distributed data
* Good for detecting extreme deviations

#### Limitations

* Less effective for skewed data
* Sensitive to outliers
* Assumes the mean and standard deviation are meaningful
* May not work well when price distributions are not normal

---

### 10.3 Isolation Forest

Isolation Forest detects unusual records using multiple features.

Features used:

* Price
* Daily return
* Lag values
* Rolling mean
* Rolling standard deviation
* Volatility

#### Strengths

* Detects complex anomalies
* Works without labeled anomaly data
* Uses multiple features
* Suitable for unsupervised anomaly detection
* Can capture unusual combinations of feature values

#### Limitations

* Less explainable than IQR or Z-score
* Requires contamination parameter tuning
* Depends on feature quality
* Needs careful interpretation before business action

---

## 11. Anomaly Detection Outputs

Expected files:

```text
outputs/anomaly_detection_results.csv
outputs/api_anomaly_results.csv
outputs/inference_anomaly_results.csv
```

Important columns:

| Column                        | Description                    |
| ----------------------------- | ------------------------------ |
| `is_anomaly_iqr`              | IQR anomaly flag               |
| `is_anomaly_zscore`           | Z-score anomaly flag           |
| `is_anomaly_isolation_forest` | Isolation Forest anomaly flag  |
| `is_anomaly`                  | Final combined anomaly flag    |
| `anomaly_method_count`        | Number of detecting methods    |
| `anomaly_severity`            | Normal, low, medium, or high   |
| `anomaly_score`               | Isolation Forest anomaly score |

---

## 12. Anomaly Summary

Expected anomaly metrics file:

```text
reports/metrics/anomaly_metrics.json
```

Metrics:

| Metric          | Description                             |
| --------------- | --------------------------------------- |
| Total records   | Number of records checked               |
| Total anomalies | Number of anomaly records               |
| Normal records  | Non-anomaly records                     |
| Anomaly rate    | Percentage of records marked as anomaly |
| Method summary  | Count by detection method               |
| Asset summary   | Count by asset                          |

---

### 12.1 Anomaly Severity Mapping

| Method Count | Severity | Meaning                       |
| ------------ | -------- | ----------------------------- |
| 0            | Normal   | No anomaly detected           |
| 1            | Low      | Detected by one method        |
| 2            | Medium   | Detected by two methods       |
| 3            | High     | Detected by all three methods |

---

## 13. Model Training Scripts

| Script                                     | Purpose                           |
| ------------------------------------------ | --------------------------------- |
| `src/training/train_forecasting_models.py` | Train ARIMA, Prophet, and XGBoost |
| `src/training/train_lstm_model.py`         | Train LSTM                        |
| `src/training/train_anomaly_model.py`      | Train anomaly model               |
| `scripts/train_all_models.py`              | Run all model training            |

Run:

```bash
python scripts/train_all_models.py
```

Expected training outputs:

```text
models/forecasting/arima_model.pkl
models/forecasting/prophet_model.pkl
models/forecasting/xgboost_model.pkl
models/forecasting/lstm_model.keras
models/anomaly/isolation_forest_model.pkl
models/scalers/price_scaler.pkl
```

---

## 14. Model Evaluation Scripts

| Script                               | Purpose           |
| ------------------------------------ | ----------------- |
| `src/evaluation/metrics.py`          | Metric functions  |
| `src/evaluation/model_comparison.py` | Model ranking     |
| `src/evaluation/visualization.py`    | Evaluation charts |
| `scripts/evaluate_models.py`         | Final evaluation  |

Run:

```bash
python scripts/evaluate_models.py
```

Expected evaluation outputs:

```text
reports/metrics/forecasting_metrics.json
reports/metrics/anomaly_metrics.json
reports/metrics/model_comparison.csv
reports/figures/model_comparison.png
reports/figures/forecast_vs_actual.png
reports/figures/anomaly_detection_plot.png
```

---

## 15. Model Strengths and Limitations Summary

| Model/Method     | Strengths                                 | Limitations                          |
| ---------------- | ----------------------------------------- | ------------------------------------ |
| ARIMA            | Interpretable, good baseline              | Limited for non-linear data          |
| Prophet          | Handles trend and seasonality             | Can be heavy, needs sufficient data  |
| XGBoost          | Strong performance on engineered features | Requires feature engineering         |
| LSTM             | Learns sequences and complex patterns     | Needs more data and compute          |
| IQR              | Simple and explainable                    | Limited to statistical outliers      |
| Z-Score          | Easy to interpret                         | Works best with normal distributions |
| Isolation Forest | Detects complex anomalies                 | Less explainable and needs tuning    |

---

