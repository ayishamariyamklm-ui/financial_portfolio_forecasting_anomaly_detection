# Troubleshooting Guide

## Financial Portfolio Forecasting & Anomaly Detection

This document lists common errors, causes, and fixes for the **Financial Portfolio Forecasting & Anomaly Detection** project.

---

## 1. Dataset Not Found

### Error

```text
FileNotFoundError: financial_portfolio_data.csv not found
```

### Cause

The raw CSV file is missing from the required folder.

### Fix

Place the file here:

```text
data/raw/financial_portfolio_data.csv
```

Then run:

```bash
python scripts/run_data_pipeline.py
```

---

## 2. Missing Required Columns

### Error

```text
ValueError: Missing required columns: ['date', 'asset', 'price']
```

### Cause

The CSV file does not contain the required columns.

### Fix

Your CSV should contain either:

```text
Date, Asset, Price
```

or:

```text
date, asset, price
```

Make sure there are no spelling mistakes or extra spaces in the column names.

---

## 3. Invalid Date Format

### Error

```text
date conversion failed
```

### Cause

Some date values cannot be parsed correctly.

### Fix

Use a standard date format:

```text
YYYY-MM-DD
```

Example:

```text
2024-01-01
```

Also check for empty date values or mixed date formats.

---

## 4. Negative Prices

### Problem

Some rows contain negative prices.

### Cause

Financial asset prices should not be negative. Negative prices may come from data entry mistakes, incorrect formatting, or corrupted records.

### Fix

The cleaning pipeline removes negative prices automatically.

Run:

```bash
python scripts/run_data_pipeline.py
```

Check your data manually if many rows are removed.

---

## 5. Empty Dataset After Cleaning

### Error

```text
Input dataframe is empty after cleaning.
```

### Cause

All rows were removed due to missing or invalid values.

### Fix

Check these fields in your raw dataset:

* Date column
* Asset column
* Price column
* Missing values
* Negative prices
* Invalid date values
* Non-numeric price values

Then run the pipeline again:

```bash
python scripts/run_data_pipeline.py
```

---

## 6. Import Error

### Error

```text
ModuleNotFoundError: No module named 'src'
```

### Cause

You are not running the command from the project root folder.

### Fix

Run commands from:

```text
financial-portfolio-forecasting-anomaly-detection/
```

Example:

```bash
python scripts/run_data_pipeline.py
```

Correct project root should contain folders like:

```text
src/
api/
dashboard/
scripts/
data/
```

---

## 7. Package Not Installed

### Error

```text
ModuleNotFoundError: No module named 'xgboost'
```

### Cause

The required Python package is not installed in the active environment.

### Fix

Install all requirements:

```bash
pip install -r requirements.txt
```

Or install the missing package directly:

```bash
pip install xgboost
```

---

## 8. Prophet Installation Error

### Error

```text
ModuleNotFoundError: No module named 'prophet'
```

### Cause

The Prophet package is not installed or failed during installation.

### Fix

Install Prophet:

```bash
pip install prophet
```

If Prophet installation fails, you can still run other models such as ARIMA, XGBoost, and LSTM.

---

## 9. TensorFlow Installation Error

### Error

```text
ModuleNotFoundError: No module named 'tensorflow'
```

### Cause

TensorFlow is not installed, or your Python version may not be compatible.

### Fix

Install TensorFlow:

```bash
pip install tensorflow
```

If TensorFlow is heavy for your system, skip LSTM temporarily and continue with ARIMA, Prophet, and XGBoost.

---

## 10. Model File Missing

### Error

```text
Saved XGBoost model not found
```

### Cause

Inference was run before model training.

### Fix

Train models first:

```bash
python scripts/train_all_models.py
```

Then run inference or API again.

---

## 11. Anomaly Model Missing

### Error

```text
Saved anomaly detection model not found
```

### Cause

The anomaly detection model artifact was not created yet.

### Fix

Run:

```bash
python src/training/train_anomaly_model.py
```

or:

```bash
python scripts/train_all_models.py
```

---

## 12. API Not Starting

### Error

```text
uvicorn: command not found
```

### Cause

FastAPI or Uvicorn is not installed.

### Fix

Install the required packages:

```bash
pip install fastapi uvicorn
```

Then run:

```bash
python scripts/run_api.py --reload
```

or:

```bash
uvicorn api.main:app --reload
```

---

## 13. API App Path Error

### Error

```text
Error loading ASGI app
```

### Cause

The API app path is wrong, or `api/main.py` does not contain a valid FastAPI app object.

### Fix

Use the correct app path:

```bash
uvicorn api.main:app --reload
```

Make sure `api/main.py` contains:

```python
app = FastAPI()
```

---

## 14. API Port Already in Use

### Error

```text
Address already in use
```

### Cause

Another process is already using the same API port.

### Fix

Use another port:

```bash
python scripts/run_api.py --port 8001 --reload
```

Or run directly:

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8001 --reload
```

---

## 15. Dashboard Not Starting

### Error

```text
streamlit: command not found
```

### Cause

Streamlit or dashboard dependencies are not installed.

### Fix

Install the required packages:

```bash
pip install streamlit plotly
```

Run:

```bash
python scripts/run_dashboard.py
```

or:

```bash
streamlit run dashboard/app.py
```

---

## 16. Dashboard Port Already in Use

### Error

```text
Port 8501 is already in use
```

### Cause

Another Streamlit app is already running on port `8501`.

### Fix

Use another port:

```bash
python scripts/run_dashboard.py --port 8502
```

Or run directly:

```bash
streamlit run dashboard/app.py --server.port 8502
```

---

## 17. Dashboard Shows Missing Files

### Cause

Model outputs, evaluation files, or processed files are not generated yet.

### Fix

Run the full preparation sequence:

```bash
python scripts/run_data_pipeline.py
python scripts/train_all_models.py
python scripts/evaluate_models.py
```

Then start the dashboard again:

```bash
python scripts/run_dashboard.py
```

---

## 18. Pytest Import Error

### Error

```text
ModuleNotFoundError
```

### Cause

Tests are being run from the wrong folder, or package initialization files are missing.

### Fix

Run tests from the project root:

```bash
pytest tests -v
```

Make sure `__init__.py` files exist in:

```text
src/
api/
tests/
```

---

## 19. FastAPI TestClient Error

### Error

```text
ModuleNotFoundError: No module named 'httpx'
```

### Cause

`httpx` is required for FastAPI testing but is not installed.

### Fix

Install `httpx`:

```bash
pip install httpx
```

Then run:

```bash
pytest tests/test_api.py -v
```

---

## 20. Docker Not Recognized

### Error

```text
docker is not recognized
```

### Cause

Docker is not installed, not added to PATH, or Docker Desktop is not running.

### Fix

Install Docker Desktop and restart your terminal.

After installation, test:

```bash
docker --version
```

---

## 21. Render Deployment Fails

### Common Causes

* Wrong start command
* Missing requirements
* App path issue
* Missing dataset
* Missing model files
* Incorrect health check path

### Correct Start Command

```bash
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

### Correct Health Check Path

```text
/health/
```

### Fix

Make sure:

* `requirements.txt` is updated.
* `api/main.py` exists.
* `api.main:app` is the correct app path.
* Required files are committed.
* Environment variables are added if needed.

---

## 22. Railway Deployment Fails

### Cause

Railway may not find the correct start command.

### Fix

Make sure `railway.json` contains:

```json
{
  "deploy": {
    "startCommand": "uvicorn api.main:app --host 0.0.0.0 --port $PORT"
  }
}
```

Also confirm that the file is in the correct location:

```text
railway.json
```

or:

```text
deployment/railway.json
```

depending on your Railway setup.

---

## 23. Forecast Results Missing

### Cause

Forecasting models or evaluation scripts have not been run yet.

### Fix

Run:

```bash
python scripts/train_all_models.py
python scripts/evaluate_models.py
```

Expected outputs:

```text
outputs/
reports/metrics/
```

---

## 24. Anomaly Results Missing

### Cause

Anomaly detection was not run yet.

### Fix

Run:

```bash
python src/training/train_anomaly_model.py
```

Expected output:

```text
outputs/anomaly_detection_results.csv
```

You can also run the full training script:

```bash
python scripts/train_all_models.py
```

---

## 25. Recommended Clean Run

If many files are missing, run this full sequence:

```bash
python scripts/run_data_pipeline.py
python scripts/train_all_models.py
python scripts/evaluate_models.py
python src/inference/predict.py
python scripts/run_api.py --reload
python scripts/run_dashboard.py
```

---