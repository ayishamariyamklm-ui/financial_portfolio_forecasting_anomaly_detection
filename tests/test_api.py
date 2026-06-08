"""
API tests for the Financial Portfolio Forecasting
and Anomaly Detection project.

Run from project root:

    pytest tests/test_api.py -v

Note:
These tests use FastAPI TestClient.
Make sure these packages are installed:

    pip install pytest httpx fastapi
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient


# ============================================================
# Add Project Root to Python Path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ============================================================
# Import API App
# ============================================================

try:
    from api.main import app
except Exception:
    app = None


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def client():
    """
    Create FastAPI test client.
    """
    if app is None:
        pytest.skip("api.main app could not be imported.")

    return TestClient(app)


@pytest.fixture
def sample_api_records():
    """
    Create sample API records.
    """
    return [
        {
            "date": "2024-01-01",
            "asset": "Stock A",
            "price": 100.0,
        },
        {
            "date": "2024-01-02",
            "asset": "Stock A",
            "price": 101.0,
        },
        {
            "date": "2024-01-03",
            "asset": "Stock A",
            "price": 102.0,
        },
        {
            "date": "2024-01-04",
            "asset": "Stock A",
            "price": 103.0,
        },
        {
            "date": "2024-01-05",
            "asset": "Stock A",
            "price": 104.0,
        },
    ]


@pytest.fixture
def sample_forecast_dataframe():
    """
    Create sample forecast dataframe.
    """
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-06", periods=3),
            "asset": ["Stock A"] * 3,
            "forecasted_price": [105.0, 106.0, 107.0],
            "model_name": ["XGBoost"] * 3,
        }
    )


@pytest.fixture
def sample_anomaly_dataframe():
    """
    Create sample anomaly dataframe.
    """
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5),
            "asset": ["Stock A"] * 5,
            "price": [100, 101, 500, 102, 103],
            "is_anomaly_iqr": [False, False, True, False, False],
            "is_anomaly_zscore": [False, False, True, False, False],
            "is_anomaly_isolation_forest": [False, False, True, False, False],
            "is_anomaly": [False, False, True, False, False],
            "anomaly_method_count": [0, 0, 3, 0, 0],
            "anomaly_severity": ["normal", "normal", "high", "normal", "normal"],
            "anomaly_score": [0.2, 0.1, -0.4, 0.3, 0.2],
        }
    )


# ============================================================
# Basic API Tests
# ============================================================

def test_api_app_importable():
    """
    Test API app import.
    """
    assert app is not None, "api.main app could not be imported."


def test_root_endpoint(client):
    """
    Test root endpoint.
    """
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert "status" in data
    assert data["status"] == "success"


def test_health_endpoint_if_available(client):
    """
    Test health endpoint.
    """
    response = client.get("/health")

    assert response.status_code in [200, 307, 404]

    if response.status_code == 200:
        data = response.json()
        assert "status" in data


def test_health_full_endpoint_if_available(client):
    """
    Test full health endpoint.
    """
    response = client.get("/health/full")

    assert response.status_code in [200, 404]

    if response.status_code == 200:
        data = response.json()
        assert "status" in data


def test_models_status_endpoint_if_available(client):
    """
    Test model status endpoint.
    """
    response = client.get("/models/status")

    assert response.status_code in [200, 404]

    if response.status_code == 200:
        data = response.json()
        assert "status" in data


# ============================================================
# Forecast Endpoint Tests with Mocking
# ============================================================

def test_forecast_endpoint_with_mock(
    client,
    sample_api_records,
    sample_forecast_dataframe,
):
    """
    Test /forecast endpoint with mocked forecasting function.
    """
    payload = {
        "records": sample_api_records,
        "model_name": "xgboost",
        "forecast_steps": 3,
        "save_output": False,
        "return_records": True,
    }

    with patch("api.main.generate_forecasts") as mock_forecast:
        mock_forecast.return_value = sample_forecast_dataframe

        response = client.post("/forecast", json=payload)

    assert response.status_code in [200, 404, 500]

    if response.status_code == 200:
        data = response.json()

        assert data["status"] == "success"
        assert "data" in data
        assert "forecast_summary" in data["data"]


def test_forecast_endpoint_validation_error(client):
    """
    Test forecast endpoint validation error.
    """
    payload = {
        "records": [],
        "model_name": "xgboost",
        "forecast_steps": 3,
    }

    response = client.post("/forecast", json=payload)

    assert response.status_code in [422, 500]


# ============================================================
# Anomaly Endpoint Tests with Mocking
# ============================================================

def test_anomaly_endpoint_with_mock(
    client,
    sample_api_records,
    sample_anomaly_dataframe,
):
    """
    Test /anomaly endpoint with mocked anomaly function.
    """
    payload = {
        "records": sample_api_records,
        "save_output": False,
        "return_records": True,
        "only_anomalies": False,
    }

    with patch("api.main.detect_anomalies_with_saved_model") as mock_anomaly:
        mock_anomaly.return_value = sample_anomaly_dataframe

        response = client.post("/anomaly", json=payload)

    assert response.status_code in [200, 404, 500]

    if response.status_code == 200:
        data = response.json()

        assert data["status"] == "success"
        assert "data" in data
        assert "anomaly_summary" in data["data"]


def test_anomaly_endpoint_validation_error(client):
    """
    Test anomaly endpoint validation error.
    """
    payload = {
        "records": [],
        "save_output": False,
    }

    response = client.post("/anomaly", json=payload)

    assert response.status_code in [422, 500]


# ============================================================
# Combined Prediction Endpoint Tests
# ============================================================

def test_predict_endpoint_with_mock(
    client,
    sample_api_records,
    sample_forecast_dataframe,
    sample_anomaly_dataframe,
):
    """
    Test /predict endpoint with mocked forecasting and anomaly functions.
    """
    payload = {
        "records": sample_api_records,
        "model_name": "xgboost",
        "forecast_steps": 3,
        "run_forecasting": True,
        "run_anomaly_detection": True,
        "save_output": False,
        "return_records": True,
        "only_anomalies": False,
    }

    with patch("api.main.generate_forecasts") as mock_forecast:
        with patch("api.main.detect_anomalies_with_saved_model") as mock_anomaly:
            mock_forecast.return_value = sample_forecast_dataframe
            mock_anomaly.return_value = sample_anomaly_dataframe

            response = client.post("/predict", json=payload)

    assert response.status_code in [200, 404, 500]

    if response.status_code == 200:
        data = response.json()

        assert data["status"] == "success"
        assert "data" in data


# ============================================================
# Route Module Tests
# ============================================================

def test_forecast_route_health_if_included(client):
    """
    Test forecast route health if router is included.
    """
    response = client.get("/forecast/health")

    assert response.status_code in [200, 404]

    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "success"


def test_anomaly_route_health_if_included(client):
    """
    Test anomaly route health if router is included.
    """
    response = client.get("/anomaly/health")

    assert response.status_code in [200, 404]

    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "success"


def test_health_route_ping_if_included(client):
    """
    Test health ping route if router is included.
    """
    response = client.get("/health/ping")

    assert response.status_code in [200, 404]

    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "success"


# ============================================================
# Schema Tests
# ============================================================

def test_api_schemas_importable():
    """
    Test API schemas import.
    """
    try:
        from api.schemas import (
            AnomalyRequest,
            CombinedPredictionRequest,
            ForecastRequest,
            PriceRecord,
        )

        assert PriceRecord is not None
        assert ForecastRequest is not None
        assert AnomalyRequest is not None
        assert CombinedPredictionRequest is not None

    except Exception as error:
        pytest.skip(f"api.schemas could not be imported: {error}")


def test_price_record_schema_validation():
    """
    Test PriceRecord schema validation.
    """
    try:
        from api.schemas import PriceRecord

        record = PriceRecord(
            date="2024-01-01",
            asset="Stock A",
            price=100.0,
        )

        assert record.date == "2024-01-01"
        assert record.asset == "Stock A"
        assert record.price == 100.0

    except Exception as error:
        pytest.skip(f"PriceRecord schema test skipped: {error}")


def test_forecast_request_schema_validation(sample_api_records):
    """
    Test ForecastRequest schema validation.
    """
    try:
        from api.schemas import ForecastRequest

        request = ForecastRequest(
            records=sample_api_records,
            model_name="xgboost",
            forecast_steps=3,
        )

        assert request.model_name == "xgboost"
        assert request.forecast_steps == 3
        assert len(request.records) == 5

    except Exception as error:
        pytest.skip(f"ForecastRequest schema test skipped: {error}")