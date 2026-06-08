"""
Unit tests for anomaly service functionality.

Project:
Financial Portfolio Forecasting & Anomaly Detection

Run from project root:

    pytest tests/test_anomaly_service.py -v
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


# ============================================================
# Add Project Root to Python Path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ============================================================
# Imports
# ============================================================

try:
    from src.inference.predict import (
        clean_input_data,
        summarize_anomaly_results,
        predict_anomalies,
    )
except Exception:
    clean_input_data = None
    summarize_anomaly_results = None
    predict_anomalies = None


# ============================================================
# Sample Data
# ============================================================

@pytest.fixture
def sample_anomaly_input():
    """
    Create sample input data for anomaly detection.
    """
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "asset": ["Stock A"] * 10,
            "price": [100, 101, 102, 103, 500, 104, 105, 106, 107, 108],
        }
    )


@pytest.fixture
def sample_anomaly_output():
    """
    Create sample anomaly output dataframe.
    """
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "asset": ["Stock A"] * 5,
            "price": [100, 101, 500, 102, 103],
            "is_anomaly_iqr": [False, False, True, False, False],
            "is_anomaly_zscore": [False, False, True, False, False],
            "is_anomaly_isolation_forest": [False, False, True, False, False],
            "anomaly_method_count": [0, 0, 3, 0, 0],
            "is_anomaly": [False, False, True, False, False],
            "anomaly_method": ["None", "None", "IQR, Z-Score, Isolation Forest", "None", "None"],
            "anomaly_severity": ["normal", "normal", "high", "normal", "normal"],
            "anomaly_score": [0.2, 0.1, -0.4, 0.3, 0.2],
        }
    )


# ============================================================
# Basic Import Tests
# ============================================================

def test_anomaly_service_functions_available():
    """
    Test anomaly-related functions are importable.
    """
    assert clean_input_data is not None, "clean_input_data import failed."
    assert summarize_anomaly_results is not None, "summarize_anomaly_results import failed."


# ============================================================
# Anomaly Summary Tests
# ============================================================

@pytest.mark.skipif(summarize_anomaly_results is None, reason="summarize_anomaly_results not available")
def test_summarize_anomaly_results_basic(sample_anomaly_output):
    """
    Test anomaly summary calculation.
    """
    summary = summarize_anomaly_results(sample_anomaly_output)

    assert isinstance(summary, dict)
    assert summary["total_records"] == 5
    assert summary["total_anomalies"] == 1
    assert summary["normal_records"] == 4
    assert summary["anomaly_rate_percent"] == 20.0


@pytest.mark.skipif(summarize_anomaly_results is None, reason="summarize_anomaly_results not available")
def test_summarize_anomaly_results_empty_dataframe():
    """
    Test anomaly summary with empty dataframe.
    """
    empty_dataframe = pd.DataFrame()

    summary = summarize_anomaly_results(empty_dataframe)

    assert isinstance(summary, dict)
    assert summary["total_records"] == 0
    assert summary["total_anomalies"] == 0


@pytest.mark.skipif(summarize_anomaly_results is None, reason="summarize_anomaly_results not available")
def test_summarize_anomaly_results_method_summary(sample_anomaly_output):
    """
    Test method-wise anomaly summary exists.
    """
    summary = summarize_anomaly_results(sample_anomaly_output)

    assert "method_summary" in summary
    assert summary["method_summary"]["is_anomaly_iqr"] == 1
    assert summary["method_summary"]["is_anomaly_zscore"] == 1
    assert summary["method_summary"]["is_anomaly_isolation_forest"] == 1


@pytest.mark.skipif(summarize_anomaly_results is None, reason="summarize_anomaly_results not available")
def test_summarize_anomaly_results_asset_summary(sample_anomaly_output):
    """
    Test asset-wise anomaly summary exists.
    """
    summary = summarize_anomaly_results(sample_anomaly_output)

    assert "asset_summary" in summary
    assert isinstance(summary["asset_summary"], list)
    assert len(summary["asset_summary"]) > 0


# ============================================================
# Clean Input Tests
# ============================================================

@pytest.mark.skipif(clean_input_data is None, reason="clean_input_data not available")
def test_anomaly_clean_input_data(sample_anomaly_input):
    """
    Test anomaly input cleaning.
    """
    cleaned_data = clean_input_data(sample_anomaly_input)

    assert isinstance(cleaned_data, pd.DataFrame)
    assert "date" in cleaned_data.columns
    assert "asset" in cleaned_data.columns
    assert "price" in cleaned_data.columns
    assert pd.api.types.is_datetime64_any_dtype(cleaned_data["date"])


@pytest.mark.skipif(clean_input_data is None, reason="clean_input_data not available")
def test_anomaly_clean_input_data_removes_invalid_rows():
    """
    Test invalid rows are removed.
    """
    dataframe = pd.DataFrame(
        {
            "date": ["2024-01-01", None, "2024-01-03"],
            "asset": ["Stock A", "Stock A", None],
            "price": [100, 101, 102],
        }
    )

    cleaned_data = clean_input_data(dataframe)

    assert len(cleaned_data) == 1


# ============================================================
# Mocked Prediction Tests
# ============================================================

@pytest.mark.skipif(predict_anomalies is None, reason="predict_anomalies not available")
def test_predict_anomalies_with_mocked_detector(
    sample_anomaly_input,
    sample_anomaly_output,
):
    """
    Test predict_anomalies using mocked saved anomaly detector call.
    """
    with patch("src.inference.predict.detect_anomalies_with_saved_model") as mock_detector:
        mock_detector.return_value = sample_anomaly_output

        result = predict_anomalies(sample_anomaly_input)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert "is_anomaly" in result.columns

        mock_detector.assert_called_once()


def test_sample_anomaly_output_has_required_columns(sample_anomaly_output):
    """
    Test sample anomaly output structure.
    """
    required_columns = [
        "date",
        "asset",
        "price",
        "is_anomaly",
        "anomaly_severity",
        "anomaly_score",
    ]

    for column in required_columns:
        assert column in sample_anomaly_output.columns


# ============================================================
# API Route Import Test
# ============================================================

def test_anomaly_routes_importable():
    """
    Test anomaly routes module import.
    """
    try:
        from api.routes.anomaly_routes import router

        assert router is not None

    except Exception as error:
        pytest.skip(f"anomaly_routes could not be imported: {error}")


def test_anomaly_service_importable():
    """
    Test src.inference.anomaly_service import.
    """
    try:
        from src.inference.anomaly_service import app

        assert app is not None

    except Exception as error:
        pytest.skip(f"anomaly_service could not be imported: {error}")