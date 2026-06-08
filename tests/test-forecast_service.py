"""
Unit tests for forecast service functionality.

Run from project root:

    pytest tests/test_forecast_service.py -v
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
        generate_forecasts,
        summarize_forecast_results,
        predict_forecast,
    )
except Exception:
    clean_input_data = None
    generate_forecasts = None
    summarize_forecast_results = None
    predict_forecast = None


# ============================================================
# Sample Data
# ============================================================

@pytest.fixture
def sample_forecast_input():
    """
    Create sample historical price data.
    """
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=15, freq="D"),
            "asset": ["Stock A"] * 15,
            "price": [
                100,
                101,
                102,
                103,
                104,
                105,
                106,
                107,
                108,
                109,
                110,
                111,
                112,
                113,
                114,
            ],
        }
    )


@pytest.fixture
def sample_forecast_output():
    """
    Create sample forecast output dataframe.
    """
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-16", periods=5, freq="D"),
            "asset": ["Stock A"] * 5,
            "forecasted_price": [115, 116, 117, 118, 119],
            "model_name": ["XGBoost"] * 5,
        }
    )


# ============================================================
# Basic Import Tests
# ============================================================

def test_forecast_service_functions_available():
    """
    Test forecast-related functions are importable.
    """
    assert clean_input_data is not None, "clean_input_data import failed."
    assert summarize_forecast_results is not None, "summarize_forecast_results import failed."


# ============================================================
# Forecast Summary Tests
# ============================================================

@pytest.mark.skipif(summarize_forecast_results is None, reason="summarize_forecast_results not available")
def test_summarize_forecast_results_basic(sample_forecast_output):
    """
    Test forecast summary calculation.
    """
    summary = summarize_forecast_results(sample_forecast_output)

    assert isinstance(summary, dict)
    assert summary["forecast_rows"] == 5
    assert "XGBoost" in summary["models_used"]
    assert "Stock A" in summary["assets_forecasted"]
    assert summary["average_forecasted_price"] == 117.0


@pytest.mark.skipif(summarize_forecast_results is None, reason="summarize_forecast_results not available")
def test_summarize_forecast_results_empty_dataframe():
    """
    Test forecast summary with empty dataframe.
    """
    empty_dataframe = pd.DataFrame()

    summary = summarize_forecast_results(empty_dataframe)

    assert isinstance(summary, dict)
    assert summary["forecast_rows"] == 0
    assert summary["models_used"] == []
    assert summary["assets_forecasted"] == []


@pytest.mark.skipif(summarize_forecast_results is None, reason="summarize_forecast_results not available")
def test_summarize_forecast_results_predicted_price_column():
    """
    Test forecast summary with predicted_price instead of forecasted_price.
    """
    dataframe = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3),
            "asset": ["Stock A", "Stock A", "Stock A"],
            "predicted_price": [100, 101, 102],
            "model_name": ["TestModel"] * 3,
        }
    )

    summary = summarize_forecast_results(dataframe)

    assert summary["forecast_rows"] == 3
    assert summary["average_forecasted_price"] == 101.0


# ============================================================
# Clean Input Tests
# ============================================================

@pytest.mark.skipif(clean_input_data is None, reason="clean_input_data not available")
def test_forecast_clean_input_data(sample_forecast_input):
    """
    Test forecast input cleaning.
    """
    cleaned_data = clean_input_data(sample_forecast_input)

    assert isinstance(cleaned_data, pd.DataFrame)
    assert "date" in cleaned_data.columns
    assert "asset" in cleaned_data.columns
    assert "price" in cleaned_data.columns
    assert pd.api.types.is_datetime64_any_dtype(cleaned_data["date"])


@pytest.mark.skipif(clean_input_data is None, reason="clean_input_data not available")
def test_forecast_clean_input_data_with_capital_columns():
    """
    Test cleaning with capitalized columns.
    """
    dataframe = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02"],
            "Asset": ["Stock A", "Stock A"],
            "Price": [100, 101],
        }
    )

    cleaned_data = clean_input_data(dataframe)

    assert "date" in cleaned_data.columns
    assert "asset" in cleaned_data.columns
    assert "price" in cleaned_data.columns


# ============================================================
# Mocked Forecast Tests
# ============================================================

@pytest.mark.skipif(generate_forecasts is None, reason="generate_forecasts not available")
def test_generate_forecasts_invalid_model_raises_error(sample_forecast_input):
    """
    Test invalid forecasting model name.
    """
    with pytest.raises(ValueError):
        generate_forecasts(
            dataframe=sample_forecast_input,
            model_name="invalid_model",
            forecast_steps=5,
        )


@pytest.mark.skipif(predict_forecast is None, reason="predict_forecast not available")
def test_predict_forecast_with_mocked_generate_forecasts(
    sample_forecast_input,
    sample_forecast_output,
):
    """
    Test predict_forecast using mocked generate_forecasts.
    """
    with patch("src.inference.predict.generate_forecasts") as mock_generate:
        mock_generate.return_value = sample_forecast_output

        result = predict_forecast(
            dataframe=sample_forecast_input,
            model_name="xgboost",
            forecast_steps=5,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert "forecasted_price" in result.columns

        mock_generate.assert_called_once()


def test_forecast_output_has_required_columns(sample_forecast_output):
    """
    Test sample forecast output structure.
    """
    required_columns = ["date", "asset", "forecasted_price", "model_name"]

    for column in required_columns:
        assert column in sample_forecast_output.columns


# ============================================================
# API Route Import Test
# ============================================================

def test_forecast_routes_importable():
    """
    Test forecast routes module import.
    """
    try:
        from api.routes.forecast_routes import router

        assert router is not None

    except Exception as error:
        pytest.skip(f"forecast_routes could not be imported: {error}")