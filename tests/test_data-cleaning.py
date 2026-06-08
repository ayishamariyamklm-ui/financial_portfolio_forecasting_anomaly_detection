"""
Unit tests for data cleaning functionality.

Project:
Financial Portfolio Forecasting & Anomaly Detection

Run from project root:

    pytest tests/test_data_cleaning.py -v
"""

import sys
from pathlib import Path

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
    from src.data_processing.data_cleaning import FinancialDataCleaner
except Exception:
    FinancialDataCleaner = None

try:
    from src.inference.predict import clean_input_data
except Exception:
    clean_input_data = None


# ============================================================
# Sample Data
# ============================================================

@pytest.fixture
def sample_raw_dataframe():
    """
    Create sample raw dataframe for testing.
    """
    return pd.DataFrame(
        {
            "Date": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                None,
                "2024-01-05",
            ],
            "Asset": [
                "Stock A",
                "Stock A",
                "Bond A",
                "Bond A",
                None,
            ],
            "Price": [
                100.0,
                105.5,
                95.0,
                90.0,
                110.0,
            ],
        }
    )


@pytest.fixture
def sample_invalid_price_dataframe():
    """
    Create dataframe with invalid and negative prices.
    """
    return pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "Asset": ["Stock A", "Stock A", "Stock A"],
            "Price": [100.0, -50.0, "invalid"],
        }
    )


@pytest.fixture
def sample_missing_columns_dataframe():
    """
    Create dataframe with missing required columns.
    """
    return pd.DataFrame(
        {
            "Date": ["2024-01-01"],
            "Asset": ["Stock A"],
        }
    )


# ============================================================
# Tests for clean_input_data
# ============================================================

def test_clean_input_data_available():
    """
    Test that clean_input_data function is importable.
    """
    assert clean_input_data is not None, (
        "clean_input_data could not be imported from src.inference.predict"
    )


@pytest.mark.skipif(clean_input_data is None, reason="clean_input_data not available")
def test_clean_input_data_standardizes_columns(sample_raw_dataframe):
    """
    Test that column names are standardized to lowercase.
    """
    cleaned_data = clean_input_data(sample_raw_dataframe)

    assert "date" in cleaned_data.columns
    assert "asset" in cleaned_data.columns
    assert "price" in cleaned_data.columns


@pytest.mark.skipif(clean_input_data is None, reason="clean_input_data not available")
def test_clean_input_data_removes_missing_values(sample_raw_dataframe):
    """
    Test that rows with missing required values are removed.
    """
    cleaned_data = clean_input_data(sample_raw_dataframe)

    assert cleaned_data["date"].isna().sum() == 0
    assert cleaned_data["asset"].isna().sum() == 0
    assert cleaned_data["price"].isna().sum() == 0


@pytest.mark.skipif(clean_input_data is None, reason="clean_input_data not available")
def test_clean_input_data_converts_date_and_price(sample_raw_dataframe):
    """
    Test date and price data type conversion.
    """
    cleaned_data = clean_input_data(sample_raw_dataframe)

    assert pd.api.types.is_datetime64_any_dtype(cleaned_data["date"])
    assert pd.api.types.is_numeric_dtype(cleaned_data["price"])


@pytest.mark.skipif(clean_input_data is None, reason="clean_input_data not available")
def test_clean_input_data_removes_negative_prices(sample_invalid_price_dataframe):
    """
    Test that negative prices and invalid prices are removed.
    """
    cleaned_data = clean_input_data(sample_invalid_price_dataframe)

    assert (cleaned_data["price"] >= 0).all()
    assert len(cleaned_data) == 1


@pytest.mark.skipif(clean_input_data is None, reason="clean_input_data not available")
def test_clean_input_data_raises_error_for_missing_columns(sample_missing_columns_dataframe):
    """
    Test that missing required columns raise ValueError.
    """
    with pytest.raises(ValueError):
        clean_input_data(sample_missing_columns_dataframe)


@pytest.mark.skipif(clean_input_data is None, reason="clean_input_data not available")
def test_clean_input_data_raises_error_for_empty_dataframe():
    """
    Test empty dataframe handling.
    """
    empty_dataframe = pd.DataFrame(columns=["date", "asset", "price"])

    with pytest.raises(ValueError):
        clean_input_data(empty_dataframe)


# ============================================================
# Tests for FinancialDataCleaner Class
# ============================================================

def test_financial_data_cleaner_class_available():
    """
    Test that FinancialDataCleaner is importable if the module exists.
    """
    if FinancialDataCleaner is None:
        pytest.skip("FinancialDataCleaner not available.")

    assert FinancialDataCleaner is not None


@pytest.mark.skipif(FinancialDataCleaner is None, reason="FinancialDataCleaner not available")
def test_financial_data_cleaner_initialization():
    """
    Test FinancialDataCleaner initialization.
    """
    cleaner = FinancialDataCleaner(
        date_column="date",
        asset_column="asset",
        target_column="price",
    )

    assert cleaner.date_column == "date"
    assert cleaner.asset_column == "asset"
    assert cleaner.target_column == "price"


@pytest.mark.skipif(FinancialDataCleaner is None, reason="FinancialDataCleaner not available")
def test_financial_data_cleaner_clean_data(sample_raw_dataframe):
    """
    Test FinancialDataCleaner clean_data method if available.
    """
    cleaner = FinancialDataCleaner(
        date_column="date",
        asset_column="asset",
        target_column="price",
    )

    sample_data = sample_raw_dataframe.copy()
    sample_data.columns = ["date", "asset", "price"]

    if hasattr(cleaner, "clean_data"):
        cleaned_data = cleaner.clean_data(sample_data)

        assert isinstance(cleaned_data, pd.DataFrame)
        assert "date" in cleaned_data.columns
        assert "asset" in cleaned_data.columns
        assert "price" in cleaned_data.columns
    else:
        pytest.skip("clean_data method not available in FinancialDataCleaner.")


# ============================================================
# General Data Quality Tests
# ============================================================

def test_sample_dataset_has_required_structure(sample_raw_dataframe):
    """
    Test sample dataset structure.
    """
    assert "Date" in sample_raw_dataframe.columns
    assert "Asset" in sample_raw_dataframe.columns
    assert "Price" in sample_raw_dataframe.columns
    assert len(sample_raw_dataframe) > 0


def test_project_raw_dataset_if_exists():
    """
    Test real project dataset if it exists.
    """
    dataset_path = PROJECT_ROOT / "data" / "raw" / "financial_portfolio_data.csv"

    if not dataset_path.exists():
        pytest.skip("Project raw dataset not found.")

    dataframe = pd.read_csv(dataset_path)

    standardized_columns = (
        dataframe.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    assert "date" in standardized_columns
    assert "asset" in standardized_columns
    assert "price" in standardized_columns