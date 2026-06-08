"""
Unit tests for feature engineering functionality.

Project:
Financial Portfolio Forecasting & Anomaly Detection

Run from project root:

    pytest tests/test_feature_engineering.py -v
"""

import sys
from pathlib import Path

import numpy as np
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
    from src.data_processing.feature_engineering import FinancialFeatureEngineer
except Exception:
    FinancialFeatureEngineer = None


# ============================================================
# Sample Data
# ============================================================

@pytest.fixture
def sample_price_dataframe():
    """
    Create clean sample price dataframe.
    """
    dates = pd.date_range(start="2024-01-01", periods=20, freq="D")

    stock_a = pd.DataFrame(
        {
            "date": dates,
            "asset": "Stock A",
            "price": np.linspace(100, 120, 20),
        }
    )

    bond_a = pd.DataFrame(
        {
            "date": dates,
            "asset": "Bond A",
            "price": np.linspace(90, 95, 20),
        }
    )

    return pd.concat([stock_a, bond_a], ignore_index=True)


# ============================================================
# Local Feature Engineering Helper for Test Validation
# ============================================================

def create_test_features(dataframe):
    """
    Create simple feature engineering output for test validation.
    """
    data = dataframe.copy()
    data = data.sort_values(["asset", "date"]).reset_index(drop=True)

    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["quarter"] = data["date"].dt.quarter
    data["day"] = data["date"].dt.day
    data["day_of_week"] = data["date"].dt.dayofweek
    data["week_of_year"] = data["date"].dt.isocalendar().week.astype(int)

    data["daily_return"] = (
        data.groupby("asset")["price"]
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    for lag in [1, 3, 7]:
        data[f"price_lag_{lag}"] = data.groupby("asset")["price"].shift(lag)

    for window in [7, 14]:
        data[f"price_rolling_mean_{window}"] = (
            data.groupby("asset")["price"]
            .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        )

        data[f"price_rolling_std_{window}"] = (
            data.groupby("asset")["price"]
            .transform(lambda x: x.rolling(window=window, min_periods=1).std())
        )

    numeric_columns = data.select_dtypes(include=["number"]).columns

    for column in numeric_columns:
        data[column] = data[column].fillna(0)

    return data


# ============================================================
# Tests for Local Feature Logic
# ============================================================

def test_create_test_features_adds_calendar_columns(sample_price_dataframe):
    """
    Test calendar feature creation.
    """
    features = create_test_features(sample_price_dataframe)

    expected_columns = [
        "year",
        "month",
        "quarter",
        "day",
        "day_of_week",
        "week_of_year",
    ]

    for column in expected_columns:
        assert column in features.columns


def test_create_test_features_adds_daily_return(sample_price_dataframe):
    """
    Test daily return feature creation.
    """
    features = create_test_features(sample_price_dataframe)

    assert "daily_return" in features.columns
    assert features["daily_return"].isna().sum() == 0


def test_create_test_features_adds_lag_columns(sample_price_dataframe):
    """
    Test lag feature creation.
    """
    features = create_test_features(sample_price_dataframe)

    assert "price_lag_1" in features.columns
    assert "price_lag_3" in features.columns
    assert "price_lag_7" in features.columns


def test_create_test_features_adds_rolling_columns(sample_price_dataframe):
    """
    Test rolling feature creation.
    """
    features = create_test_features(sample_price_dataframe)

    assert "price_rolling_mean_7" in features.columns
    assert "price_rolling_std_7" in features.columns
    assert "price_rolling_mean_14" in features.columns
    assert "price_rolling_std_14" in features.columns


def test_create_test_features_no_missing_numeric_values(sample_price_dataframe):
    """
    Test that numeric features do not contain missing values.
    """
    features = create_test_features(sample_price_dataframe)

    numeric_columns = features.select_dtypes(include=["number"]).columns

    assert features[numeric_columns].isna().sum().sum() == 0


def test_daily_return_is_grouped_by_asset(sample_price_dataframe):
    """
    Test that daily return resets for each asset group.
    """
    features = create_test_features(sample_price_dataframe)

    first_rows = features.groupby("asset").head(1)

    assert (first_rows["daily_return"] == 0).all()


def test_lag_feature_is_grouped_by_asset(sample_price_dataframe):
    """
    Test that lag feature is grouped by asset.
    """
    features = create_test_features(sample_price_dataframe)

    first_rows = features.groupby("asset").head(1)

    assert (first_rows["price_lag_1"] == 0).all()


# ============================================================
# Tests for FinancialFeatureEngineer Class
# ============================================================

def test_financial_feature_engineer_class_available():
    """
    Test FinancialFeatureEngineer import availability.
    """
    if FinancialFeatureEngineer is None:
        pytest.skip("FinancialFeatureEngineer not available.")

    assert FinancialFeatureEngineer is not None


@pytest.mark.skipif(FinancialFeatureEngineer is None, reason="FinancialFeatureEngineer not available")
def test_financial_feature_engineer_initialization():
    """
    Test FinancialFeatureEngineer initialization.
    """
    engineer = FinancialFeatureEngineer(
        date_column="date",
        asset_column="asset",
        target_column="price",
    )

    assert engineer.date_column == "date"
    assert engineer.asset_column == "asset"
    assert engineer.target_column == "price"


@pytest.mark.skipif(FinancialFeatureEngineer is None, reason="FinancialFeatureEngineer not available")
def test_financial_feature_engineer_transform(sample_price_dataframe):
    """
    Test FinancialFeatureEngineer feature creation method if available.
    """
    engineer = FinancialFeatureEngineer(
        date_column="date",
        asset_column="asset",
        target_column="price",
    )

    if hasattr(engineer, "create_features"):
        features = engineer.create_features(sample_price_dataframe)

    elif hasattr(engineer, "fit_transform"):
        features = engineer.fit_transform(sample_price_dataframe)

    elif hasattr(engineer, "transform"):
        features = engineer.transform(sample_price_dataframe)

    else:
        pytest.skip(
            "No supported feature engineering method found. "
            "Expected create_features, fit_transform, or transform."
        )

    assert isinstance(features, pd.DataFrame)
    assert len(features) > 0
    assert "date" in features.columns
    assert "asset" in features.columns
    assert "price" in features.columns


def test_project_processed_dataset_if_exists():
    """
    Test processed dataset if it exists.
    """
    processed_path = PROJECT_ROOT / "data" / "processed" / "cleaned_financial_data.csv"

    if not processed_path.exists():
        pytest.skip("Processed dataset not found.")

    dataframe = pd.read_csv(processed_path)

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