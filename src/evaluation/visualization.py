"""
Visualization utilities for the Financial Portfolio Forecasting
and Anomaly Detection project.

This module creates and saves visualizations for:

1. Dataset exploration
2. Time series trends
3. Forecasting model results
4. Model comparison
5. Anomaly detection results
6. Feature importance

Dataset expected columns:
- date
- asset
- price
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


# ============================================================
# Directory Helper
# ============================================================

def ensure_visualization_directories() -> None:
    """
    Create required directories for visualization outputs.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Data Cleaning Helper
# ============================================================

def clean_visualization_data(
    dataframe: pd.DataFrame,
    date_column: str = "date",
    asset_column: str = "asset",
    price_column: str = "price",
) -> pd.DataFrame:
    """
    Clean dataframe for visualization.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        date_column (str): Date column name.
        asset_column (str): Asset column name.
        price_column (str): Price column name.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    data = dataframe.copy()

    data.columns = (
        data.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    required_columns = [date_column, asset_column, price_column]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    data[price_column] = pd.to_numeric(data[price_column], errors="coerce")

    data = data.dropna(subset=[date_column, asset_column, price_column])
    data = data[data[price_column] >= 0]
    data = data.sort_values([asset_column, date_column]).reset_index(drop=True)

    return data


def save_current_figure(
    file_path: Path,
    dpi: int = 300,
) -> None:
    """
    Save current matplotlib figure.

    Args:
        file_path (Path): Output file path.
        dpi (int): Image resolution.
    """
    ensure_visualization_directories()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(file_path, dpi=dpi, bbox_inches="tight")


# ============================================================
# Dataset Visualizations
# ============================================================

def plot_records_per_asset(
    dataframe: pd.DataFrame,
    asset_column: str = "asset",
    save_path: Path = FIGURES_DIR / "records_per_asset.png",
    show: bool = True,
) -> Path:
    """
    Plot number of records per asset.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        asset_column (str): Asset column name.
        save_path (Path): Output figure path.
        show (bool): Whether to display plot.

    Returns:
        Path: Saved figure path.
    """
    ensure_visualization_directories()

    if asset_column not in dataframe.columns:
        raise ValueError(f"Missing asset column: {asset_column}")

    asset_counts = dataframe[asset_column].value_counts().sort_values(ascending=False)

    plt.figure(figsize=(10, 5))
    asset_counts.plot(kind="bar")
    plt.title("Number of Records per Asset")
    plt.xlabel("Asset")
    plt.ylabel("Record Count")
    plt.xticks(rotation=45, ha="right")
    save_current_figure(save_path)

    if show:
        plt.show()
    else:
        plt.close()

    return save_path


def plot_price_distribution(
    dataframe: pd.DataFrame,
    price_column: str = "price",
    save_path: Path = FIGURES_DIR / "price_distribution.png",
    show: bool = True,
) -> Path:
    """
    Plot overall price distribution.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        price_column (str): Price column name.
        save_path (Path): Output figure path.
        show (bool): Whether to display plot.

    Returns:
        Path: Saved figure path.
    """
    ensure_visualization_directories()

    if price_column not in dataframe.columns:
        raise ValueError(f"Missing price column: {price_column}")

    plt.figure(figsize=(10, 5))
    plt.hist(dataframe[price_column].dropna(), bins=30)
    plt.title("Overall Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    save_current_figure(save_path)

    if show:
        plt.show()
    else:
        plt.close()

    return save_path


def plot_price_distribution_by_asset(
    dataframe: pd.DataFrame,
    asset_column: str = "asset",
    price_column: str = "price",
    save_path: Path = FIGURES_DIR / "price_distribution_by_asset.png",
    show: bool = True,
) -> Path:
    """
    Plot price distribution by asset using boxplot.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        asset_column (str): Asset column name.
        price_column (str): Price column name.
        save_path (Path): Output figure path.
        show (bool): Whether to display plot.

    Returns:
        Path: Saved figure path.
    """
    ensure_visualization_directories()

    required_columns = [asset_column, price_column]

    missing_columns = [
        column for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    plt.figure(figsize=(12, 6))
    dataframe.boxplot(column=price_column, by=asset_column, rot=45)
    plt.title("Price Distribution by Asset")
    plt.suptitle("")
    plt.xlabel("Asset")
    plt.ylabel("Price")
    save_current_figure(save_path)

    if show:
        plt.show()
    else:
        plt.close()

    return save_path


# ============================================================
# Time Series Visualizations
# ============================================================

def plot_asset_price_trends(
    dataframe: pd.DataFrame,
    date_column: str = "date",
    asset_column: str = "asset",
    price_column: str = "price",
    save_path: Path = FIGURES_DIR / "asset_price_trends.png",
    show: bool = True,
) -> Path:
    """
    Plot price trend over time for all assets.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        date_column (str): Date column name.
        asset_column (str): Asset column name.
        price_column (str): Price column name.
        save_path (Path): Output figure path.
        show (bool): Whether to display plot.

    Returns:
        Path: Saved figure path.
    """
    ensure_visualization_directories()

    data = dataframe.copy()
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    data = data.sort_values([asset_column, date_column])

    plt.figure(figsize=(14, 6))

    for asset in data[asset_column].dropna().unique():
        asset_data = data[data[asset_column] == asset]

        plt.plot(
            asset_data[date_column],
            asset_data[price_column],
            linewidth=1,
            label=str(asset),
        )

    plt.title("Asset Price Trends Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(title="Asset", bbox_to_anchor=(1.05, 1), loc="upper left")
    save_current_figure(save_path)

    if show:
        plt.show()
    else:
        plt.close()

    return save_path


def plot_individual_asset_trends(
    dataframe: pd.DataFrame,
    date_column: str = "date",
    asset_column: str = "asset",
    price_column: str = "price",
    output_dir: Path = FIGURES_DIR,
    show: bool = True,
) -> List[Path]:
    """
    Plot separate price trend chart for each asset.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        date_column (str): Date column name.
        asset_column (str): Asset column name.
        price_column (str): Price column name.
        output_dir (Path): Output directory.
        show (bool): Whether to display plots.

    Returns:
        List[Path]: Saved figure paths.
    """
    ensure_visualization_directories()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = dataframe.copy()
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    data = data.sort_values([asset_column, date_column])

    saved_paths = []

    for asset in data[asset_column].dropna().unique():
        asset_data = data[data[asset_column] == asset]

        safe_asset_name = (
            str(asset)
            .lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )

        save_path = output_dir / f"price_trend_{safe_asset_name}.png"

        plt.figure(figsize=(12, 5))
        plt.plot(
            asset_data[date_column],
            asset_data[price_column],
            marker="o",
            linewidth=1.5,
        )
        plt.title(f"Price Trend Over Time - {asset}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        save_current_figure(save_path)

        if show:
            plt.show()
        else:
            plt.close()

        saved_paths.append(save_path)

    return saved_paths


def plot_daily_average_price(
    dataframe: pd.DataFrame,
    date_column: str = "date",
    price_column: str = "price",
    save_path: Path = FIGURES_DIR / "daily_average_price_trend.png",
    show: bool = True,
) -> Path:
    """
    Plot daily average asset price trend.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        date_column (str): Date column name.
        price_column (str): Price column name.
        save_path (Path): Output figure path.
        show (bool): Whether to display plot.

    Returns:
        Path: Saved figure path.
    """
    ensure_visualization_directories()

    data = dataframe.copy()
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")

    daily_average = (
        data.groupby(date_column, as_index=False)
        .agg(average_price=(price_column, "mean"))
        .sort_values(date_column)
    )

    plt.figure(figsize=(12, 5))
    plt.plot(
        daily_average[date_column],
        daily_average["average_price"],
        marker="o",
        linewidth=1.5,
    )
    plt.title("Daily Average Asset Price Trend")
    plt.xlabel("Date")
    plt.ylabel("Average Price")
    save_current_figure(save_path)

    if show:
        plt.show()
    else:
        plt.close()

    return save_path


def plot_monthly_average_price_by_asset(
    dataframe: pd.DataFrame,
    date_column: str = "date",
    asset_column: str = "asset",
    price_column: str = "price",
    save_path: Path = FIGURES_DIR / "monthly_average_price_by_asset.png",
    show: bool = True,
) -> Path:
    """
    Plot monthly average price by asset.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        date_column (str): Date column name.
        asset_column (str): Asset column name.
        price_column (str): Price column name.
        save_path (Path): Output figure path.
        show (bool): Whether to display plot.

    Returns:
        Path: Saved figure path.
    """
    ensure_visualization_directories()

    data = dataframe.copy()
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    data["year_month"] = data[date_column].dt.to_period("M").astype(str)

    monthly_data = (
        data.groupby(["year_month", asset_column], as_index=False)
        .agg(average_price=(price_column, "mean"))
        .sort_values(["year_month", asset_column])
    )

    plt.figure(figsize=(14, 6))

    for asset in monthly_data[asset_column].dropna().unique():
        asset_data = monthly_data[monthly_data[asset_column] == asset]

        plt.plot(
            asset_data["year_month"],
            asset_data["average_price"],
            marker="o",
            linewidth=1.5,
            label=str(asset),
        )

    plt.title("Monthly Average Price by Asset")
    plt.xlabel("Month")
    plt.ylabel("Average Price")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Asset", bbox_to_anchor=(1.05, 1), loc="upper left")
    save_current_figure(save_path)

    if show:
        plt.show()
    else:
        plt.close()

    return save_path


# ============================================================
# Return and Volatility Visualizations
# ============================================================

def add_return_column(
    dataframe: pd.DataFrame,
    asset_column: str = "asset",
    price_column: str = "price",
) -> pd.DataFrame:
    """
    Add daily return column.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        asset_column (str): Asset column name.
        price_column (str): Price column name.

    Returns:
        pd.DataFrame: Dataframe with daily_return.
    """
    data = dataframe.copy()

    data["daily_return"] = (
        data.groupby(asset_column)[price_column]
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
    )

    data["daily_return"] = data["daily_return"].fillna(0)

    return data


def plot_daily_return_distribution(
    dataframe: pd.DataFrame,
    return_column: str = "daily_return",
    save_path: Path = FIGURES_DIR / "daily_return_distribution.png",
    show: bool = True,
) -> Path:
    """
    Plot daily return distribution.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        return_column (str): Return column name.
        save_path (Path): Output figure path.
        show (bool): Whether to display plot.

    Returns:
        Path: Saved figure path.
    """
    ensure_visualization_directories()

    data = dataframe.copy()

    if return_column not in data.columns:
        data = add_return_column(data)

    plt.figure(figsize=(10, 5))
    plt.hist(data[return_column].dropna(), bins=30)
    plt.title("Daily Return Distribution")
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    save_current_figure(save_path)

    if show:
        plt.show()
    else:
        plt.close()

    return save_path


def plot_daily_return_trends(
    dataframe: pd.DataFrame,
    date_column: str = "date",
    asset_column: str = "asset",
    return_column: str = "daily_return",
    save_path: Path = FIGURES_DIR / "daily_return_trends.png",
    show: bool = True,
) -> Path:
    """
    Plot daily return trend by asset.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        date_column (str): Date column name.
        asset_column (str): Asset column name.
        return_column (str): Return column name.
        save_path (Path): Output figure path.
        show (bool): Whether to display plot.

    Returns:
        Path: Saved figure path.
    """
    ensure_visualization_directories()

    data = dataframe.copy()

    if return_column not in data.columns:
        data = add_return_column(data)

    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")

    plt.figure(figsize=(14, 6))

    for asset in data[asset_column].dropna().unique():
        asset_data = data[data[asset_column] == asset].sort_values(date_column)

        plt.plot(
            asset_data[date_column],
            asset_data[return_column],
            linewidth=1,
            label=str(asset),
        )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title("Daily Return Trend by Asset")
    plt.xlabel("Date")
    plt.ylabel("Daily Return")
    plt.legend(title="Asset", bbox_to_anchor=(1.05, 1), loc="upper left")
    save_current_figure(save_path)

    if show:
        plt.show()
    else:
        plt.close()

    return save_path


def plot_volatility_ranking(
    dataframe: pd.DataFrame,
    asset_column: str = "asset",
    return_column: str = "daily_return",
    save_path: Path = FIGURES_DIR / "volatility_ranking_by_asset.png",
    show: bool = True,
) -> Path:
    """
    Plot asset volatility ranking using daily return standard deviation.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        asset_column (str): Asset column name.
        return_column (str): Return column name.
        save_path (Path): Output figure path.
        show (bool): Whether to display plot.

    Returns:
        Path: Saved figure path.
    """
    ensure_visualization_directories()

    data = dataframe.copy()

    if return_column not in data.columns:
        data = add_return_column(data)

    volatility = (
        data.groupby(asset_column)[return_column]
        .std()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(10, 5))
    volatility.plot(kind="bar")
    plt.title("Return Volatility Ranking by Asset")
    plt.xlabel("Asset")
    plt.ylabel("Return Standard Deviation")
    plt.xticks(rotation=45, ha="right")
    save_current_figure(save_path)

    if show:
        plt.show()
    else:
        plt.close()

    return save_path


# ============================================================
# Forecasting Visualizations
# ============================================================

def plot_actual_vs_predicted(
    dataframe: pd.DataFrame,
    date_column: str = "date",
    asset_column: Optional[str] = "asset",
    actual_column: str = "actual_price",
    predicted_column: str = "predicted_price",
    save_path: Path = FIGURES_DIR / "actual_vs_predicted.png",
    show: bool = True,
) -> Path:
    """
    Plot actual vs predicted price values.

    Args:
        dataframe (pd.DataFrame): Prediction dataframe.
        date_column (str): Date column name.
        asset_column (Optional[str]): Asset column name.
        actual_column (str): Actual price column.
        predicted_column (str): Predicted price column.
        save_path (Path): Output figure path.
        show (bool): Whether to display plot.

    Returns:
        Path: Saved figure path.
    """
    ensure_visualization_directories()

    required_columns = [date_column, actual_column, predicted_column]

    missing_columns = [
        column for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = dataframe.copy()
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    data = data.sort_values(date_column)

    plt.figure(figsize=(14, 6))

    if asset_column and asset_column in data.columns:
        for asset in data[asset_column].dropna().unique():
            asset_data = data[data[asset_column] == asset].sort_values(date_column)

            plt.plot(
                asset_data[date_column],
                asset_data[actual_column],
                linewidth=1.5,
                label=f"{asset} Actual",
            )

            plt.plot(
                asset_data[date_column],
                asset_data[predicted_column],
                linestyle="--",
                linewidth=1.5,
                label=f"{asset} Predicted",
            )
    else:
        plt.plot(
            data[date_column],
            data[actual_column],
            linewidth=1.5,
            label="Actual",
        )
        plt.plot(
            data[date_column],
            data[predicted_column],
            linestyle="--",
            linewidth=1.5,
            label="Predicted",
        )

    plt.title("Actual vs Predicted Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    save_current_figure(save_path)

    if show:
        plt.show()
    else:
        plt.close()

    return save_path


def plot_forecast_results(
    historical_dataframe: pd.DataFrame,
    forecast_dataframe: pd.DataFrame,
    date_column: str = "date",
    asset_column: str = "asset",
    price_column: str = "price",
    forecast_column: str = "forecasted_price",
    save_path: Path = FIGURES_DIR / "forecast_results.png",
    show: bool = True,
) -> Path:
    """
    Plot historical price and future forecasted price.

    Args:
        historical_dataframe (pd.DataFrame): Historical dataframe.
        forecast_dataframe (pd.DataFrame): Forecast dataframe.
        date_column (str): Date column name.
        asset_column (str): Asset column name.
        price_column (str): Historical price column.
        forecast_column (str): Forecast price column.
        save_path (Path): Output figure path.
        show (bool): Whether to display plot.

    Returns:
        Path: Saved figure path.
    """
    ensure_visualization_directories()

    historical = historical_dataframe.copy()
    forecast = forecast_dataframe.copy()

    historical[date_column] = pd.to_datetime(historical[date_column], errors="coerce")
    forecast[date_column] = pd.to_datetime(forecast[date_column], errors="coerce")

    plt.figure(figsize=(14, 6))

    for asset in historical[asset_column].dropna().unique():
        historical_asset = historical[historical[asset_column] == asset].sort_values(date_column)
        forecast_asset = forecast[forecast[asset_column] == asset].sort_values(date_column)

        plt.plot(
            historical_asset[date_column],
            historical_asset[price_column],
            linewidth=1.5,
            label=f"{asset} Historical",
        )

        if not forecast_asset.empty:
            plt.plot(
                forecast_asset[date_column],
                forecast_asset[forecast_column],
                linestyle="--",
                linewidth=1.5,
                label=f"{asset} Forecast",
            )

    plt.title("Historical and Forecasted Asset Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    save_current_figure(save_path)

    if show:
        plt.show()
    else:
        plt.close()

    return save_path


def plot_model_comparison(
    comparison_dataframe: pd.DataFrame,
    metric_column: str = "rmse",
    model_column: str = "model_name",
    save_path: Path = FIGURES_DIR / "model_comparison_rmse.png",
    show: bool = True,
) -> Path:
    """
    Plot model comparison using a selected metric.

    Args:
        comparison_dataframe (pd.DataFrame): Model comparison dataframe.
        metric_column (str): Metric column to plot.
        model_column (str): Model name column.
        save_path (Path): Output figure path.
        show (bool): Whether to display plot.

    Returns:
        Path: Saved figure path.
    """
    ensure_visualization_directories()

    required_columns = [model_column, metric_column]

    missing_columns = [
        column for column in required_columns
        if column not in comparison_dataframe.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = comparison_dataframe.copy()
    data[metric_column] = pd.to_numeric(data[metric_column], errors="coerce")
    data = data.dropna(subset=[metric_column])
    data = data.sort_values(metric_column, ascending=True)

    plt.figure(figsize=(10, 5))
    plt.bar(data[model_column], data[metric_column])
    plt.title(f"Model Comparison by {metric_column.upper()}")
    plt.xlabel("Model")
    plt.ylabel(metric_column.upper())
    plt.xticks(rotation=45, ha="right")
    save_current_figure(save_path)

    if show:
        plt.show()
    else:
        plt.close()

    return save_path


# ============================================================
# Anomaly Visualizations
# ============================================================

def plot_anomaly_results(
    dataframe: pd.DataFrame,
    date_column: str = "date",
    asset_column: str = "asset",
    price_column: str = "price",
    anomaly_column: str = "is_anomaly",
    save_path: Path = FIGURES_DIR / "anomaly_results.png",
    show: bool = True,
) -> Path:
    """
    Plot asset prices with anomaly points marked.

    Args:
        dataframe (pd.DataFrame): Anomaly result dataframe.
        date_column (str): Date column name.
        asset_column (str): Asset column name.
        price_column (str): Price column name.
        anomaly_column (str): Boolean anomaly column.
        save_path (Path): Output figure path.
        show (bool): Whether to display plot.

    Returns:
        Path: Saved figure path.
    """
    ensure_visualization_directories()

    required_columns = [date_column, asset_column, price_column, anomaly_column]

    missing_columns = [
        column for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = dataframe.copy()
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    data[anomaly_column] = data[anomaly_column].astype(bool)

    plt.figure(figsize=(14, 6))

    for asset in data[asset_column].dropna().unique():
        asset_data = data[data[asset_column] == asset].sort_values(date_column)

        plt.plot(
            asset_data[date_column],
            asset_data[price_column],
            linewidth=1,
            label=str(asset),
        )

    anomaly_points = data[data[anomaly_column]]

    if not anomaly_points.empty:
        plt.scatter(
            anomaly_points[date_column],
            anomaly_points[price_column],
            marker="x",
            s=80,
            label="Anomaly",
        )

    plt.title("Asset Price Anomalies")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    save_current_figure(save_path)

    if show:
        plt.show()
    else:
        plt.close()

    return save_path


def plot_anomaly_count_by_asset(
    dataframe: pd.DataFrame,
    asset_column: str = "asset",
    anomaly_column: str = "is_anomaly",
    save_path: Path = FIGURES_DIR / "anomaly_count_by_asset.png",
    show: bool = True,
) -> Path:
    """
    Plot anomaly count by asset.

    Args:
        dataframe (pd.DataFrame): Anomaly result dataframe.
        asset_column (str): Asset column name.
        anomaly_column (str): Anomaly column name.
        save_path (Path): Output figure path.
        show (bool): Whether to display plot.

    Returns:
        Path: Saved figure path.
    """
    ensure_visualization_directories()

    required_columns = [asset_column, anomaly_column]

    missing_columns = [
        column for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = dataframe.copy()
    data[anomaly_column] = data[anomaly_column].astype(bool)

    anomaly_counts = (
        data.groupby(asset_column)[anomaly_column]
        .sum()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(10, 5))
    anomaly_counts.plot(kind="bar")
    plt.title("Anomaly Count by Asset")
    plt.xlabel("Asset")
    plt.ylabel("Anomaly Count")
    plt.xticks(rotation=45, ha="right")
    save_current_figure(save_path)

    if show:
        plt.show()
    else:
        plt.close()

    return save_path


def plot_anomaly_method_comparison(
    dataframe: pd.DataFrame,
    method_columns: Optional[List[str]] = None,
    save_path: Path = FIGURES_DIR / "anomaly_method_comparison.png",
    show: bool = True,
) -> Path:
    """
    Plot anomaly count by detection method.

    Args:
        dataframe (pd.DataFrame): Anomaly result dataframe.
        method_columns (Optional[List[str]]): Method-specific anomaly columns.
        save_path (Path): Output figure path.
        show (bool): Whether to display plot.

    Returns:
        Path: Saved figure path.
    """
    ensure_visualization_directories()

    if method_columns is None:
        method_columns = [
            "is_anomaly_iqr",
            "is_anomaly_zscore",
            "is_anomaly_isolation_forest",
        ]

    available_columns = [
        column for column in method_columns
        if column in dataframe.columns
    ]

    if not available_columns:
        raise ValueError("No anomaly method columns found.")

    method_counts = {
        column.replace("is_anomaly_", "").replace("_", " ").title(): int(
            dataframe[column].astype(bool).sum()
        )
        for column in available_columns
    }

    method_series = pd.Series(method_counts).sort_values(ascending=False)

    plt.figure(figsize=(10, 5))
    method_series.plot(kind="bar")
    plt.title("Anomaly Count by Detection Method")
    plt.xlabel("Detection Method")
    plt.ylabel("Anomaly Count")
    plt.xticks(rotation=45, ha="right")
    save_current_figure(save_path)

    if show:
        plt.show()
    else:
        plt.close()

    return save_path


def plot_anomaly_severity_distribution(
    dataframe: pd.DataFrame,
    severity_column: str = "anomaly_severity",
    save_path: Path = FIGURES_DIR / "anomaly_severity_distribution.png",
    show: bool = True,
) -> Path:
    """
    Plot anomaly severity distribution.

    Args:
        dataframe (pd.DataFrame): Anomaly result dataframe.
        severity_column (str): Severity column.
        save_path (Path): Output figure path.
        show (bool): Whether to display plot.

    Returns:
        Path: Saved figure path.
    """
    ensure_visualization_directories()

    if severity_column not in dataframe.columns:
        raise ValueError(f"Missing severity column: {severity_column}")

    severity_counts = dataframe[severity_column].value_counts()

    plt.figure(figsize=(8, 5))
    severity_counts.plot(kind="bar")
    plt.title("Anomaly Severity Distribution")
    plt.xlabel("Severity")
    plt.ylabel("Record Count")
    plt.xticks(rotation=45, ha="right")
    save_current_figure(save_path)

    if show:
        plt.show()
    else:
        plt.close()

    return save_path


# ============================================================
# Feature Importance Visualization
# ============================================================

def plot_feature_importance(
    feature_importance_dataframe: pd.DataFrame,
    feature_column: str = "feature",
    importance_column: str = "importance",
    top_n: int = 15,
    save_path: Path = FIGURES_DIR / "feature_importance.png",
    show: bool = True,
) -> Path:
    """
    Plot feature importance.

    Args:
        feature_importance_dataframe (pd.DataFrame): Feature importance dataframe.
        feature_column (str): Feature name column.
        importance_column (str): Importance value column.
        top_n (int): Number of top features.
        save_path (Path): Output figure path.
        show (bool): Whether to display plot.

    Returns:
        Path: Saved figure path.
    """
    ensure_visualization_directories()

    required_columns = [feature_column, importance_column]

    missing_columns = [
        column for column in required_columns
        if column not in feature_importance_dataframe.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = feature_importance_dataframe.copy()
    data[importance_column] = pd.to_numeric(data[importance_column], errors="coerce")
    data = data.dropna(subset=[importance_column])
    data = data.sort_values(importance_column, ascending=False).head(top_n)
    data = data.sort_values(importance_column, ascending=True)

    plt.figure(figsize=(10, 7))
    plt.barh(data[feature_column], data[importance_column])
    plt.title(f"Top {top_n} Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    save_current_figure(save_path)

    if show:
        plt.show()
    else:
        plt.close()

    return save_path


# ============================================================
# Complete Visualization Report Runner
# ============================================================

def generate_basic_visualizations(
    dataframe: pd.DataFrame,
    show: bool = False,
) -> Dict[str, str]:
    """
    Generate basic dataset and time series visualizations.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        show (bool): Whether to display plots.

    Returns:
        Dict[str, str]: Saved figure paths.
    """
    ensure_visualization_directories()

    data = clean_visualization_data(dataframe)

    data = add_return_column(data)

    saved_paths = {}

    saved_paths["records_per_asset"] = str(
        plot_records_per_asset(data, show=show)
    )

    saved_paths["price_distribution"] = str(
        plot_price_distribution(data, show=show)
    )

    saved_paths["price_distribution_by_asset"] = str(
        plot_price_distribution_by_asset(data, show=show)
    )

    saved_paths["asset_price_trends"] = str(
        plot_asset_price_trends(data, show=show)
    )

    saved_paths["daily_average_price"] = str(
        plot_daily_average_price(data, show=show)
    )

    saved_paths["monthly_average_price_by_asset"] = str(
        plot_monthly_average_price_by_asset(data, show=show)
    )

    saved_paths["daily_return_distribution"] = str(
        plot_daily_return_distribution(data, show=show)
    )

    saved_paths["daily_return_trends"] = str(
        plot_daily_return_trends(data, show=show)
    )

    saved_paths["volatility_ranking"] = str(
        plot_volatility_ranking(data, show=show)
    )

    return saved_paths


def generate_forecasting_visualizations(
    historical_dataframe: pd.DataFrame,
    forecast_dataframe: Optional[pd.DataFrame] = None,
    prediction_dataframe: Optional[pd.DataFrame] = None,
    model_comparison_dataframe: Optional[pd.DataFrame] = None,
    show: bool = False,
) -> Dict[str, str]:
    """
    Generate forecasting-related visualizations.

    Args:
        historical_dataframe (pd.DataFrame): Historical dataframe.
        forecast_dataframe (Optional[pd.DataFrame]): Forecast dataframe.
        prediction_dataframe (Optional[pd.DataFrame]): Test prediction dataframe.
        model_comparison_dataframe (Optional[pd.DataFrame]): Model comparison dataframe.
        show (bool): Whether to display plots.

    Returns:
        Dict[str, str]: Saved figure paths.
    """
    ensure_visualization_directories()

    saved_paths = {}

    if prediction_dataframe is not None and not prediction_dataframe.empty:
        saved_paths["actual_vs_predicted"] = str(
            plot_actual_vs_predicted(
                prediction_dataframe,
                show=show,
            )
        )

    if forecast_dataframe is not None and not forecast_dataframe.empty:
        saved_paths["forecast_results"] = str(
            plot_forecast_results(
                historical_dataframe=historical_dataframe,
                forecast_dataframe=forecast_dataframe,
                show=show,
            )
        )

    if model_comparison_dataframe is not None and not model_comparison_dataframe.empty:
        saved_paths["model_comparison"] = str(
            plot_model_comparison(
                model_comparison_dataframe,
                metric_column="rmse",
                show=show,
            )
        )

    return saved_paths


def generate_anomaly_visualizations(
    anomaly_dataframe: pd.DataFrame,
    show: bool = False,
) -> Dict[str, str]:
    """
    Generate anomaly-related visualizations.

    Args:
        anomaly_dataframe (pd.DataFrame): Anomaly result dataframe.
        show (bool): Whether to display plots.

    Returns:
        Dict[str, str]: Saved figure paths.
    """
    ensure_visualization_directories()

    saved_paths = {}

    saved_paths["anomaly_results"] = str(
        plot_anomaly_results(anomaly_dataframe, show=show)
    )

    saved_paths["anomaly_count_by_asset"] = str(
        plot_anomaly_count_by_asset(anomaly_dataframe, show=show)
    )

    saved_paths["anomaly_method_comparison"] = str(
        plot_anomaly_method_comparison(anomaly_dataframe, show=show)
    )

    if "anomaly_severity" in anomaly_dataframe.columns:
        saved_paths["anomaly_severity_distribution"] = str(
            plot_anomaly_severity_distribution(anomaly_dataframe, show=show)
        )

    return saved_paths


# ============================================================
# Script Usage Example
# ============================================================

if __name__ == "__main__":
    ensure_visualization_directories()

    raw_data_path = PROJECT_ROOT / "data" / "raw" / "financial_portfolio_data.csv"
    anomaly_results_path = OUTPUTS_DIR / "anomaly_detection_results.csv"
    model_comparison_path = TABLES_DIR / "model_comparison.csv"

    if raw_data_path.exists():
        raw_data = pd.read_csv(raw_data_path)

        print("Generating basic visualizations...")
        basic_paths = generate_basic_visualizations(
            dataframe=raw_data,
            show=False,
        )

        print("Basic visualization files:")
        for name, path in basic_paths.items():
            print(f"- {name}: {path}")

    else:
        print(f"Raw data not found at: {raw_data_path}")

    if anomaly_results_path.exists():
        anomaly_data = pd.read_csv(anomaly_results_path)

        print("\nGenerating anomaly visualizations...")
        anomaly_paths = generate_anomaly_visualizations(
            anomaly_dataframe=anomaly_data,
            show=False,
        )

        print("Anomaly visualization files:")
        for name, path in anomaly_paths.items():
            print(f"- {name}: {path}")

    else:
        print(f"\nAnomaly results not found at: {anomaly_results_path}")

    if model_comparison_path.exists():
        comparison_data = pd.read_csv(model_comparison_path)

        print("\nGenerating model comparison visualization...")
        model_comparison_figure = plot_model_comparison(
            comparison_dataframe=comparison_data,
            show=False,
        )

        print(f"Model comparison figure: {model_comparison_figure}")

    else:
        print(f"\nModel comparison file not found at: {model_comparison_path}")