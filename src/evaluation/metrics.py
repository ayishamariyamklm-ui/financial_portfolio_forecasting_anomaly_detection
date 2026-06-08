"""
Evaluation metrics utilities for the Financial Portfolio Forecasting
and Anomaly Detection project.

This module provides reusable metric functions for:

1. Forecasting models
   - MAE
   - MSE
   - RMSE
   - MAPE
   - SMAPE
   - R2 Score

2. Anomaly detection results
   - Total anomalies
   - Anomaly rate
   - Asset-wise anomaly summary
   - Method-wise anomaly summary

Dataset expected columns:
- date
- asset
- price
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

FORECAST_METRICS_PATH = TABLES_DIR / "forecast_metrics.json"
FORECAST_METRICS_CSV_PATH = TABLES_DIR / "forecast_metrics.csv"
ANOMALY_METRICS_PATH = TABLES_DIR / "anomaly_metrics.json"
ANOMALY_METRICS_CSV_PATH = TABLES_DIR / "anomaly_metrics.csv"


# ============================================================
# Directory Helper
# ============================================================

def ensure_evaluation_directories() -> None:
    """
    Create required directories for evaluation outputs.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Forecasting Metric Functions
# ============================================================

def calculate_mae(
    y_true: Union[pd.Series, np.ndarray, List[float]],
    y_pred: Union[pd.Series, np.ndarray, List[float]],
) -> float:
    """
    Calculate Mean Absolute Error.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        float: MAE value.
    """
    return float(mean_absolute_error(y_true, y_pred))


def calculate_mse(
    y_true: Union[pd.Series, np.ndarray, List[float]],
    y_pred: Union[pd.Series, np.ndarray, List[float]],
) -> float:
    """
    Calculate Mean Squared Error.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        float: MSE value.
    """
    return float(mean_squared_error(y_true, y_pred))


def calculate_rmse(
    y_true: Union[pd.Series, np.ndarray, List[float]],
    y_pred: Union[pd.Series, np.ndarray, List[float]],
) -> float:
    """
    Calculate Root Mean Squared Error.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        float: RMSE value.
    """
    mse = calculate_mse(y_true, y_pred)
    return float(np.sqrt(mse))


def calculate_mape(
    y_true: Union[pd.Series, np.ndarray, List[float]],
    y_pred: Union[pd.Series, np.ndarray, List[float]],
) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        float: MAPE percentage value.
    """
    y_true_array = np.array(y_true, dtype=float)
    y_pred_array = np.array(y_pred, dtype=float)

    non_zero_mask = y_true_array != 0

    if non_zero_mask.sum() == 0:
        return 0.0

    mape = np.mean(
        np.abs(
            (y_true_array[non_zero_mask] - y_pred_array[non_zero_mask])
            / y_true_array[non_zero_mask]
        )
    ) * 100

    return float(mape)


def calculate_smape(
    y_true: Union[pd.Series, np.ndarray, List[float]],
    y_pred: Union[pd.Series, np.ndarray, List[float]],
) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        float: SMAPE percentage value.
    """
    y_true_array = np.array(y_true, dtype=float)
    y_pred_array = np.array(y_pred, dtype=float)

    denominator = (
        np.abs(y_true_array) + np.abs(y_pred_array)
    ) / 2

    non_zero_mask = denominator != 0

    if non_zero_mask.sum() == 0:
        return 0.0

    smape = np.mean(
        np.abs(y_true_array[non_zero_mask] - y_pred_array[non_zero_mask])
        / denominator[non_zero_mask]
    ) * 100

    return float(smape)


def calculate_r2_score(
    y_true: Union[pd.Series, np.ndarray, List[float]],
    y_pred: Union[pd.Series, np.ndarray, List[float]],
) -> float:
    """
    Calculate R2 score.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        float: R2 score.
    """
    try:
        return float(r2_score(y_true, y_pred))
    except Exception:
        return 0.0


def calculate_forecasting_metrics(
    y_true: Union[pd.Series, np.ndarray, List[float]],
    y_pred: Union[pd.Series, np.ndarray, List[float]],
    round_digits: int = 6,
) -> Dict[str, float]:
    """
    Calculate all forecasting metrics.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        round_digits (int): Decimal places for rounding.

    Returns:
        Dict[str, float]: Forecasting metrics.
    """
    metrics = {
        "mae": calculate_mae(y_true, y_pred),
        "mse": calculate_mse(y_true, y_pred),
        "rmse": calculate_rmse(y_true, y_pred),
        "mape": calculate_mape(y_true, y_pred),
        "smape": calculate_smape(y_true, y_pred),
        "r2_score": calculate_r2_score(y_true, y_pred),
    }

    return {
        metric_name: round(float(metric_value), round_digits)
        for metric_name, metric_value in metrics.items()
    }


# ============================================================
# Forecast Evaluation from DataFrame
# ============================================================

def evaluate_forecast_dataframe(
    dataframe: pd.DataFrame,
    actual_column: str = "actual_price",
    predicted_column: str = "predicted_price",
    asset_column: Optional[str] = "asset",
) -> Dict[str, Any]:
    """
    Evaluate forecast results from a dataframe.

    Args:
        dataframe (pd.DataFrame): Forecast results dataframe.
        actual_column (str): Actual value column.
        predicted_column (str): Predicted value column.
        asset_column (Optional[str]): Asset column.

    Returns:
        Dict[str, Any]: Overall and asset-wise metrics.
    """
    if dataframe.empty:
        raise ValueError("Forecast dataframe is empty.")

    required_columns = [actual_column, predicted_column]

    missing_columns = [
        column for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    clean_data = dataframe.dropna(
        subset=[actual_column, predicted_column]
    ).copy()

    overall_metrics = calculate_forecasting_metrics(
        y_true=clean_data[actual_column],
        y_pred=clean_data[predicted_column],
    )

    asset_metrics = {}

    if asset_column and asset_column in clean_data.columns:
        for asset_name, asset_data in clean_data.groupby(asset_column):
            asset_metrics[str(asset_name)] = calculate_forecasting_metrics(
                y_true=asset_data[actual_column],
                y_pred=asset_data[predicted_column],
            )

    return {
        "overall_metrics": overall_metrics,
        "asset_metrics": asset_metrics,
        "total_records": int(len(clean_data)),
        "actual_column": actual_column,
        "predicted_column": predicted_column,
    }


# ============================================================
# Anomaly Evaluation Metrics
# ============================================================

def calculate_anomaly_summary(
    dataframe: pd.DataFrame,
    anomaly_column: str = "is_anomaly",
    asset_column: str = "asset",
    price_column: str = "price",
) -> Dict[str, Any]:
    """
    Calculate anomaly detection summary.

    Args:
        dataframe (pd.DataFrame): Anomaly result dataframe.
        anomaly_column (str): Boolean anomaly column.
        asset_column (str): Asset column.
        price_column (str): Price column.

    Returns:
        Dict[str, Any]: Anomaly summary.
    """
    if dataframe.empty:
        raise ValueError("Anomaly dataframe is empty.")

    if anomaly_column not in dataframe.columns:
        raise ValueError(f"Missing anomaly column: {anomaly_column}")

    data = dataframe.copy()

    data[anomaly_column] = data[anomaly_column].astype(bool)

    total_records = len(data)
    total_anomalies = int(data[anomaly_column].sum())

    anomaly_rate = (
        total_anomalies / total_records * 100
        if total_records > 0
        else 0.0
    )

    summary = {
        "total_records": int(total_records),
        "total_anomalies": int(total_anomalies),
        "normal_records": int(total_records - total_anomalies),
        "anomaly_rate_percent": round(float(anomaly_rate), 6),
    }

    if asset_column in data.columns:
        summary["asset_count"] = int(data[asset_column].nunique())

        asset_summary = (
            data.groupby(asset_column)
            .agg(
                total_records=(anomaly_column, "count"),
                anomaly_count=(anomaly_column, "sum"),
                average_price=(price_column, "mean")
                if price_column in data.columns
                else (anomaly_column, "count"),
            )
            .reset_index()
        )

        asset_summary["anomaly_rate_percent"] = (
            asset_summary["anomaly_count"]
            / asset_summary["total_records"]
            * 100
        )

        summary["asset_summary"] = asset_summary.to_dict(orient="records")

    return summary


def calculate_method_anomaly_summary(
    dataframe: pd.DataFrame,
    method_columns: Optional[List[str]] = None,
) -> Dict[str, int]:
    """
    Calculate anomaly count for each detection method.

    Args:
        dataframe (pd.DataFrame): Anomaly result dataframe.
        method_columns (Optional[List[str]]): Method-specific anomaly columns.

    Returns:
        Dict[str, int]: Method-wise anomaly counts.
    """
    if method_columns is None:
        method_columns = [
            "is_anomaly_iqr",
            "is_anomaly_zscore",
            "is_anomaly_isolation_forest",
        ]

    method_summary = {}

    for column in method_columns:
        if column in dataframe.columns:
            method_summary[column] = int(dataframe[column].astype(bool).sum())

    return method_summary


def calculate_anomaly_severity_summary(
    dataframe: pd.DataFrame,
    severity_column: str = "anomaly_severity",
) -> Dict[str, int]:
    """
    Calculate anomaly severity distribution.

    Args:
        dataframe (pd.DataFrame): Anomaly result dataframe.
        severity_column (str): Severity column.

    Returns:
        Dict[str, int]: Severity counts.
    """
    if severity_column not in dataframe.columns:
        return {}

    return {
        str(key): int(value)
        for key, value in dataframe[severity_column].value_counts().to_dict().items()
    }


def get_top_anomalies(
    dataframe: pd.DataFrame,
    anomaly_column: str = "is_anomaly",
    score_column: str = "anomaly_score",
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Get top anomaly records.

    Args:
        dataframe (pd.DataFrame): Anomaly result dataframe.
        anomaly_column (str): Anomaly flag column.
        score_column (str): Anomaly score column.
        top_n (int): Number of top records.

    Returns:
        pd.DataFrame: Top anomaly records.
    """
    if anomaly_column not in dataframe.columns:
        raise ValueError(f"Missing anomaly column: {anomaly_column}")

    anomalies = dataframe[dataframe[anomaly_column].astype(bool)].copy()

    if anomalies.empty:
        return anomalies

    if score_column in anomalies.columns:
        anomalies = anomalies.sort_values(score_column, ascending=True)

    return anomalies.head(top_n)


def evaluate_anomaly_dataframe(
    dataframe: pd.DataFrame,
    anomaly_column: str = "is_anomaly",
    asset_column: str = "asset",
    price_column: str = "price",
    top_n: int = 10,
) -> Dict[str, Any]:
    """
    Evaluate anomaly detection dataframe.

    Args:
        dataframe (pd.DataFrame): Anomaly result dataframe.
        anomaly_column (str): Anomaly flag column.
        asset_column (str): Asset column.
        price_column (str): Price column.
        top_n (int): Number of top anomalies.

    Returns:
        Dict[str, Any]: Anomaly evaluation dictionary.
    """
    summary = calculate_anomaly_summary(
        dataframe=dataframe,
        anomaly_column=anomaly_column,
        asset_column=asset_column,
        price_column=price_column,
    )

    method_summary = calculate_method_anomaly_summary(dataframe)

    severity_summary = calculate_anomaly_severity_summary(dataframe)

    top_anomalies = get_top_anomalies(
        dataframe=dataframe,
        anomaly_column=anomaly_column,
        top_n=top_n,
    )

    return {
        "summary": summary,
        "method_summary": method_summary,
        "severity_summary": severity_summary,
        "top_anomalies": top_anomalies.to_dict(orient="records"),
    }


# ============================================================
# Model Comparison Helpers
# ============================================================

def compare_forecasting_models(
    model_metrics: Dict[str, Dict[str, Any]],
    selection_metric: str = "rmse",
    lower_is_better: bool = True,
) -> pd.DataFrame:
    """
    Compare forecasting models using provided metrics.

    Args:
        model_metrics (Dict[str, Dict[str, Any]]): Model metrics dictionary.
        selection_metric (str): Metric used for ranking.
        lower_is_better (bool): Whether lower metric value is better.

    Returns:
        pd.DataFrame: Model comparison dataframe.
    """
    rows = []

    for model_name, metrics in model_metrics.items():
        row = {"model_name": model_name}
        row.update(metrics)
        rows.append(row)

    comparison_dataframe = pd.DataFrame(rows)

    if (
        not comparison_dataframe.empty
        and selection_metric in comparison_dataframe.columns
    ):
        comparison_dataframe = comparison_dataframe.sort_values(
            selection_metric,
            ascending=lower_is_better,
        ).reset_index(drop=True)

        comparison_dataframe["rank"] = range(1, len(comparison_dataframe) + 1)

    return comparison_dataframe


def select_best_forecasting_model(
    comparison_dataframe: pd.DataFrame,
    selection_metric: str = "rmse",
    lower_is_better: bool = True,
) -> Dict[str, Any]:
    """
    Select best forecasting model from comparison dataframe.

    Args:
        comparison_dataframe (pd.DataFrame): Model comparison dataframe.
        selection_metric (str): Metric used for selection.
        lower_is_better (bool): Whether lower metric value is better.

    Returns:
        Dict[str, Any]: Best model information.
    """
    if comparison_dataframe.empty:
        return {
            "best_model": None,
            "selection_metric": selection_metric,
            "metric_value": None,
        }

    if selection_metric not in comparison_dataframe.columns:
        return {
            "best_model": None,
            "selection_metric": selection_metric,
            "metric_value": None,
        }

    sorted_dataframe = comparison_dataframe.dropna(
        subset=[selection_metric]
    ).sort_values(
        selection_metric,
        ascending=lower_is_better,
    )

    if sorted_dataframe.empty:
        return {
            "best_model": None,
            "selection_metric": selection_metric,
            "metric_value": None,
        }

    best_row = sorted_dataframe.iloc[0]

    return {
        "best_model": best_row.get("model_name"),
        "selection_metric": selection_metric,
        "metric_value": float(best_row.get(selection_metric)),
    }


# ============================================================
# Save Helpers
# ============================================================

def save_json(
    data: Dict[str, Any],
    file_path: Path,
) -> None:
    """
    Save dictionary as JSON.

    Args:
        data (Dict[str, Any]): Data dictionary.
        file_path (Path): Output path.
    """
    ensure_evaluation_directories()
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, default=str)


def save_dataframe(
    dataframe: pd.DataFrame,
    file_path: Path,
) -> None:
    """
    Save dataframe as CSV.

    Args:
        dataframe (pd.DataFrame): Dataframe to save.
        file_path (Path): Output path.
    """
    ensure_evaluation_directories()
    file_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(file_path, index=False)


def save_forecast_metrics(
    metrics: Dict[str, Any],
    json_path: Path = FORECAST_METRICS_PATH,
    csv_path: Path = FORECAST_METRICS_CSV_PATH,
) -> None:
    """
    Save forecast metrics as JSON and CSV.

    Args:
        metrics (Dict[str, Any]): Forecast metrics.
        json_path (Path): JSON output path.
        csv_path (Path): CSV output path.
    """
    save_json(metrics, json_path)

    overall_metrics = metrics.get("overall_metrics", {})

    if overall_metrics:
        metrics_dataframe = pd.DataFrame([overall_metrics])
        save_dataframe(metrics_dataframe, csv_path)


def save_anomaly_metrics(
    metrics: Dict[str, Any],
    json_path: Path = ANOMALY_METRICS_PATH,
    csv_path: Path = ANOMALY_METRICS_CSV_PATH,
) -> None:
    """
    Save anomaly metrics as JSON and CSV.

    Args:
        metrics (Dict[str, Any]): Anomaly metrics.
        json_path (Path): JSON output path.
        csv_path (Path): CSV output path.
    """
    save_json(metrics, json_path)

    summary = metrics.get("summary", {})

    if summary:
        summary_dataframe = pd.DataFrame([summary])
        save_dataframe(summary_dataframe, csv_path)


# ============================================================
# Script Usage Example
# ============================================================

if __name__ == "__main__":
    ensure_evaluation_directories()

    example_actual = [100, 120, 130, 150, 170]
    example_predicted = [98, 125, 128, 148, 175]

    forecast_metrics = calculate_forecasting_metrics(
        y_true=example_actual,
        y_pred=example_predicted,
    )

    print("Example Forecasting Metrics:")
    print(forecast_metrics)

    example_anomaly_data = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=5),
            "asset": ["Stock A", "Stock A", "Stock B", "Stock B", "Bond A"],
            "price": [100, 500, 120, 125, 90],
            "is_anomaly": [False, True, False, False, True],
            "is_anomaly_iqr": [False, True, False, False, False],
            "is_anomaly_zscore": [False, False, False, False, True],
            "is_anomaly_isolation_forest": [False, True, False, False, True],
            "anomaly_score": [0.1, -0.3, 0.2, 0.15, -0.2],
            "anomaly_severity": ["normal", "medium", "normal", "normal", "high"],
        }
    )

    anomaly_metrics = evaluate_anomaly_dataframe(example_anomaly_data)

    print("\nExample Anomaly Metrics:")
    print(anomaly_metrics)