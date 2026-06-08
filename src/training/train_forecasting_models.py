"""
Train forecasting models for the Financial Portfolio Forecasting
and Anomaly Detection project.

This script trains and compares multiple forecasting models:

1. ARIMA
2. XGBoost
3. Prophet
4. LSTM

Dataset expected columns:
- date
- asset
- price

Raw dataset expected path:
data/raw/financial_portfolio_data.csv

Run from project root:

    python src/training/train_forecasting_models.py
"""

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ============================================================
# Add Project Root to Python Path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ============================================================
# Project Imports
# ============================================================

from src.models.arima_model import ARIMAForecaster
from src.models.xgboost_model import XGBoostForecaster
from src.models.prophet_model import ProphetForecaster
from src.models.lstm_model import LSTMForecaster
from src.monitoring.logger import (
    get_training_logger,
    log_section,
    log_step_end,
    log_step_error,
    log_step_start,
)
from src.monitoring.performance_tracker import (
    get_performance_summary,
    save_performance_reports,
    track_step,
)
from src.utils.paths import create_required_directories


# ============================================================
# Paths
# ============================================================

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "financial_portfolio_data.csv"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"

MODEL_COMPARISON_CSV_PATH = TABLES_DIR / "forecasting_model_comparison.csv"
MODEL_COMPARISON_JSON_PATH = TABLES_DIR / "forecasting_model_comparison.json"
TRAINING_SUMMARY_PATH = REPORTS_DIR / "forecasting_training_summary.md"


# ============================================================
# Logger
# ============================================================

logger = get_training_logger()


# ============================================================
# Helper Functions
# ============================================================

def ensure_training_directories() -> None:
    """
    Create all folders needed for training outputs.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def load_financial_data(file_path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load financial portfolio dataset.

    Args:
        file_path (Path): CSV file path.

    Returns:
        pd.DataFrame: Loaded dataframe.

    Raises:
        FileNotFoundError: If dataset file does not exist.
        ValueError: If required columns are missing.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {file_path}\n"
            "Please place your CSV file at data/raw/financial_portfolio_data.csv"
        )

    data = pd.read_csv(file_path)

    data.columns = (
        data.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    required_columns = ["date", "asset", "price"]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            "Expected columns: date, asset, price"
        )

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["price"] = pd.to_numeric(data["price"], errors="coerce")

    data = data.dropna(subset=["date", "asset", "price"])
    data = data[data["price"] >= 0]
    data = data.sort_values(["asset", "date"]).reset_index(drop=True)

    if data.empty:
        raise ValueError("Dataset is empty after cleaning.")

    logger.info("Financial data loaded successfully with shape: %s", data.shape)

    return data


def get_dataset_summary(dataframe: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate dataset summary.

    Args:
        dataframe (pd.DataFrame): Input dataframe.

    Returns:
        Dict[str, Any]: Dataset summary.
    """
    return {
        "total_records": int(len(dataframe)),
        "total_assets": int(dataframe["asset"].nunique()),
        "assets": sorted(dataframe["asset"].unique().tolist()),
        "start_date": str(dataframe["date"].min()),
        "end_date": str(dataframe["date"].max()),
        "average_price": round(float(dataframe["price"].mean()), 4),
        "minimum_price": round(float(dataframe["price"].min()), 4),
        "maximum_price": round(float(dataframe["price"].max()), 4),
    }


def safe_get_metric(
    metrics: Dict[str, Any],
    metric_name: str,
    default: Optional[float] = None,
) -> Optional[float]:
    """
    Safely get a metric value from a metrics dictionary.

    Args:
        metrics (Dict[str, Any]): Metrics dictionary.
        metric_name (str): Metric name.
        default (Optional[float]): Default value.

    Returns:
        Optional[float]: Metric value.
    """
    value = metrics.get(metric_name, default)

    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def average_asset_metrics(asset_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """
    Average asset-wise metrics into one model-level metric dictionary.

    Args:
        asset_metrics (Dict[str, Dict[str, Any]]): Asset-wise metrics.

    Returns:
        Dict[str, float]: Averaged metrics.
    """
    if not asset_metrics:
        return {
            "mae": None,
            "mse": None,
            "rmse": None,
            "mape": None,
            "r2_score": None,
        }

    metric_names = ["mae", "mse", "rmse", "mape", "r2_score"]
    averaged_metrics = {}

    for metric_name in metric_names:
        values = []

        for metrics in asset_metrics.values():
            metric_value = safe_get_metric(metrics, metric_name)

            if metric_value is not None:
                values.append(metric_value)

        averaged_metrics[metric_name] = (
            round(sum(values) / len(values), 6)
            if values
            else None
        )

    return averaged_metrics


def build_model_result_row(
    model_name: str,
    status: str,
    metrics: Optional[Dict[str, Any]] = None,
    trained_assets: Optional[List[str]] = None,
    forecast_rows: int = 0,
    error_message: Optional[str] = None,
    training_time_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Build one row for model comparison.

    Args:
        model_name (str): Model name.
        status (str): success or failed.
        metrics (Optional[Dict[str, Any]]): Model metrics.
        trained_assets (Optional[List[str]]): Assets trained.
        forecast_rows (int): Number of forecast rows.
        error_message (Optional[str]): Error message.
        training_time_seconds (Optional[float]): Training duration.

    Returns:
        Dict[str, Any]: Model comparison row.
    """
    metrics = metrics or {}

    return {
        "model_name": model_name,
        "status": status,
        "mae": metrics.get("mae"),
        "mse": metrics.get("mse"),
        "rmse": metrics.get("rmse"),
        "mape": metrics.get("mape"),
        "r2_score": metrics.get("r2_score"),
        "trained_asset_count": len(trained_assets or []),
        "trained_assets": ", ".join(trained_assets or []),
        "forecast_rows": forecast_rows,
        "training_time_seconds": training_time_seconds,
        "error_message": error_message,
    }


def save_model_comparison(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Save model comparison as CSV and JSON.

    Args:
        results (List[Dict[str, Any]]): Model comparison rows.

    Returns:
        pd.DataFrame: Comparison dataframe.
    """
    comparison_dataframe = pd.DataFrame(results)

    comparison_dataframe.to_csv(MODEL_COMPARISON_CSV_PATH, index=False)

    with MODEL_COMPARISON_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, default=str)

    logger.info("Model comparison CSV saved at: %s", MODEL_COMPARISON_CSV_PATH)
    logger.info("Model comparison JSON saved at: %s", MODEL_COMPARISON_JSON_PATH)

    return comparison_dataframe


def get_best_model(comparison_dataframe: pd.DataFrame) -> Dict[str, Any]:
    """
    Select best model based on RMSE.

    Lower RMSE is better.

    Args:
        comparison_dataframe (pd.DataFrame): Model comparison dataframe.

    Returns:
        Dict[str, Any]: Best model information.
    """
    successful_models = comparison_dataframe[
        comparison_dataframe["status"] == "success"
    ].copy()

    successful_models = successful_models.dropna(subset=["rmse"])

    if successful_models.empty:
        return {
            "best_model": None,
            "best_metric": "rmse",
            "best_metric_value": None,
        }

    best_row = successful_models.sort_values("rmse", ascending=True).iloc[0]

    return {
        "best_model": best_row["model_name"],
        "best_metric": "rmse",
        "best_metric_value": float(best_row["rmse"]),
    }


def save_training_summary(
    dataset_summary: Dict[str, Any],
    comparison_dataframe: pd.DataFrame,
    best_model_info: Dict[str, Any],
    performance_summary: Dict[str, Any],
) -> None:
    """
    Save training summary as Markdown.

    Args:
        dataset_summary (Dict[str, Any]): Dataset summary.
        comparison_dataframe (pd.DataFrame): Model comparison table.
        best_model_info (Dict[str, Any]): Best model info.
        performance_summary (Dict[str, Any]): Performance tracking summary.
    """
    lines = [
        "# Forecasting Training Summary",
        "",
        "## Project",
        "",
        "Financial Portfolio Forecasting & Anomaly Detection",
        "",
        "## Training Run Information",
        "",
        f"- Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Dataset Path: `{DATA_PATH}`",
        "",
        "## Dataset Summary",
        "",
        f"- Total Records: {dataset_summary.get('total_records')}",
        f"- Total Assets: {dataset_summary.get('total_assets')}",
        f"- Assets: {', '.join(dataset_summary.get('assets', []))}",
        f"- Start Date: {dataset_summary.get('start_date')}",
        f"- End Date: {dataset_summary.get('end_date')}",
        f"- Average Price: {dataset_summary.get('average_price')}",
        f"- Minimum Price: {dataset_summary.get('minimum_price')}",
        f"- Maximum Price: {dataset_summary.get('maximum_price')}",
        "",
        "## Model Comparison",
        "",
        "| Model | Status | MAE | RMSE | MAPE | R2 Score | Assets | Forecast Rows |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in comparison_dataframe.iterrows():
        lines.append(
            "| {model} | {status} | {mae} | {rmse} | {mape} | {r2} | {assets} | {forecast_rows} |".format(
                model=row.get("model_name"),
                status=row.get("status"),
                mae=row.get("mae"),
                rmse=row.get("rmse"),
                mape=row.get("mape"),
                r2=row.get("r2_score"),
                assets=row.get("trained_asset_count"),
                forecast_rows=row.get("forecast_rows"),
            )
        )

    lines.extend(
        [
            "",
            "## Best Model",
            "",
            f"- Best Model: {best_model_info.get('best_model')}",
            f"- Selection Metric: {best_model_info.get('best_metric')}",
            f"- Best Metric Value: {best_model_info.get('best_metric_value')}",
            "",
            "## Output Files",
            "",
            f"- Model Comparison CSV: `{MODEL_COMPARISON_CSV_PATH}`",
            f"- Model Comparison JSON: `{MODEL_COMPARISON_JSON_PATH}`",
            f"- ARIMA Forecasts: `outputs/arima_forecast_results.csv`",
            f"- XGBoost Forecasts: `outputs/xgboost_forecast_results.csv`",
            f"- Prophet Forecasts: `outputs/prophet_forecast_results.csv`",
            f"- LSTM Forecasts: `outputs/lstm_forecast_results.csv`",
            "",
            "## Performance Summary",
            "",
            f"- Total Records Tracked: {performance_summary.get('total_records')}",
            f"- Total Execution Time: {performance_summary.get('total_execution_time_seconds')} seconds",
            f"- Successful Steps: {performance_summary.get('successful_steps')}",
            f"- Failed Steps: {performance_summary.get('failed_steps')}",
            "",
            "## Interpretation",
            "",
            "- ARIMA is a classical statistical forecasting model.",
            "- XGBoost uses engineered lag, rolling, and calendar features.",
            "- Prophet is useful for trend and seasonality forecasting.",
            "- LSTM is a deep learning sequence model.",
            "- The model with the lowest RMSE is selected as the best forecasting model.",
        ]
    )

    TRAINING_SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")

    logger.info("Training summary report saved at: %s", TRAINING_SUMMARY_PATH)


# ============================================================
# Forecasting Model Trainer
# ============================================================

class ForecastingModelTrainer:
    """
    Train and compare forecasting models.
    """

    def __init__(
        self,
        forecast_steps: int = 30,
        run_arima: bool = True,
        run_xgboost: bool = True,
        run_prophet: bool = True,
        run_lstm: bool = True,
        lstm_epochs: int = 10,
    ) -> None:
        """
        Initialize ForecastingModelTrainer.

        Args:
            forecast_steps (int): Forecast horizon.
            run_arima (bool): Whether to train ARIMA.
            run_xgboost (bool): Whether to train XGBoost.
            run_prophet (bool): Whether to train Prophet.
            run_lstm (bool): Whether to train LSTM.
            lstm_epochs (int): LSTM epochs.
        """
        self.forecast_steps = forecast_steps
        self.run_arima = run_arima
        self.run_xgboost = run_xgboost
        self.run_prophet = run_prophet
        self.run_lstm = run_lstm
        self.lstm_epochs = lstm_epochs

        self.results: List[Dict[str, Any]] = []
        self.dataset_summary: Dict[str, Any] = {}

    def train_arima(self, dataframe: pd.DataFrame) -> None:
        """
        Train ARIMA model and append results.
        """
        if not self.run_arima:
            return

        log_step_start(logger, "Train ARIMA model")

        try:
            with track_step("train_arima_model", category="forecasting_training"):
                forecaster = ARIMAForecaster(
                    order=(5, 1, 0),
                    date_column="date",
                    asset_column="asset",
                    target_column="price",
                    frequency="D",
                    test_size=0.2,
                )

                summary = forecaster.train_forecast_save(
                    dataframe=dataframe,
                    forecast_steps=self.forecast_steps,
                )

            averaged_metrics = average_asset_metrics(forecaster.metrics)

            row = build_model_result_row(
                model_name="ARIMA",
                status="success",
                metrics=averaged_metrics,
                trained_assets=summary.get("trained_assets", []),
                forecast_rows=summary.get("forecast_rows", 0),
            )

            self.results.append(row)

            log_step_end(logger, "Train ARIMA model")

        except Exception as error:
            log_step_error(logger, "Train ARIMA model", error)

            self.results.append(
                build_model_result_row(
                    model_name="ARIMA",
                    status="failed",
                    error_message=str(error),
                )
            )

    def train_xgboost(self, dataframe: pd.DataFrame) -> None:
        """
        Train XGBoost model and append results.
        """
        if not self.run_xgboost:
            return

        log_step_start(logger, "Train XGBoost model")

        try:
            with track_step("train_xgboost_model", category="forecasting_training"):
                forecaster = XGBoostForecaster(
                    date_column="date",
                    asset_column="asset",
                    target_column="price",
                    test_size=0.2,
                    lag_periods=[1, 3, 7, 14],
                    rolling_windows=[7, 14, 30],
                    model_params={
                        "n_estimators": 300,
                        "learning_rate": 0.05,
                        "max_depth": 5,
                        "subsample": 0.9,
                        "colsample_bytree": 0.9,
                        "objective": "reg:squarederror",
                        "random_state": 42,
                    },
                )

                summary = forecaster.train_forecast_save(
                    dataframe=dataframe,
                    forecast_steps=self.forecast_steps,
                )

            row = build_model_result_row(
                model_name="XGBoost",
                status="success",
                metrics=forecaster.metrics,
                trained_assets=sorted(dataframe["asset"].unique().tolist()),
                forecast_rows=summary.get("forecast_rows", 0),
            )

            self.results.append(row)

            log_step_end(logger, "Train XGBoost model")

        except Exception as error:
            log_step_error(logger, "Train XGBoost model", error)

            self.results.append(
                build_model_result_row(
                    model_name="XGBoost",
                    status="failed",
                    error_message=str(error),
                )
            )

    def train_prophet(self, dataframe: pd.DataFrame) -> None:
        """
        Train Prophet model and append results.
        """
        if not self.run_prophet:
            return

        log_step_start(logger, "Train Prophet model")

        try:
            with track_step("train_prophet_model", category="forecasting_training"):
                forecaster = ProphetForecaster(
                    date_column="date",
                    asset_column="asset",
                    target_column="price",
                    test_size=0.2,
                    frequency="D",
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    seasonality_mode="additive",
                    changepoint_prior_scale=0.05,
                )

                summary = forecaster.train_forecast_save(
                    dataframe=dataframe,
                    forecast_periods=self.forecast_steps,
                )

            averaged_metrics = average_asset_metrics(forecaster.metrics)

            row = build_model_result_row(
                model_name="Prophet",
                status="success",
                metrics=averaged_metrics,
                trained_assets=summary.get("trained_assets", []),
                forecast_rows=summary.get("forecast_rows", 0),
            )

            self.results.append(row)

            log_step_end(logger, "Train Prophet model")

        except Exception as error:
            log_step_error(logger, "Train Prophet model", error)

            self.results.append(
                build_model_result_row(
                    model_name="Prophet",
                    status="failed",
                    error_message=str(error),
                )
            )

    def train_lstm(self, dataframe: pd.DataFrame) -> None:
        """
        Train LSTM model and append results.
        """
        if not self.run_lstm:
            return

        log_step_start(logger, "Train LSTM model")

        try:
            with track_step("train_lstm_model", category="forecasting_training"):
                forecaster = LSTMForecaster(
                    date_column="date",
                    asset_column="asset",
                    target_column="price",
                    frequency="D",
                    sequence_length=10,
                    test_size=0.2,
                    epochs=self.lstm_epochs,
                    batch_size=16,
                    random_state=42,
                )

                summary = forecaster.train_forecast_save(
                    dataframe=dataframe,
                    forecast_steps=self.forecast_steps,
                )

            averaged_metrics = average_asset_metrics(forecaster.metrics)

            row = build_model_result_row(
                model_name="LSTM",
                status="success",
                metrics=averaged_metrics,
                trained_assets=summary.get("trained_assets", []),
                forecast_rows=summary.get("forecast_rows", 0),
            )

            self.results.append(row)

            log_step_end(logger, "Train LSTM model")

        except Exception as error:
            log_step_error(logger, "Train LSTM model", error)

            self.results.append(
                build_model_result_row(
                    model_name="LSTM",
                    status="failed",
                    error_message=str(error),
                )
            )

    def run(self) -> Dict[str, Any]:
        """
        Run full forecasting model training workflow.

        Returns:
            Dict[str, Any]: Training workflow summary.
        """
        log_section(logger, "Starting Forecasting Model Training")

        ensure_training_directories()
        create_required_directories()

        try:
            with track_step("load_training_data", category="forecasting_training"):
                dataframe = load_financial_data(DATA_PATH)

            self.dataset_summary = get_dataset_summary(dataframe)

            logger.info("Dataset summary: %s", self.dataset_summary)

            self.train_arima(dataframe)
            self.train_xgboost(dataframe)
            self.train_prophet(dataframe)
            self.train_lstm(dataframe)

            comparison_dataframe = save_model_comparison(self.results)

            best_model_info = get_best_model(comparison_dataframe)

            save_performance_reports()
            performance_summary = get_performance_summary()

            save_training_summary(
                dataset_summary=self.dataset_summary,
                comparison_dataframe=comparison_dataframe,
                best_model_info=best_model_info,
                performance_summary=performance_summary,
            )

            final_summary = {
                "status": "success",
                "dataset_summary": self.dataset_summary,
                "model_results": self.results,
                "best_model": best_model_info,
                "comparison_csv": str(MODEL_COMPARISON_CSV_PATH),
                "comparison_json": str(MODEL_COMPARISON_JSON_PATH),
                "training_summary_report": str(TRAINING_SUMMARY_PATH),
                "performance_summary": performance_summary,
            }

            log_section(logger, "Forecasting Model Training Completed")

            return final_summary

        except Exception as error:
            logger.error(
                "Forecasting model training failed: %s",
                error,
                exc_info=True,
            )

            save_performance_reports()

            return {
                "status": "failed",
                "error_type": error.__class__.__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            }


# ============================================================
# Main Function
# ============================================================

def main() -> None:
    """
    Main entry point.
    """
    trainer = ForecastingModelTrainer(
        forecast_steps=30,
        run_arima=True,
        run_xgboost=True,
        run_prophet=True,
        run_lstm=True,
        lstm_epochs=10,
    )

    summary = trainer.run()

    print("\n" + "=" * 80)
    print("FORECASTING MODEL TRAINING SUMMARY")
    print("=" * 80)

    print("Status:", summary.get("status"))

    if summary.get("status") == "success":
        print("\nDataset Summary:")
        for key, value in summary.get("dataset_summary", {}).items():
            print(f"- {key}: {value}")

        print("\nBest Model:")
        for key, value in summary.get("best_model", {}).items():
            print(f"- {key}: {value}")

        print("\nSaved Files:")
        print("- reports/tables/forecasting_model_comparison.csv")
        print("- reports/tables/forecasting_model_comparison.json")
        print("- reports/forecasting_training_summary.md")
        print("- reports/tables/performance_metrics.json")
        print("- outputs/arima_forecast_results.csv")
        print("- outputs/xgboost_forecast_results.csv")
        print("- outputs/prophet_forecast_results.csv")
        print("- outputs/lstm_forecast_results.csv")

    else:
        print("\nTraining failed.")
        print("Error Type:", summary.get("error_type"))
        print("Error Message:", summary.get("error_message"))
        print("\nCheck logs/training.log or logs/app.log for details.")

    print("=" * 80)


# ============================================================
# Script Entry Point
# ============================================================

if __name__ == "__main__":
    main()