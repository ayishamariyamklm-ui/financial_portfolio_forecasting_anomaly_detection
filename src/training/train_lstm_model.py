"""
Train LSTM forecasting model for the Financial Portfolio Forecasting
and Anomaly Detection project.

This script trains only the LSTM model for asset price forecasting.

Dataset expected columns:
- date
- asset
- price

Raw dataset expected path:
data/raw/financial_portfolio_data.csv

Run from project root:

    python src/training/train_lstm_model.py
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

from src.models.lstm_model import LSTMForecaster
from src.monitoring.logger import (
    get_training_logger,
    log_dataframe_info,
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

LSTM_TRAINING_SUMMARY_JSON_PATH = TABLES_DIR / "lstm_training_summary.json"
LSTM_TRAINING_SUMMARY_MD_PATH = REPORTS_DIR / "lstm_training_summary.md"
LSTM_METRICS_COMPARISON_PATH = TABLES_DIR / "lstm_asset_metrics_comparison.csv"


# ============================================================
# Logger
# ============================================================

logger = get_training_logger()


# ============================================================
# Helper Functions
# ============================================================

def ensure_training_directories() -> None:
    """
    Create all folders required for LSTM training outputs.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def load_financial_data(file_path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load and clean the financial portfolio dataset.

    Args:
        file_path (Path): CSV file path.

    Returns:
        pd.DataFrame: Cleaned dataframe.

    Raises:
        FileNotFoundError: If dataset is missing.
        ValueError: If required columns are missing or data is empty.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {file_path}\n"
            "Please place your CSV file at data/raw/financial_portfolio_data.csv"
        )

    dataframe = pd.read_csv(file_path)

    dataframe.columns = (
        dataframe.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    required_columns = ["date", "asset", "price"]

    missing_columns = [
        column for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            "Expected columns are: date, asset, price."
        )

    dataframe["date"] = pd.to_datetime(dataframe["date"], errors="coerce")
    dataframe["price"] = pd.to_numeric(dataframe["price"], errors="coerce")

    dataframe = dataframe.dropna(subset=["date", "asset", "price"])
    dataframe = dataframe[dataframe["price"] >= 0]
    dataframe = dataframe.sort_values(["asset", "date"]).reset_index(drop=True)

    if dataframe.empty:
        raise ValueError("Dataset is empty after cleaning.")

    logger.info("Dataset loaded successfully with shape: %s", dataframe.shape)

    return dataframe


def get_dataset_summary(dataframe: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary for the dataset.

    Args:
        dataframe (pd.DataFrame): Input dataframe.

    Returns:
        Dict[str, Any]: Dataset summary.
    """
    asset_summary = (
        dataframe.groupby("asset")
        .agg(
            record_count=("price", "count"),
            start_date=("date", "min"),
            end_date=("date", "max"),
            average_price=("price", "mean"),
            min_price=("price", "min"),
            max_price=("price", "max"),
            price_std=("price", "std"),
        )
        .reset_index()
    )

    return {
        "total_records": int(len(dataframe)),
        "total_assets": int(dataframe["asset"].nunique()),
        "assets": sorted(dataframe["asset"].unique().tolist()),
        "start_date": str(dataframe["date"].min()),
        "end_date": str(dataframe["date"].max()),
        "average_price": round(float(dataframe["price"].mean()), 4),
        "minimum_price": round(float(dataframe["price"].min()), 4),
        "maximum_price": round(float(dataframe["price"].max()), 4),
        "asset_summary": asset_summary.to_dict(orient="records"),
    }


def metrics_to_dataframe(metrics: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert asset-wise LSTM metrics into dataframe.

    Args:
        metrics (Dict[str, Dict[str, Any]]): Asset-wise metrics.

    Returns:
        pd.DataFrame: Metrics dataframe.
    """
    if not metrics:
        return pd.DataFrame()

    metrics_dataframe = (
        pd.DataFrame(metrics)
        .T
        .reset_index()
        .rename(columns={"index": "asset"})
    )

    return metrics_dataframe


def average_lstm_metrics(metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """
    Calculate average metrics across assets.

    Args:
        metrics (Dict[str, Dict[str, Any]]): Asset-wise metrics.

    Returns:
        Dict[str, Optional[float]]: Average metrics.
    """
    if not metrics:
        return {
            "mae": None,
            "mse": None,
            "rmse": None,
            "mape": None,
            "r2_score": None,
        }

    metric_names = ["mae", "mse", "rmse", "mape", "r2_score"]
    average_metrics: Dict[str, Optional[float]] = {}

    for metric_name in metric_names:
        values = []

        for asset_metrics in metrics.values():
            metric_value = asset_metrics.get(metric_name)

            if metric_value is not None:
                try:
                    values.append(float(metric_value))
                except (TypeError, ValueError):
                    continue

        average_metrics[metric_name] = (
            round(sum(values) / len(values), 6)
            if values
            else None
        )

    return average_metrics


def save_json_summary(summary: Dict[str, Any]) -> None:
    """
    Save LSTM training summary as JSON.

    Args:
        summary (Dict[str, Any]): Training summary.
    """
    with LSTM_TRAINING_SUMMARY_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, default=str)

    logger.info(
        "LSTM training summary JSON saved at: %s",
        LSTM_TRAINING_SUMMARY_JSON_PATH,
    )


def save_metrics_comparison(metrics: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Save LSTM asset-wise metrics as CSV.

    Args:
        metrics (Dict[str, Dict[str, Any]]): Asset-wise metrics.

    Returns:
        pd.DataFrame: Metrics dataframe.
    """
    metrics_dataframe = metrics_to_dataframe(metrics)

    if not metrics_dataframe.empty:
        metrics_dataframe.to_csv(LSTM_METRICS_COMPARISON_PATH, index=False)
        logger.info(
            "LSTM asset metrics comparison saved at: %s",
            LSTM_METRICS_COMPARISON_PATH,
        )

    return metrics_dataframe


def save_markdown_summary(
    summary: Dict[str, Any],
    metrics_dataframe: pd.DataFrame,
) -> None:
    """
    Save LSTM training summary as Markdown.

    Args:
        summary (Dict[str, Any]): Training summary.
        metrics_dataframe (pd.DataFrame): Asset-wise metrics dataframe.
    """
    dataset_summary = summary.get("dataset_summary", {})
    model_summary = summary.get("model_summary", {})
    average_metrics = summary.get("average_metrics", {})
    performance_summary = summary.get("performance_summary", {})

    lines = [
        "# LSTM Training Summary",
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
        "## LSTM Configuration",
        "",
        f"- Sequence Length: {model_summary.get('sequence_length')}",
        f"- Forecast Steps: {model_summary.get('forecast_steps')}",
        f"- Epochs: {model_summary.get('epochs')}",
        f"- Batch Size: {model_summary.get('batch_size')}",
        f"- Test Size: {model_summary.get('test_size')}",
        f"- Frequency: {model_summary.get('frequency')}",
        "",
        "## Average LSTM Metrics",
        "",
        f"- MAE: {average_metrics.get('mae')}",
        f"- MSE: {average_metrics.get('mse')}",
        f"- RMSE: {average_metrics.get('rmse')}",
        f"- MAPE: {average_metrics.get('mape')}",
        f"- R2 Score: {average_metrics.get('r2_score')}",
        "",
        "## Asset-wise Metrics",
        "",
        "| Asset | MAE | RMSE | MAPE | R2 Score | Train Records | Test Records |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    if not metrics_dataframe.empty:
        for _, row in metrics_dataframe.iterrows():
            lines.append(
                "| {asset} | {mae} | {rmse} | {mape} | {r2} | {train} | {test} |".format(
                    asset=row.get("asset"),
                    mae=row.get("mae"),
                    rmse=row.get("rmse"),
                    mape=row.get("mape"),
                    r2=row.get("r2_score"),
                    train=row.get("train_records"),
                    test=row.get("test_records"),
                )
            )
    else:
        lines.append("| No metrics available | - | - | - | - | - | - |")

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- LSTM Models Directory: `models/lstm_models/`",
            "- LSTM Scalers: `models/lstm_scalers.pkl`",
            "- LSTM Metrics JSON: `reports/tables/lstm_metrics.json`",
            "- LSTM Metrics CSV: `reports/tables/lstm_metrics.csv`",
            "- LSTM Test Predictions: `outputs/lstm_test_predictions.csv`",
            "- LSTM Forecast Results: `outputs/lstm_forecast_results.csv`",
            "- LSTM Asset Metrics Comparison: `reports/tables/lstm_asset_metrics_comparison.csv`",
            "- LSTM Training Summary JSON: `reports/tables/lstm_training_summary.json`",
            "- LSTM Training Summary Markdown: `reports/lstm_training_summary.md`",
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
            "- LSTM is a deep learning sequence model suitable for time series forecasting.",
            "- It learns patterns from previous price sequences to predict future price values.",
            "- Lower RMSE and MAE indicate better forecasting accuracy.",
            "- MAPE shows average percentage error.",
            "- R2 score shows how much variation in price is explained by the model.",
            "",
            "## Notes",
            "",
            "- For faster training on a beginner system, use fewer epochs such as 5 or 10.",
            "- For better accuracy, increase epochs only after confirming the pipeline works.",
            "- Python 3.11 is recommended for TensorFlow compatibility.",
        ]
    )

    LSTM_TRAINING_SUMMARY_MD_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    logger.info(
        "LSTM training summary Markdown saved at: %s",
        LSTM_TRAINING_SUMMARY_MD_PATH,
    )


# ============================================================
# LSTM Training Runner
# ============================================================

class LSTMTrainingRunner:
    """
    Train and save LSTM forecasting model.
    """

    def __init__(
        self,
        forecast_steps: int = 30,
        sequence_length: int = 10,
        test_size: float = 0.2,
        epochs: int = 10,
        batch_size: int = 16,
        frequency: str = "D",
        selected_assets: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize LSTMTrainingRunner.

        Args:
            forecast_steps (int): Number of future days/periods to forecast.
            sequence_length (int): Previous records used for prediction.
            test_size (float): Test split ratio.
            epochs (int): Number of training epochs.
            batch_size (int): Training batch size.
            frequency (str): Time frequency.
            selected_assets (Optional[List[str]]): Optional selected assets.
        """
        self.forecast_steps = forecast_steps
        self.sequence_length = sequence_length
        self.test_size = test_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.frequency = frequency
        self.selected_assets = selected_assets

        self.forecaster = LSTMForecaster(
            date_column="date",
            asset_column="asset",
            target_column="price",
            frequency=self.frequency,
            sequence_length=self.sequence_length,
            test_size=self.test_size,
            epochs=self.epochs,
            batch_size=self.batch_size,
            random_state=42,
        )

    def run(self) -> Dict[str, Any]:
        """
        Run complete LSTM training workflow.

        Returns:
            Dict[str, Any]: Training summary.
        """
        log_section(logger, "Starting LSTM Model Training")

        try:
            ensure_training_directories()
            create_required_directories()

            with track_step("load_lstm_training_data", category="lstm_training"):
                dataframe = load_financial_data(DATA_PATH)

            log_dataframe_info(logger, dataframe, "LSTM Training Data")

            dataset_summary = get_dataset_summary(dataframe)

            logger.info("Dataset summary: %s", dataset_summary)

            with track_step("train_lstm_forecaster", category="lstm_training"):
                workflow_summary = self.forecaster.train_forecast_save(
                    dataframe=dataframe,
                    forecast_steps=self.forecast_steps,
                    assets=self.selected_assets,
                )

            average_metrics = average_lstm_metrics(self.forecaster.metrics)
            metrics_dataframe = save_metrics_comparison(self.forecaster.metrics)

            save_performance_reports()
            performance_summary = get_performance_summary()

            training_summary = {
                "status": "success",
                "dataset_summary": dataset_summary,
                "model_summary": {
                    "model_type": "LSTM",
                    "sequence_length": self.sequence_length,
                    "forecast_steps": self.forecast_steps,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "test_size": self.test_size,
                    "frequency": self.frequency,
                    "trained_assets": list(self.forecaster.models.keys()),
                    "asset_count": len(self.forecaster.models),
                    "forecast_rows": workflow_summary.get("forecast_rows"),
                },
                "average_metrics": average_metrics,
                "asset_metrics": self.forecaster.metrics,
                "output_paths": {
                    "models_dir": "models/lstm_models/",
                    "scalers": "models/lstm_scalers.pkl",
                    "metrics_json": "reports/tables/lstm_metrics.json",
                    "metrics_csv": "reports/tables/lstm_metrics.csv",
                    "predictions": "outputs/lstm_test_predictions.csv",
                    "forecasts": "outputs/lstm_forecast_results.csv",
                    "asset_metrics_comparison": str(LSTM_METRICS_COMPARISON_PATH),
                    "training_summary_json": str(LSTM_TRAINING_SUMMARY_JSON_PATH),
                    "training_summary_md": str(LSTM_TRAINING_SUMMARY_MD_PATH),
                },
                "performance_summary": performance_summary,
            }

            save_json_summary(training_summary)
            save_markdown_summary(training_summary, metrics_dataframe)

            log_section(logger, "LSTM Model Training Completed Successfully")

            return training_summary

        except Exception as error:
            log_step_error(logger, "LSTM model training", error)

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
    Main function to train LSTM model.
    """
    runner = LSTMTrainingRunner(
        forecast_steps=30,
        sequence_length=10,
        test_size=0.2,
        epochs=10,
        batch_size=16,
        frequency="D",
        selected_assets=None,
    )

    summary = runner.run()

    print("\n" + "=" * 80)
    print("LSTM MODEL TRAINING SUMMARY")
    print("=" * 80)

    print("Status:", summary.get("status"))

    if summary.get("status") == "success":
        dataset_summary = summary.get("dataset_summary", {})
        model_summary = summary.get("model_summary", {})
        average_metrics = summary.get("average_metrics", {})
        output_paths = summary.get("output_paths", {})

        print("\nDataset Summary:")
        print("- Total Records:", dataset_summary.get("total_records"))
        print("- Total Assets:", dataset_summary.get("total_assets"))
        print("- Start Date:", dataset_summary.get("start_date"))
        print("- End Date:", dataset_summary.get("end_date"))

        print("\nModel Summary:")
        print("- Model Type:", model_summary.get("model_type"))
        print("- Sequence Length:", model_summary.get("sequence_length"))
        print("- Forecast Steps:", model_summary.get("forecast_steps"))
        print("- Epochs:", model_summary.get("epochs"))
        print("- Batch Size:", model_summary.get("batch_size"))
        print("- Trained Assets:", model_summary.get("trained_assets"))
        print("- Forecast Rows:", model_summary.get("forecast_rows"))

        print("\nAverage Metrics:")
        print("- MAE:", average_metrics.get("mae"))
        print("- RMSE:", average_metrics.get("rmse"))
        print("- MAPE:", average_metrics.get("mape"))
        print("- R2 Score:", average_metrics.get("r2_score"))

        print("\nSaved Files:")
        for name, path in output_paths.items():
            print(f"- {name}: {path}")

    else:
        print("\nLSTM training failed.")
        print("Error Type:", summary.get("error_type"))
        print("Error Message:", summary.get("error_message"))
        print("\nCheck logs/training.log or logs/app.log for details.")

    print("=" * 80)


# ============================================================
# Script Entry Point
# ============================================================

if __name__ == "__main__":
    main()