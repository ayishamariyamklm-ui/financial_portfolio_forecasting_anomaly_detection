"""
Prediction and inference utilities for the Financial Portfolio Forecasting
and Anomaly Detection project.

This module supports:

1. Loading trained forecasting models
   - XGBoost
   - ARIMA
   - Prophet
   - LSTM

2. Generating future forecasts

3. Loading trained anomaly detector

4. Detecting anomalies on new data

5. Saving prediction outputs

Dataset expected columns:
- date
- asset
- price

Example usage from project root:

    python src/inference/predict.py
"""

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

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

from src.models.anomaly_detector import (
    ANOMALY_MODEL_PATH,
    load_anomaly_detector,
)
from src.models.arima_model import (
    ARIMA_MODEL_PATH,
    load_arima_forecaster,
)
from src.models.lstm_model import (
    LSTM_MODEL_DIR,
    LSTM_SCALER_PATH,
    load_lstm_forecaster,
)
from src.models.prophet_model import (
    PROPHET_MODEL_PATH,
    load_prophet_forecaster,
)
from src.models.xgboost_model import (
    XGBOOST_MODEL_PATH,
    load_xgboost_forecaster,
)
from src.monitoring.logger import (
    get_project_logger,
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

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
LOGS_DIR = PROJECT_ROOT / "logs"

INFERENCE_OUTPUT_PATH = OUTPUTS_DIR / "inference_predictions.csv"
FORECAST_OUTPUT_PATH = OUTPUTS_DIR / "inference_forecasts.csv"
ANOMALY_OUTPUT_PATH = OUTPUTS_DIR / "inference_anomaly_results.csv"
INFERENCE_SUMMARY_JSON_PATH = TABLES_DIR / "inference_summary.json"
INFERENCE_SUMMARY_MD_PATH = REPORTS_DIR / "inference_summary.md"


# ============================================================
# Logger
# ============================================================

logger = get_project_logger()


# ============================================================
# Directory Helpers
# ============================================================

def ensure_inference_directories() -> None:
    """
    Create required directories for inference outputs.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Data Loading and Cleaning
# ============================================================

def load_input_data(file_path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load input financial data.

    Args:
        file_path (Path): Input CSV path.

    Returns:
        pd.DataFrame: Loaded and cleaned dataframe.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If required columns are missing.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Input data file not found at: {file_path}\n"
            "Please place financial_portfolio_data.csv inside data/raw/."
        )

    dataframe = pd.read_csv(file_path)

    return clean_input_data(dataframe)


def clean_input_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize input dataframe.

    Args:
        dataframe (pd.DataFrame): Input dataframe.

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

    required_columns = ["date", "asset", "price"]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            "Expected columns are: date, asset, price."
        )

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["price"] = pd.to_numeric(data["price"], errors="coerce")

    data = data.dropna(subset=["date", "asset", "price"])
    data = data[data["price"] >= 0]
    data = data.sort_values(["asset", "date"]).reset_index(drop=True)

    if data.empty:
        raise ValueError("Input dataframe is empty after cleaning.")

    return data


def get_input_summary(dataframe: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate input data summary.

    Args:
        dataframe (pd.DataFrame): Input dataframe.

    Returns:
        Dict[str, Any]: Input summary.
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


# ============================================================
# Forecasting Inference
# ============================================================

def forecast_with_xgboost(
    dataframe: pd.DataFrame,
    forecast_steps: int = 30,
) -> pd.DataFrame:
    """
    Generate forecasts using saved XGBoost model.

    Args:
        dataframe (pd.DataFrame): Historical dataframe.
        forecast_steps (int): Forecast horizon.

    Returns:
        pd.DataFrame: Forecast dataframe.
    """
    if not XGBOOST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved XGBoost model not found at: {XGBOOST_MODEL_PATH}"
        )

    forecaster = load_xgboost_forecaster(XGBOOST_MODEL_PATH)

    forecast_dataframe = forecaster.forecast_all_assets(
        dataframe=dataframe,
        forecast_steps=forecast_steps,
    )

    forecast_dataframe["model_name"] = "XGBoost"

    return forecast_dataframe


def forecast_with_arima(
    forecast_steps: int = 30,
) -> pd.DataFrame:
    """
    Generate forecasts using saved ARIMA models.

    Args:
        forecast_steps (int): Forecast horizon.

    Returns:
        pd.DataFrame: Forecast dataframe.
    """
    if not ARIMA_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved ARIMA model not found at: {ARIMA_MODEL_PATH}"
        )

    forecaster = load_arima_forecaster(ARIMA_MODEL_PATH)

    forecast_dataframe = forecaster.forecast_all(
        steps=forecast_steps,
    )

    forecast_dataframe["model_name"] = "ARIMA"

    return forecast_dataframe


def forecast_with_prophet(
    forecast_steps: int = 30,
) -> pd.DataFrame:
    """
    Generate forecasts using saved Prophet models.

    Args:
        forecast_steps (int): Forecast horizon.

    Returns:
        pd.DataFrame: Forecast dataframe.
    """
    if not PROPHET_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved Prophet model not found at: {PROPHET_MODEL_PATH}"
        )

    forecaster = load_prophet_forecaster(PROPHET_MODEL_PATH)

    forecast_dataframe = forecaster.forecast_all(
        periods=forecast_steps,
    )

    forecast_dataframe["model_name"] = "Prophet"

    return forecast_dataframe


def forecast_with_lstm(
    dataframe: pd.DataFrame,
    forecast_steps: int = 30,
) -> pd.DataFrame:
    """
    Generate forecasts using saved LSTM models.

    Args:
        dataframe (pd.DataFrame): Historical dataframe.
        forecast_steps (int): Forecast horizon.

    Returns:
        pd.DataFrame: Forecast dataframe.
    """
    if not LSTM_SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Saved LSTM scaler file not found at: {LSTM_SCALER_PATH}"
        )

    if not LSTM_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Saved LSTM model directory not found at: {LSTM_MODEL_DIR}"
        )

    forecaster = load_lstm_forecaster(
        model_dir=LSTM_MODEL_DIR,
        scaler_path=LSTM_SCALER_PATH,
    )

    forecast_dataframe = forecaster.forecast_all(
        dataframe=dataframe,
        forecast_steps=forecast_steps,
    )

    forecast_dataframe["model_name"] = "LSTM"

    return forecast_dataframe


def generate_forecasts(
    dataframe: pd.DataFrame,
    model_name: str = "xgboost",
    forecast_steps: int = 30,
) -> pd.DataFrame:
    """
    Generate forecasts using selected saved model.

    Args:
        dataframe (pd.DataFrame): Historical dataframe.
        model_name (str): Model name. Options:
            - xgboost
            - arima
            - prophet
            - lstm
            - all
        forecast_steps (int): Forecast horizon.

    Returns:
        pd.DataFrame: Forecast dataframe.
    """
    model_name = model_name.lower().strip()

    forecast_frames = []

    if model_name in ["xgboost", "all"]:
        try:
            forecast_frames.append(
                forecast_with_xgboost(
                    dataframe=dataframe,
                    forecast_steps=forecast_steps,
                )
            )
        except Exception as error:
            logger.warning("XGBoost forecast skipped: %s", error)

    if model_name in ["arima", "all"]:
        try:
            forecast_frames.append(
                forecast_with_arima(
                    forecast_steps=forecast_steps,
                )
            )
        except Exception as error:
            logger.warning("ARIMA forecast skipped: %s", error)

    if model_name in ["prophet", "all"]:
        try:
            forecast_frames.append(
                forecast_with_prophet(
                    forecast_steps=forecast_steps,
                )
            )
        except Exception as error:
            logger.warning("Prophet forecast skipped: %s", error)

    if model_name in ["lstm", "all"]:
        try:
            forecast_frames.append(
                forecast_with_lstm(
                    dataframe=dataframe,
                    forecast_steps=forecast_steps,
                )
            )
        except Exception as error:
            logger.warning("LSTM forecast skipped: %s", error)

    if not forecast_frames:
        raise ValueError(
            "No forecasts were generated. "
            "Please train at least one forecasting model first."
        )

    forecast_dataframe = pd.concat(forecast_frames, ignore_index=True)

    return forecast_dataframe


# ============================================================
# Anomaly Inference
# ============================================================

def detect_anomalies_with_saved_model(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Detect anomalies using saved anomaly detector.

    Args:
        dataframe (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Anomaly result dataframe.
    """
    if not ANOMALY_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved anomaly detection model not found at: {ANOMALY_MODEL_PATH}"
        )

    detector = load_anomaly_detector(ANOMALY_MODEL_PATH)

    anomaly_results = detector.predict_new_data(dataframe)

    return anomaly_results


def summarize_forecast_results(
    forecast_dataframe: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Summarize forecast dataframe.

    Args:
        forecast_dataframe (pd.DataFrame): Forecast dataframe.

    Returns:
        Dict[str, Any]: Forecast summary.
    """
    if forecast_dataframe.empty:
        return {
            "forecast_rows": 0,
            "models_used": [],
            "assets_forecasted": [],
        }

    forecast_column = (
        "forecasted_price"
        if "forecasted_price" in forecast_dataframe.columns
        else "predicted_price"
    )

    return {
        "forecast_rows": int(len(forecast_dataframe)),
        "models_used": sorted(forecast_dataframe["model_name"].unique().tolist())
        if "model_name" in forecast_dataframe.columns
        else [],
        "assets_forecasted": sorted(forecast_dataframe["asset"].unique().tolist())
        if "asset" in forecast_dataframe.columns
        else [],
        "forecast_start_date": str(forecast_dataframe["date"].min())
        if "date" in forecast_dataframe.columns
        else None,
        "forecast_end_date": str(forecast_dataframe["date"].max())
        if "date" in forecast_dataframe.columns
        else None,
        "average_forecasted_price": round(
            float(forecast_dataframe[forecast_column].mean()),
            4,
        )
        if forecast_column in forecast_dataframe.columns
        else None,
        "minimum_forecasted_price": round(
            float(forecast_dataframe[forecast_column].min()),
            4,
        )
        if forecast_column in forecast_dataframe.columns
        else None,
        "maximum_forecasted_price": round(
            float(forecast_dataframe[forecast_column].max()),
            4,
        )
        if forecast_column in forecast_dataframe.columns
        else None,
    }


def summarize_anomaly_results(
    anomaly_dataframe: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Summarize anomaly results.

    Args:
        anomaly_dataframe (pd.DataFrame): Anomaly result dataframe.

    Returns:
        Dict[str, Any]: Anomaly summary.
    """
    if anomaly_dataframe.empty or "is_anomaly" not in anomaly_dataframe.columns:
        return {
            "total_records": 0,
            "total_anomalies": 0,
            "anomaly_rate_percent": 0.0,
        }

    anomaly_dataframe = anomaly_dataframe.copy()
    anomaly_dataframe["is_anomaly"] = anomaly_dataframe["is_anomaly"].astype(bool)

    total_records = len(anomaly_dataframe)
    total_anomalies = int(anomaly_dataframe["is_anomaly"].sum())

    anomaly_rate = (
        total_anomalies / total_records * 100
        if total_records > 0
        else 0.0
    )

    method_summary = {}

    for column in [
        "is_anomaly_iqr",
        "is_anomaly_zscore",
        "is_anomaly_isolation_forest",
    ]:
        if column in anomaly_dataframe.columns:
            method_summary[column] = int(anomaly_dataframe[column].astype(bool).sum())

    asset_summary = []

    if "asset" in anomaly_dataframe.columns:
        asset_summary_dataframe = (
            anomaly_dataframe.groupby("asset")
            .agg(
                total_records=("is_anomaly", "count"),
                anomaly_count=("is_anomaly", "sum"),
            )
            .reset_index()
        )

        asset_summary_dataframe["anomaly_rate_percent"] = (
            asset_summary_dataframe["anomaly_count"]
            / asset_summary_dataframe["total_records"]
            * 100
        )

        asset_summary = asset_summary_dataframe.to_dict(orient="records")

    return {
        "total_records": int(total_records),
        "total_anomalies": int(total_anomalies),
        "normal_records": int(total_records - total_anomalies),
        "anomaly_rate_percent": round(float(anomaly_rate), 4),
        "method_summary": method_summary,
        "asset_summary": asset_summary,
    }


# ============================================================
# Save Inference Outputs
# ============================================================

def save_forecasts(
    forecast_dataframe: pd.DataFrame,
    file_path: Path = FORECAST_OUTPUT_PATH,
) -> None:
    """
    Save forecast dataframe as CSV.

    Args:
        forecast_dataframe (pd.DataFrame): Forecast dataframe.
        file_path (Path): Output CSV path.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    forecast_dataframe.to_csv(file_path, index=False)
    logger.info("Forecast results saved at: %s", file_path)


def save_anomaly_results(
    anomaly_dataframe: pd.DataFrame,
    file_path: Path = ANOMALY_OUTPUT_PATH,
) -> None:
    """
    Save anomaly result dataframe as CSV.

    Args:
        anomaly_dataframe (pd.DataFrame): Anomaly result dataframe.
        file_path (Path): Output CSV path.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    anomaly_dataframe.to_csv(file_path, index=False)
    logger.info("Anomaly inference results saved at: %s", file_path)


def save_inference_summary(
    summary: Dict[str, Any],
    json_path: Path = INFERENCE_SUMMARY_JSON_PATH,
    markdown_path: Path = INFERENCE_SUMMARY_MD_PATH,
) -> None:
    """
    Save inference summary as JSON and Markdown.

    Args:
        summary (Dict[str, Any]): Inference summary.
        json_path (Path): JSON output path.
        markdown_path (Path): Markdown output path.
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, default=str)

    forecast_summary = summary.get("forecast_summary", {})
    anomaly_summary = summary.get("anomaly_summary", {})
    input_summary = summary.get("input_summary", {})
    performance_summary = summary.get("performance_summary", {})

    lines = [
        "# Inference Summary",
        "",
        "## Project",
        "",
        "Financial Portfolio Forecasting & Anomaly Detection",
        "",
        "## Run Information",
        "",
        f"- Generated At: {summary.get('generated_at')}",
        f"- Status: {summary.get('status')}",
        f"- Input Data Path: `{summary.get('input_data_path')}`",
        "",
        "## Input Data Summary",
        "",
        f"- Total Records: {input_summary.get('total_records')}",
        f"- Total Assets: {input_summary.get('total_assets')}",
        f"- Assets: {', '.join(input_summary.get('assets', [])) if input_summary.get('assets') else 'N/A'}",
        f"- Start Date: {input_summary.get('start_date')}",
        f"- End Date: {input_summary.get('end_date')}",
        "",
        "## Forecast Summary",
        "",
        f"- Forecast Rows: {forecast_summary.get('forecast_rows')}",
        f"- Models Used: {', '.join(forecast_summary.get('models_used', [])) if forecast_summary.get('models_used') else 'N/A'}",
        f"- Assets Forecasted: {', '.join(forecast_summary.get('assets_forecasted', [])) if forecast_summary.get('assets_forecasted') else 'N/A'}",
        f"- Forecast Start Date: {forecast_summary.get('forecast_start_date')}",
        f"- Forecast End Date: {forecast_summary.get('forecast_end_date')}",
        f"- Average Forecasted Price: {forecast_summary.get('average_forecasted_price')}",
        "",
        "## Anomaly Summary",
        "",
        f"- Total Records Checked: {anomaly_summary.get('total_records')}",
        f"- Total Anomalies: {anomaly_summary.get('total_anomalies')}",
        f"- Anomaly Rate: {anomaly_summary.get('anomaly_rate_percent')}%",
        "",
        "## Output Files",
        "",
        f"- Forecast Output: `{summary.get('forecast_output_path')}`",
        f"- Anomaly Output: `{summary.get('anomaly_output_path')}`",
        f"- Summary JSON: `{json_path}`",
        f"- Summary Markdown: `{markdown_path}`",
        "",
        "## Performance Summary",
        "",
        f"- Total Records Tracked: {performance_summary.get('total_records')}",
        f"- Total Execution Time: {performance_summary.get('total_execution_time_seconds')} seconds",
        f"- Successful Steps: {performance_summary.get('successful_steps')}",
        f"- Failed Steps: {performance_summary.get('failed_steps')}",
    ]

    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    logger.info("Inference summary JSON saved at: %s", json_path)
    logger.info("Inference summary Markdown saved at: %s", markdown_path)


# ============================================================
# Inference Runner
# ============================================================

class PredictionPipeline:
    """
    Run forecasting and anomaly detection inference.
    """

    def __init__(
        self,
        input_path: Path = DATA_PATH,
        forecast_model: str = "xgboost",
        forecast_steps: int = 30,
        run_forecasting: bool = True,
        run_anomaly_detection: bool = True,
    ) -> None:
        """
        Initialize PredictionPipeline.

        Args:
            input_path (Path): Input CSV path.
            forecast_model (str): Forecasting model to use.
                Options: xgboost, arima, prophet, lstm, all
            forecast_steps (int): Number of future forecast periods.
            run_forecasting (bool): Whether to generate forecasts.
            run_anomaly_detection (bool): Whether to detect anomalies.
        """
        self.input_path = input_path
        self.forecast_model = forecast_model
        self.forecast_steps = forecast_steps
        self.run_forecasting = run_forecasting
        self.run_anomaly_detection = run_anomaly_detection

    def run(self) -> Dict[str, Any]:
        """
        Run the full inference workflow.

        Returns:
            Dict[str, Any]: Inference summary.
        """
        log_section(logger, "Starting Prediction Pipeline")

        try:
            ensure_inference_directories()
            create_required_directories()

            with track_step("load_inference_data", category="inference"):
                input_dataframe = load_input_data(self.input_path)

            input_summary = get_input_summary(input_dataframe)

            forecast_dataframe = pd.DataFrame()
            anomaly_dataframe = pd.DataFrame()

            if self.run_forecasting:
                log_step_start(logger, "Generate forecasts")

                with track_step("generate_forecasts", category="inference"):
                    forecast_dataframe = generate_forecasts(
                        dataframe=input_dataframe,
                        model_name=self.forecast_model,
                        forecast_steps=self.forecast_steps,
                    )

                    save_forecasts(forecast_dataframe)

                log_step_end(logger, "Generate forecasts")

            if self.run_anomaly_detection:
                log_step_start(logger, "Detect anomalies")

                with track_step("detect_anomalies", category="inference"):
                    anomaly_dataframe = detect_anomalies_with_saved_model(
                        dataframe=input_dataframe,
                    )

                    save_anomaly_results(anomaly_dataframe)

                log_step_end(logger, "Detect anomalies")

            save_performance_reports()
            performance_summary = get_performance_summary()

            forecast_summary = summarize_forecast_results(forecast_dataframe)
            anomaly_summary = summarize_anomaly_results(anomaly_dataframe)

            summary = {
                "status": "success",
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "input_data_path": str(self.input_path),
                "configuration": {
                    "forecast_model": self.forecast_model,
                    "forecast_steps": self.forecast_steps,
                    "run_forecasting": self.run_forecasting,
                    "run_anomaly_detection": self.run_anomaly_detection,
                },
                "input_summary": input_summary,
                "forecast_summary": forecast_summary,
                "anomaly_summary": anomaly_summary,
                "forecast_output_path": str(FORECAST_OUTPUT_PATH),
                "anomaly_output_path": str(ANOMALY_OUTPUT_PATH),
                "summary_json_path": str(INFERENCE_SUMMARY_JSON_PATH),
                "summary_markdown_path": str(INFERENCE_SUMMARY_MD_PATH),
                "performance_summary": performance_summary,
            }

            save_inference_summary(summary)

            log_section(logger, "Prediction Pipeline Completed Successfully")

            return summary

        except Exception as error:
            log_step_error(logger, "Prediction pipeline", error)

            save_performance_reports()

            summary = {
                "status": "failed",
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "input_data_path": str(self.input_path),
                "error_type": error.__class__.__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            }

            save_inference_summary(summary)

            return summary


# ============================================================
# Public Convenience Functions
# ============================================================

def predict_forecast(
    dataframe: pd.DataFrame,
    model_name: str = "xgboost",
    forecast_steps: int = 30,
) -> pd.DataFrame:
    """
    Public helper to generate forecasts from a dataframe.

    Args:
        dataframe (pd.DataFrame): Input historical dataframe.
        model_name (str): Forecast model name.
        forecast_steps (int): Forecast horizon.

    Returns:
        pd.DataFrame: Forecast dataframe.
    """
    clean_dataframe = clean_input_data(dataframe)

    return generate_forecasts(
        dataframe=clean_dataframe,
        model_name=model_name,
        forecast_steps=forecast_steps,
    )


def predict_anomalies(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Public helper to detect anomalies from a dataframe.

    Args:
        dataframe (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Anomaly result dataframe.
    """
    clean_dataframe = clean_input_data(dataframe)

    return detect_anomalies_with_saved_model(clean_dataframe)


def run_prediction_pipeline(
    input_path: Path = DATA_PATH,
    forecast_model: str = "xgboost",
    forecast_steps: int = 30,
    run_forecasting: bool = True,
    run_anomaly_detection: bool = True,
) -> Dict[str, Any]:
    """
    Public helper to run the full prediction pipeline.

    Args:
        input_path (Path): Input CSV path.
        forecast_model (str): Forecasting model name.
        forecast_steps (int): Forecast horizon.
        run_forecasting (bool): Whether to run forecasting.
        run_anomaly_detection (bool): Whether to run anomaly detection.

    Returns:
        Dict[str, Any]: Inference summary.
    """
    pipeline = PredictionPipeline(
        input_path=input_path,
        forecast_model=forecast_model,
        forecast_steps=forecast_steps,
        run_forecasting=run_forecasting,
        run_anomaly_detection=run_anomaly_detection,
    )

    return pipeline.run()


# ============================================================
# Main Function
# ============================================================

def main() -> None:
    """
    Main function to run inference.
    """
    pipeline = PredictionPipeline(
        input_path=DATA_PATH,
        forecast_model="xgboost",
        forecast_steps=30,
        run_forecasting=True,
        run_anomaly_detection=True,
    )

    summary = pipeline.run()

    print("\n" + "=" * 80)
    print("PREDICTION PIPELINE SUMMARY")
    print("=" * 80)

    print("Status:", summary.get("status"))

    if summary.get("status") == "success":
        print("\nInput Summary:")
        input_summary = summary.get("input_summary", {})
        print("- Total Records:", input_summary.get("total_records"))
        print("- Total Assets:", input_summary.get("total_assets"))
        print("- Start Date:", input_summary.get("start_date"))
        print("- End Date:", input_summary.get("end_date"))

        print("\nForecast Summary:")
        forecast_summary = summary.get("forecast_summary", {})
        print("- Forecast Rows:", forecast_summary.get("forecast_rows"))
        print("- Models Used:", forecast_summary.get("models_used"))
        print("- Assets Forecasted:", forecast_summary.get("assets_forecasted"))
        print("- Forecast Start Date:", forecast_summary.get("forecast_start_date"))
        print("- Forecast End Date:", forecast_summary.get("forecast_end_date"))

        print("\nAnomaly Summary:")
        anomaly_summary = summary.get("anomaly_summary", {})
        print("- Total Records:", anomaly_summary.get("total_records"))
        print("- Total Anomalies:", anomaly_summary.get("total_anomalies"))
        print("- Anomaly Rate %:", anomaly_summary.get("anomaly_rate_percent"))

        print("\nSaved Files:")
        print("- outputs/inference_forecasts.csv")
        print("- outputs/inference_anomaly_results.csv")
        print("- reports/tables/inference_summary.json")
        print("- reports/inference_summary.md")

    else:
        print("\nPrediction pipeline failed.")
        print("Error Type:", summary.get("error_type"))
        print("Error Message:", summary.get("error_message"))
        print("\nCheck logs/app.log or logs/training.log for details.")

    print("=" * 80)


# ============================================================
# Script Entry Point
# ============================================================

if __name__ == "__main__":
    main()