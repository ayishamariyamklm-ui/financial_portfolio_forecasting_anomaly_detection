"""
Forecast service for the Financial Portfolio Forecasting
and Anomaly Detection project.

This module provides a FastAPI service for:

1. Health check
2. Forecast generation
3. Anomaly detection
4. Combined forecast + anomaly inference
5. CSV upload prediction
6. Saving API outputs

Dataset expected columns:
- date
- asset
- price

Run from project root:

    uvicorn src.inference.forecast-service:app --reload

Recommended rename for production:

    src/inference/forecast_service.py

Then run:

    uvicorn src.inference.forecast_service:app --reload
"""

import json
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# Add Project Root to Python Path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ============================================================
# Project Imports
# ============================================================

from src.inference.predict import (
    ANOMALY_OUTPUT_PATH,
    FORECAST_OUTPUT_PATH,
    INFERENCE_SUMMARY_JSON_PATH,
    clean_input_data,
    detect_anomalies_with_saved_model,
    generate_forecasts,
    get_input_summary,
    save_anomaly_results,
    save_forecasts,
    summarize_anomaly_results,
    summarize_forecast_results,
)
from src.models.anomaly_detector import ANOMALY_MODEL_PATH
from src.models.arima_model import ARIMA_MODEL_PATH
from src.models.lstm_model import LSTM_MODEL_DIR, LSTM_SCALER_PATH
from src.models.prophet_model import PROPHET_MODEL_PATH
from src.models.xgboost_model import XGBOOST_MODEL_PATH
from src.monitoring.logger import get_project_logger
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

API_FORECAST_OUTPUT_PATH = OUTPUTS_DIR / "api_forecast_results.csv"
API_ANOMALY_OUTPUT_PATH = OUTPUTS_DIR / "api_anomaly_results.csv"
API_COMBINED_SUMMARY_PATH = TABLES_DIR / "api_inference_summary.json"


# ============================================================
# Logger
# ============================================================

logger = get_project_logger()


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="Financial Portfolio Forecasting & Anomaly Detection API",
    description=(
        "API service for forecasting future asset prices and detecting "
        "financial asset price anomalies."
    ),
    version="1.0.0",
)


# ============================================================
# CORS Middleware
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Pydantic Schemas
# ============================================================

class PriceRecord(BaseModel):
    """
    Single financial asset price record.
    """

    date: str = Field(
        ...,
        description="Date of price record. Example: 2024-01-01",
    )
    asset: str = Field(
        ...,
        description="Asset name. Example: Stock A",
    )
    price: float = Field(
        ...,
        ge=0,
        description="Asset price. Must be non-negative.",
    )


class ForecastRequest(BaseModel):
    """
    Forecast request schema.
    """

    records: List[PriceRecord] = Field(
        ...,
        description="Historical price records.",
    )
    model_name: str = Field(
        default="xgboost",
        description="Forecast model: xgboost, arima, prophet, lstm, or all.",
    )
    forecast_steps: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Number of future periods to forecast.",
    )
    save_output: bool = Field(
        default=True,
        description="Whether to save forecast results as CSV.",
    )


class AnomalyRequest(BaseModel):
    """
    Anomaly detection request schema.
    """

    records: List[PriceRecord] = Field(
        ...,
        description="Historical price records.",
    )
    save_output: bool = Field(
        default=True,
        description="Whether to save anomaly results as CSV.",
    )


class CombinedInferenceRequest(BaseModel):
    """
    Combined forecast and anomaly detection request schema.
    """

    records: List[PriceRecord] = Field(
        ...,
        description="Historical price records.",
    )
    model_name: str = Field(
        default="xgboost",
        description="Forecast model: xgboost, arima, prophet, lstm, or all.",
    )
    forecast_steps: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Number of future periods to forecast.",
    )
    run_forecasting: bool = Field(
        default=True,
        description="Whether to run forecasting.",
    )
    run_anomaly_detection: bool = Field(
        default=True,
        description="Whether to run anomaly detection.",
    )
    save_output: bool = Field(
        default=True,
        description="Whether to save outputs.",
    )


# ============================================================
# Utility Functions
# ============================================================

def ensure_service_directories() -> None:
    """
    Create required service directories.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    create_required_directories()


def records_to_dataframe(records: List[PriceRecord]) -> pd.DataFrame:
    """
    Convert API records into cleaned dataframe.

    Args:
        records (List[PriceRecord]): Input records.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    if not records:
        raise ValueError("No records provided.")

    dataframe = pd.DataFrame([record.dict() for record in records])

    cleaned_dataframe = clean_input_data(dataframe)

    return cleaned_dataframe


def dataframe_to_records(dataframe: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert dataframe into JSON-safe records.

    Args:
        dataframe (pd.DataFrame): Input dataframe.

    Returns:
        List[Dict[str, Any]]: JSON-safe records.
    """
    data = dataframe.copy()

    for column in data.columns:
        if pd.api.types.is_datetime64_any_dtype(data[column]):
            data[column] = data[column].astype(str)

    data = data.replace([float("inf"), float("-inf")], None)
    data = data.where(pd.notnull(data), None)

    return data.to_dict(orient="records")


def save_json_summary(summary: Dict[str, Any], file_path: Path) -> None:
    """
    Save summary dictionary as JSON.

    Args:
        summary (Dict[str, Any]): Summary dictionary.
        file_path (Path): Output path.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, default=str)


def load_default_data() -> pd.DataFrame:
    """
    Load default dataset from data/raw.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Default dataset not found at: {DATA_PATH}"
        )

    dataframe = pd.read_csv(DATA_PATH)

    return clean_input_data(dataframe)


def get_model_artifact_status() -> Dict[str, Any]:
    """
    Check availability of trained model artifacts.

    Returns:
        Dict[str, Any]: Model artifact status.
    """
    return {
        "xgboost": {
            "available": XGBOOST_MODEL_PATH.exists(),
            "path": str(XGBOOST_MODEL_PATH),
        },
        "arima": {
            "available": ARIMA_MODEL_PATH.exists(),
            "path": str(ARIMA_MODEL_PATH),
        },
        "prophet": {
            "available": PROPHET_MODEL_PATH.exists(),
            "path": str(PROPHET_MODEL_PATH),
        },
        "lstm": {
            "available": LSTM_SCALER_PATH.exists() and LSTM_MODEL_DIR.exists(),
            "model_dir": str(LSTM_MODEL_DIR),
            "scaler_path": str(LSTM_SCALER_PATH),
        },
        "anomaly_detector": {
            "available": ANOMALY_MODEL_PATH.exists(),
            "path": str(ANOMALY_MODEL_PATH),
        },
    }


def build_success_response(
    message: str,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build standard success response.

    Args:
        message (str): Response message.
        data (Optional[Dict[str, Any]]): Response data.

    Returns:
        Dict[str, Any]: API response.
    """
    response = {
        "status": "success",
        "message": message,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    if data is not None:
        response["data"] = data

    return response


def build_error_response(error: Exception) -> Dict[str, Any]:
    """
    Build standard error response.

    Args:
        error (Exception): Exception object.

    Returns:
        Dict[str, Any]: Error response.
    """
    return {
        "status": "failed",
        "error_type": error.__class__.__name__,
        "error_message": str(error),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ============================================================
# API Routes
# ============================================================

@app.get("/")
def root() -> Dict[str, Any]:
    """
    Root endpoint.
    """
    return build_success_response(
        message="Financial Portfolio Forecasting API is running.",
        data={
            "project": "Financial Portfolio Forecasting & Anomaly Detection",
            "version": "1.0.0",
            "docs_url": "/docs",
            "health_url": "/health",
        },
    )


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.
    """
    ensure_service_directories()

    model_status = get_model_artifact_status()

    return build_success_response(
        message="Service health check completed.",
        data={
            "service": "forecast-service",
            "project_root": str(PROJECT_ROOT),
            "default_data_available": DATA_PATH.exists(),
            "default_data_path": str(DATA_PATH),
            "model_artifacts": model_status,
        },
    )


@app.get("/models/status")
def model_status() -> Dict[str, Any]:
    """
    Return trained model artifact status.
    """
    return build_success_response(
        message="Model artifact status loaded.",
        data=get_model_artifact_status(),
    )


@app.post("/forecast")
def forecast(request: ForecastRequest) -> Dict[str, Any]:
    """
    Generate future forecasts from request records.
    """
    try:
        ensure_service_directories()

        with track_step("api_forecast", category="api"):
            input_dataframe = records_to_dataframe(request.records)

            forecast_dataframe = generate_forecasts(
                dataframe=input_dataframe,
                model_name=request.model_name,
                forecast_steps=request.forecast_steps,
            )

            if request.save_output:
                forecast_dataframe.to_csv(API_FORECAST_OUTPUT_PATH, index=False)

            forecast_summary = summarize_forecast_results(forecast_dataframe)

            save_performance_reports()

        return build_success_response(
            message="Forecast generated successfully.",
            data={
                "input_summary": get_input_summary(input_dataframe),
                "forecast_summary": forecast_summary,
                "forecast_records": dataframe_to_records(forecast_dataframe),
                "saved_path": str(API_FORECAST_OUTPUT_PATH)
                if request.save_output
                else None,
                "performance_summary": get_performance_summary(),
            },
        )

    except Exception as error:
        logger.error("Forecast API failed: %s", error, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=build_error_response(error),
        )


@app.post("/anomaly")
def detect_anomaly(request: AnomalyRequest) -> Dict[str, Any]:
    """
    Detect anomalies from request records.
    """
    try:
        ensure_service_directories()

        with track_step("api_anomaly_detection", category="api"):
            input_dataframe = records_to_dataframe(request.records)

            anomaly_dataframe = detect_anomalies_with_saved_model(
                input_dataframe,
            )

            if request.save_output:
                anomaly_dataframe.to_csv(API_ANOMALY_OUTPUT_PATH, index=False)

            anomaly_summary = summarize_anomaly_results(anomaly_dataframe)

            save_performance_reports()

        return build_success_response(
            message="Anomaly detection completed successfully.",
            data={
                "input_summary": get_input_summary(input_dataframe),
                "anomaly_summary": anomaly_summary,
                "anomaly_records": dataframe_to_records(anomaly_dataframe),
                "saved_path": str(API_ANOMALY_OUTPUT_PATH)
                if request.save_output
                else None,
                "performance_summary": get_performance_summary(),
            },
        )

    except Exception as error:
        logger.error("Anomaly API failed: %s", error, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=build_error_response(error),
        )


@app.post("/inference")
def combined_inference(request: CombinedInferenceRequest) -> Dict[str, Any]:
    """
    Run combined forecasting and anomaly detection.
    """
    try:
        ensure_service_directories()

        with track_step("api_combined_inference", category="api"):
            input_dataframe = records_to_dataframe(request.records)

            forecast_dataframe = pd.DataFrame()
            anomaly_dataframe = pd.DataFrame()

            if request.run_forecasting:
                forecast_dataframe = generate_forecasts(
                    dataframe=input_dataframe,
                    model_name=request.model_name,
                    forecast_steps=request.forecast_steps,
                )

                if request.save_output:
                    forecast_dataframe.to_csv(API_FORECAST_OUTPUT_PATH, index=False)

            if request.run_anomaly_detection:
                anomaly_dataframe = detect_anomalies_with_saved_model(
                    input_dataframe,
                )

                if request.save_output:
                    anomaly_dataframe.to_csv(API_ANOMALY_OUTPUT_PATH, index=False)

            forecast_summary = summarize_forecast_results(forecast_dataframe)
            anomaly_summary = summarize_anomaly_results(anomaly_dataframe)

            save_performance_reports()

            summary = {
                "input_summary": get_input_summary(input_dataframe),
                "forecast_summary": forecast_summary,
                "anomaly_summary": anomaly_summary,
                "forecast_output_path": str(API_FORECAST_OUTPUT_PATH)
                if request.save_output and request.run_forecasting
                else None,
                "anomaly_output_path": str(API_ANOMALY_OUTPUT_PATH)
                if request.save_output and request.run_anomaly_detection
                else None,
                "performance_summary": get_performance_summary(),
            }

            if request.save_output:
                save_json_summary(summary, API_COMBINED_SUMMARY_PATH)

        return build_success_response(
            message="Combined inference completed successfully.",
            data={
                **summary,
                "forecast_records": dataframe_to_records(forecast_dataframe),
                "anomaly_records": dataframe_to_records(anomaly_dataframe),
                "summary_path": str(API_COMBINED_SUMMARY_PATH)
                if request.save_output
                else None,
            },
        )

    except Exception as error:
        logger.error("Combined inference API failed: %s", error, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=build_error_response(error),
        )


@app.get("/forecast/default")
def forecast_default_data(
    model_name: str = Query(
        default="xgboost",
        description="Forecast model: xgboost, arima, prophet, lstm, or all.",
    ),
    forecast_steps: int = Query(
        default=30,
        ge=1,
        le=365,
        description="Number of future periods.",
    ),
    save_output: bool = Query(
        default=True,
        description="Whether to save output.",
    ),
) -> Dict[str, Any]:
    """
    Generate forecasts using default dataset from data/raw.
    """
    try:
        ensure_service_directories()

        with track_step("api_forecast_default_data", category="api"):
            input_dataframe = load_default_data()

            forecast_dataframe = generate_forecasts(
                dataframe=input_dataframe,
                model_name=model_name,
                forecast_steps=forecast_steps,
            )

            if save_output:
                forecast_dataframe.to_csv(API_FORECAST_OUTPUT_PATH, index=False)

            save_performance_reports()

        return build_success_response(
            message="Default data forecast generated successfully.",
            data={
                "input_summary": get_input_summary(input_dataframe),
                "forecast_summary": summarize_forecast_results(forecast_dataframe),
                "forecast_records": dataframe_to_records(forecast_dataframe),
                "saved_path": str(API_FORECAST_OUTPUT_PATH)
                if save_output
                else None,
                "performance_summary": get_performance_summary(),
            },
        )

    except Exception as error:
        logger.error("Default forecast API failed: %s", error, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=build_error_response(error),
        )


@app.get("/anomaly/default")
def anomaly_default_data(
    save_output: bool = Query(
        default=True,
        description="Whether to save output.",
    ),
) -> Dict[str, Any]:
    """
    Detect anomalies using default dataset from data/raw.
    """
    try:
        ensure_service_directories()

        with track_step("api_anomaly_default_data", category="api"):
            input_dataframe = load_default_data()

            anomaly_dataframe = detect_anomalies_with_saved_model(
                input_dataframe,
            )

            if save_output:
                anomaly_dataframe.to_csv(API_ANOMALY_OUTPUT_PATH, index=False)

            save_performance_reports()

        return build_success_response(
            message="Default data anomaly detection completed successfully.",
            data={
                "input_summary": get_input_summary(input_dataframe),
                "anomaly_summary": summarize_anomaly_results(anomaly_dataframe),
                "anomaly_records": dataframe_to_records(anomaly_dataframe),
                "saved_path": str(API_ANOMALY_OUTPUT_PATH)
                if save_output
                else None,
                "performance_summary": get_performance_summary(),
            },
        )

    except Exception as error:
        logger.error("Default anomaly API failed: %s", error, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=build_error_response(error),
        )


@app.post("/upload/forecast")
async def upload_forecast(
    file: UploadFile = File(...),
    model_name: str = Query(
        default="xgboost",
        description="Forecast model: xgboost, arima, prophet, lstm, or all.",
    ),
    forecast_steps: int = Query(
        default=30,
        ge=1,
        le=365,
        description="Number of future forecast periods.",
    ),
    save_output: bool = Query(
        default=True,
        description="Whether to save output.",
    ),
) -> Dict[str, Any]:
    """
    Upload CSV and generate forecasts.

    CSV must contain:
    - date
    - asset
    - price

    or:
    - Date
    - Asset
    - Price
    """
    try:
        ensure_service_directories()

        if not file.filename.lower().endswith(".csv"):
            raise ValueError("Only CSV files are supported.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_file.write(await file.read())
            temp_path = Path(temp_file.name)

        with track_step("api_upload_forecast", category="api"):
            input_dataframe = pd.read_csv(temp_path)
            input_dataframe = clean_input_data(input_dataframe)

            forecast_dataframe = generate_forecasts(
                dataframe=input_dataframe,
                model_name=model_name,
                forecast_steps=forecast_steps,
            )

            if save_output:
                forecast_dataframe.to_csv(API_FORECAST_OUTPUT_PATH, index=False)

            save_performance_reports()

        temp_path.unlink(missing_ok=True)

        return build_success_response(
            message="Uploaded CSV forecast generated successfully.",
            data={
                "uploaded_file": file.filename,
                "input_summary": get_input_summary(input_dataframe),
                "forecast_summary": summarize_forecast_results(forecast_dataframe),
                "forecast_records": dataframe_to_records(forecast_dataframe),
                "saved_path": str(API_FORECAST_OUTPUT_PATH)
                if save_output
                else None,
                "performance_summary": get_performance_summary(),
            },
        )

    except Exception as error:
        logger.error("Upload forecast API failed: %s", error, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=build_error_response(error),
        )


@app.post("/upload/anomaly")
async def upload_anomaly(
    file: UploadFile = File(...),
    save_output: bool = Query(
        default=True,
        description="Whether to save output.",
    ),
) -> Dict[str, Any]:
    """
    Upload CSV and detect anomalies.

    CSV must contain:
    - date
    - asset
    - price

    or:
    - Date
    - Asset
    - Price
    """
    try:
        ensure_service_directories()

        if not file.filename.lower().endswith(".csv"):
            raise ValueError("Only CSV files are supported.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_file.write(await file.read())
            temp_path = Path(temp_file.name)

        with track_step("api_upload_anomaly", category="api"):
            input_dataframe = pd.read_csv(temp_path)
            input_dataframe = clean_input_data(input_dataframe)

            anomaly_dataframe = detect_anomalies_with_saved_model(
                input_dataframe,
            )

            if save_output:
                anomaly_dataframe.to_csv(API_ANOMALY_OUTPUT_PATH, index=False)

            save_performance_reports()

        temp_path.unlink(missing_ok=True)

        return build_success_response(
            message="Uploaded CSV anomaly detection completed successfully.",
            data={
                "uploaded_file": file.filename,
                "input_summary": get_input_summary(input_dataframe),
                "anomaly_summary": summarize_anomaly_results(anomaly_dataframe),
                "anomaly_records": dataframe_to_records(anomaly_dataframe),
                "saved_path": str(API_ANOMALY_OUTPUT_PATH)
                if save_output
                else None,
                "performance_summary": get_performance_summary(),
            },
        )

    except Exception as error:
        logger.error("Upload anomaly API failed: %s", error, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=build_error_response(error),
        )


# ============================================================
# Local Script Runner
# ============================================================

def run_local_service_test() -> Dict[str, Any]:
    """
    Run a local service test without starting API server.

    Returns:
        Dict[str, Any]: Local test summary.
    """
    try:
        ensure_service_directories()

        input_dataframe = load_default_data()

        forecast_dataframe = pd.DataFrame()
        anomaly_dataframe = pd.DataFrame()

        try:
            forecast_dataframe = generate_forecasts(
                dataframe=input_dataframe,
                model_name="xgboost",
                forecast_steps=30,
            )
            save_forecasts(forecast_dataframe, FORECAST_OUTPUT_PATH)

        except Exception as forecast_error:
            logger.warning("Local forecast test skipped: %s", forecast_error)

        try:
            anomaly_dataframe = detect_anomalies_with_saved_model(
                input_dataframe,
            )
            save_anomaly_results(anomaly_dataframe, ANOMALY_OUTPUT_PATH)

        except Exception as anomaly_error:
            logger.warning("Local anomaly test skipped: %s", anomaly_error)

        save_performance_reports()

        summary = {
            "status": "success",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input_summary": get_input_summary(input_dataframe),
            "forecast_summary": summarize_forecast_results(forecast_dataframe),
            "anomaly_summary": summarize_anomaly_results(anomaly_dataframe),
            "forecast_output_path": str(FORECAST_OUTPUT_PATH),
            "anomaly_output_path": str(ANOMALY_OUTPUT_PATH),
            "performance_summary": get_performance_summary(),
        }

        save_json_summary(summary, INFERENCE_SUMMARY_JSON_PATH)

        return summary

    except Exception as error:
        return {
            "status": "failed",
            "error_type": error.__class__.__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
        }


# ============================================================
# Script Entry Point
# ============================================================

if __name__ == "__main__":
    result = run_local_service_test()

    print("\n" + "=" * 80)
    print("FORECAST SERVICE LOCAL TEST SUMMARY")
    print("=" * 80)

    print("Status:", result.get("status"))

    if result.get("status") == "success":
        print("\nInput Summary:")
        input_summary = result.get("input_summary", {})
        print("- Total Records:", input_summary.get("total_records"))
        print("- Total Assets:", input_summary.get("total_assets"))
        print("- Start Date:", input_summary.get("start_date"))
        print("- End Date:", input_summary.get("end_date"))

        print("\nForecast Summary:")
        forecast_summary = result.get("forecast_summary", {})
        print("- Forecast Rows:", forecast_summary.get("forecast_rows"))
        print("- Models Used:", forecast_summary.get("models_used"))
        print("- Assets Forecasted:", forecast_summary.get("assets_forecasted"))

        print("\nAnomaly Summary:")
        anomaly_summary = result.get("anomaly_summary", {})
        print("- Total Records:", anomaly_summary.get("total_records"))
        print("- Total Anomalies:", anomaly_summary.get("total_anomalies"))
        print("- Anomaly Rate %:", anomaly_summary.get("anomaly_rate_percent"))

        print("\nSaved Files:")
        print("- outputs/inference_forecasts.csv")
        print("- outputs/inference_anomaly_results.csv")
        print("- reports/tables/inference_summary.json")

    else:
        print("\nLocal service test failed.")
        print("Error Type:", result.get("error_type"))
        print("Error Message:", result.get("error_message"))

    print("=" * 80)