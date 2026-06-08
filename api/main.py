"""
Main FastAPI application for the Financial Portfolio Forecasting
and Anomaly Detection project.

This API provides:

1. Health check
2. Forecasting endpoint
3. Anomaly detection endpoint
4. Combined prediction endpoint
5. CSV upload endpoint
6. Latest summary endpoint

Dataset expected columns:
- date
- asset
- price

Run from project root:

    uvicorn api.main:app --reload

Open API docs:

    http://127.0.0.1:8000/docs
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ============================================================
# Project Imports
# ============================================================

from src.inference.predict import (
    clean_input_data,
    detect_anomalies_with_saved_model,
    generate_forecasts,
    get_input_summary,
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
API_COMBINED_OUTPUT_PATH = OUTPUTS_DIR / "api_combined_results.csv"
API_SUMMARY_JSON_PATH = TABLES_DIR / "api_summary.json"
API_SUMMARY_MD_PATH = REPORTS_DIR / "api_summary.md"


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
        "Production-style API for forecasting asset prices and detecting "
        "financial anomalies using trained machine learning models."
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
# Request and Response Schemas
# ============================================================

class PriceRecord(BaseModel):
    """
    Single asset price record.
    """

    date: str = Field(
        ...,
        description="Date of the asset price record. Example: 2024-01-01",
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
    Forecasting request schema.
    """

    records: List[PriceRecord] = Field(
        ...,
        description="Historical asset price records.",
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
        description="Whether to save forecast output as CSV.",
    )
    return_records: bool = Field(
        default=True,
        description="Whether to return forecast records in API response.",
    )


class AnomalyRequest(BaseModel):
    """
    Anomaly detection request schema.
    """

    records: List[PriceRecord] = Field(
        ...,
        description="Historical asset price records.",
    )
    save_output: bool = Field(
        default=True,
        description="Whether to save anomaly output as CSV.",
    )
    return_records: bool = Field(
        default=True,
        description="Whether to return anomaly records in API response.",
    )
    only_anomalies: bool = Field(
        default=False,
        description="Whether to return only anomalous records.",
    )


class CombinedPredictionRequest(BaseModel):
    """
    Combined forecasting and anomaly detection request schema.
    """

    records: List[PriceRecord] = Field(
        ...,
        description="Historical asset price records.",
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
    return_records: bool = Field(
        default=True,
        description="Whether to return prediction records.",
    )
    only_anomalies: bool = Field(
        default=False,
        description="Whether anomaly response should return only anomaly records.",
    )


# ============================================================
# Directory Helpers
# ============================================================

def ensure_api_directories() -> None:
    """
    Create required API output directories.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    create_required_directories()


# ============================================================
# Data Helpers
# ============================================================

def records_to_dataframe(records: List[PriceRecord]) -> pd.DataFrame:
    """
    Convert request records to cleaned dataframe.

    Args:
        records (List[PriceRecord]): List of request records.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    if not records:
        raise ValueError("No records provided.")

    dataframe = pd.DataFrame([record.dict() for record in records])

    return clean_input_data(dataframe)


def load_default_dataset() -> pd.DataFrame:
    """
    Load default dataset from data/raw.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Default dataset not found at: {DATA_PATH}. "
            "Please place financial_portfolio_data.csv inside data/raw/."
        )

    dataframe = pd.read_csv(DATA_PATH)

    return clean_input_data(dataframe)


def dataframe_to_json_records(dataframe: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert dataframe to JSON-safe records.

    Args:
        dataframe (pd.DataFrame): Input dataframe.

    Returns:
        List[Dict[str, Any]]: JSON-safe records.
    """
    if dataframe is None or dataframe.empty:
        return []

    data = dataframe.copy()

    for column in data.columns:
        if pd.api.types.is_datetime64_any_dtype(data[column]):
            data[column] = data[column].astype(str)

    data = data.replace([float("inf"), float("-inf")], None)
    data = data.where(pd.notnull(data), None)

    return data.to_dict(orient="records")


def filter_anomaly_records(
    dataframe: pd.DataFrame,
    only_anomalies: bool = False,
) -> pd.DataFrame:
    """
    Filter anomaly results if only anomalies are requested.

    Args:
        dataframe (pd.DataFrame): Anomaly dataframe.
        only_anomalies (bool): Whether to return only anomalies.

    Returns:
        pd.DataFrame: Filtered dataframe.
    """
    if dataframe is None or dataframe.empty:
        return pd.DataFrame()

    if only_anomalies and "is_anomaly" in dataframe.columns:
        data = dataframe.copy()
        data["is_anomaly"] = data["is_anomaly"].astype(bool)
        return data[data["is_anomaly"]].copy()

    return dataframe


# ============================================================
# Status Helpers
# ============================================================

def get_model_artifact_status() -> Dict[str, Any]:
    """
    Get trained model artifact status.

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
            "available": LSTM_MODEL_DIR.exists() and LSTM_SCALER_PATH.exists(),
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
    Build a standard success API response.

    Args:
        message (str): Success message.
        data (Optional[Dict[str, Any]]): Response data.

    Returns:
        Dict[str, Any]: Response dictionary.
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
    Build a standard error API response.

    Args:
        error (Exception): Exception object.

    Returns:
        Dict[str, Any]: Error dictionary.
    """
    return {
        "status": "failed",
        "error_type": error.__class__.__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ============================================================
# Save Helpers
# ============================================================

def save_json_summary(summary: Dict[str, Any], file_path: Path) -> None:
    """
    Save summary dictionary as JSON.

    Args:
        summary (Dict[str, Any]): Summary dictionary.
        file_path (Path): Output JSON path.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, default=str)


def save_markdown_summary(summary: Dict[str, Any], file_path: Path) -> None:
    """
    Save API summary as Markdown.

    Args:
        summary (Dict[str, Any]): Summary dictionary.
        file_path (Path): Markdown output path.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    input_summary = summary.get("input_summary", {})
    forecast_summary = summary.get("forecast_summary", {})
    anomaly_summary = summary.get("anomaly_summary", {})
    performance_summary = summary.get("performance_summary", {})

    lines = [
        "# API Prediction Summary",
        "",
        "## Project",
        "",
        "Financial Portfolio Forecasting & Anomaly Detection",
        "",
        "## Run Information",
        "",
        f"- Generated At: {summary.get('generated_at')}",
        f"- Status: {summary.get('status')}",
        "",
        "## Input Summary",
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
        f"- Normal Records: {anomaly_summary.get('normal_records')}",
        f"- Anomaly Rate: {anomaly_summary.get('anomaly_rate_percent')}%",
        "",
        "## Output Files",
        "",
        f"- Forecast Output: `{summary.get('forecast_output_path')}`",
        f"- Anomaly Output: `{summary.get('anomaly_output_path')}`",
        f"- Summary JSON: `{API_SUMMARY_JSON_PATH}`",
        f"- Summary Markdown: `{API_SUMMARY_MD_PATH}`",
        "",
        "## Performance Summary",
        "",
        f"- Total Records Tracked: {performance_summary.get('total_records')}",
        f"- Total Execution Time: {performance_summary.get('total_execution_time_seconds')} seconds",
        f"- Successful Steps: {performance_summary.get('successful_steps')}",
        f"- Failed Steps: {performance_summary.get('failed_steps')}",
    ]

    file_path.write_text("\n".join(lines), encoding="utf-8")


def save_api_summary(summary: Dict[str, Any]) -> None:
    """
    Save API summary as JSON and Markdown.

    Args:
        summary (Dict[str, Any]): API summary dictionary.
    """
    save_json_summary(summary, API_SUMMARY_JSON_PATH)
    save_markdown_summary(summary, API_SUMMARY_MD_PATH)


# ============================================================
# Core API Workflows
# ============================================================

def run_forecast_workflow(
    input_dataframe: pd.DataFrame,
    model_name: str = "xgboost",
    forecast_steps: int = 30,
    save_output: bool = True,
) -> Dict[str, Any]:
    """
    Run forecasting workflow.

    Args:
        input_dataframe (pd.DataFrame): Input dataframe.
        model_name (str): Forecasting model name.
        forecast_steps (int): Forecast horizon.
        save_output (bool): Whether to save outputs.

    Returns:
        Dict[str, Any]: Forecast workflow result.
    """
    forecast_dataframe = generate_forecasts(
        dataframe=input_dataframe,
        model_name=model_name,
        forecast_steps=forecast_steps,
    )

    if save_output:
        API_FORECAST_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        forecast_dataframe.to_csv(API_FORECAST_OUTPUT_PATH, index=False)

    return {
        "forecast_dataframe": forecast_dataframe,
        "forecast_summary": summarize_forecast_results(forecast_dataframe),
        "forecast_output_path": str(API_FORECAST_OUTPUT_PATH) if save_output else None,
    }


def run_anomaly_workflow(
    input_dataframe: pd.DataFrame,
    save_output: bool = True,
) -> Dict[str, Any]:
    """
    Run anomaly detection workflow.

    Args:
        input_dataframe (pd.DataFrame): Input dataframe.
        save_output (bool): Whether to save outputs.

    Returns:
        Dict[str, Any]: Anomaly workflow result.
    """
    anomaly_dataframe = detect_anomalies_with_saved_model(input_dataframe)

    if save_output:
        API_ANOMALY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        anomaly_dataframe.to_csv(API_ANOMALY_OUTPUT_PATH, index=False)

    return {
        "anomaly_dataframe": anomaly_dataframe,
        "anomaly_summary": summarize_anomaly_results(anomaly_dataframe),
        "anomaly_output_path": str(API_ANOMALY_OUTPUT_PATH) if save_output else None,
    }


def build_combined_summary(
    input_dataframe: pd.DataFrame,
    forecast_result: Optional[Dict[str, Any]] = None,
    anomaly_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build combined API summary.

    Args:
        input_dataframe (pd.DataFrame): Input dataframe.
        forecast_result (Optional[Dict[str, Any]]): Forecast workflow result.
        anomaly_result (Optional[Dict[str, Any]]): Anomaly workflow result.

    Returns:
        Dict[str, Any]: Combined summary.
    """
    forecast_result = forecast_result or {}
    anomaly_result = anomaly_result or {}

    save_performance_reports()
    performance_summary = get_performance_summary()

    summary = {
        "status": "success",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_summary": get_input_summary(input_dataframe),
        "forecast_summary": forecast_result.get("forecast_summary", {}),
        "anomaly_summary": anomaly_result.get("anomaly_summary", {}),
        "forecast_output_path": forecast_result.get("forecast_output_path"),
        "anomaly_output_path": anomaly_result.get("anomaly_output_path"),
        "performance_summary": performance_summary,
    }

    return summary


# ============================================================
# API Routes
# ============================================================

@app.get("/")
def root() -> Dict[str, Any]:
    """
    Root endpoint.
    """
    return build_success_response(
        message="Financial Portfolio Forecasting & Anomaly Detection API is running.",
        data={
            "project": "Financial Portfolio Forecasting & Anomaly Detection",
            "version": "1.0.0",
            "docs_url": "/docs",
            "health_url": "/health",
            "default_data_path": str(DATA_PATH),
        },
    )


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.
    """
    ensure_api_directories()

    return build_success_response(
        message="Health check completed.",
        data={
            "service": "api.main",
            "project_root": str(PROJECT_ROOT),
            "default_data_available": DATA_PATH.exists(),
            "default_data_path": str(DATA_PATH),
            "model_artifacts": get_model_artifact_status(),
        },
    )


@app.get("/models/status")
def models_status() -> Dict[str, Any]:
    """
    Get trained model artifact status.
    """
    return build_success_response(
        message="Model artifact status loaded.",
        data=get_model_artifact_status(),
    )


@app.post("/forecast")
def forecast(request: ForecastRequest) -> Dict[str, Any]:
    """
    Generate future forecasts from JSON records.
    """
    try:
        ensure_api_directories()

        with track_step("api_main_forecast", category="api"):
            input_dataframe = records_to_dataframe(request.records)

            forecast_result = run_forecast_workflow(
                input_dataframe=input_dataframe,
                model_name=request.model_name,
                forecast_steps=request.forecast_steps,
                save_output=request.save_output,
            )

            summary = build_combined_summary(
                input_dataframe=input_dataframe,
                forecast_result=forecast_result,
            )

            if request.save_output:
                save_api_summary(summary)

        response_data = {
            "input_summary": summary.get("input_summary"),
            "forecast_summary": summary.get("forecast_summary"),
            "saved_path": summary.get("forecast_output_path"),
            "performance_summary": summary.get("performance_summary"),
        }

        if request.return_records:
            response_data["forecast_records"] = dataframe_to_json_records(
                forecast_result["forecast_dataframe"]
            )

        return build_success_response(
            message="Forecast generated successfully.",
            data=response_data,
        )

    except Exception as error:
        logger.error("Forecast endpoint failed: %s", error, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=build_error_response(error),
        )


@app.post("/anomaly")
def anomaly(request: AnomalyRequest) -> Dict[str, Any]:
    """
    Detect anomalies from JSON records.
    """
    try:
        ensure_api_directories()

        with track_step("api_main_anomaly", category="api"):
            input_dataframe = records_to_dataframe(request.records)

            anomaly_result = run_anomaly_workflow(
                input_dataframe=input_dataframe,
                save_output=request.save_output,
            )

            summary = build_combined_summary(
                input_dataframe=input_dataframe,
                anomaly_result=anomaly_result,
            )

            if request.save_output:
                save_api_summary(summary)

        response_data = {
            "input_summary": summary.get("input_summary"),
            "anomaly_summary": summary.get("anomaly_summary"),
            "saved_path": summary.get("anomaly_output_path"),
            "performance_summary": summary.get("performance_summary"),
        }

        if request.return_records:
            filtered_anomalies = filter_anomaly_records(
                anomaly_result["anomaly_dataframe"],
                only_anomalies=request.only_anomalies,
            )

            response_data["anomaly_records"] = dataframe_to_json_records(
                filtered_anomalies
            )

        return build_success_response(
            message="Anomaly detection completed successfully.",
            data=response_data,
        )

    except Exception as error:
        logger.error("Anomaly endpoint failed: %s", error, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=build_error_response(error),
        )


@app.post("/predict")
def predict(request: CombinedPredictionRequest) -> Dict[str, Any]:
    """
    Run combined forecasting and anomaly detection.
    """
    try:
        ensure_api_directories()

        with track_step("api_main_combined_predict", category="api"):
            input_dataframe = records_to_dataframe(request.records)

            forecast_result = {}
            anomaly_result = {}

            if request.run_forecasting:
                forecast_result = run_forecast_workflow(
                    input_dataframe=input_dataframe,
                    model_name=request.model_name,
                    forecast_steps=request.forecast_steps,
                    save_output=request.save_output,
                )

            if request.run_anomaly_detection:
                anomaly_result = run_anomaly_workflow(
                    input_dataframe=input_dataframe,
                    save_output=request.save_output,
                )

            summary = build_combined_summary(
                input_dataframe=input_dataframe,
                forecast_result=forecast_result,
                anomaly_result=anomaly_result,
            )

            if request.save_output:
                save_api_summary(summary)

        response_data = {
            "input_summary": summary.get("input_summary"),
            "forecast_summary": summary.get("forecast_summary"),
            "anomaly_summary": summary.get("anomaly_summary"),
            "forecast_saved_path": summary.get("forecast_output_path"),
            "anomaly_saved_path": summary.get("anomaly_output_path"),
            "performance_summary": summary.get("performance_summary"),
        }

        if request.return_records:
            if request.run_forecasting and forecast_result:
                response_data["forecast_records"] = dataframe_to_json_records(
                    forecast_result["forecast_dataframe"]
                )

            if request.run_anomaly_detection and anomaly_result:
                filtered_anomalies = filter_anomaly_records(
                    anomaly_result["anomaly_dataframe"],
                    only_anomalies=request.only_anomalies,
                )

                response_data["anomaly_records"] = dataframe_to_json_records(
                    filtered_anomalies
                )

        return build_success_response(
            message="Combined prediction completed successfully.",
            data=response_data,
        )

    except Exception as error:
        logger.error("Combined prediction endpoint failed: %s", error, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=build_error_response(error),
        )


@app.get("/forecast/default")
def forecast_default(
    model_name: str = Query(
        default="xgboost",
        description="Forecast model: xgboost, arima, prophet, lstm, or all.",
    ),
    forecast_steps: int = Query(
        default=30,
        ge=1,
        le=365,
        description="Number of future periods to forecast.",
    ),
    save_output: bool = Query(
        default=True,
        description="Whether to save forecast output.",
    ),
    return_records: bool = Query(
        default=True,
        description="Whether to return forecast records.",
    ),
) -> Dict[str, Any]:
    """
    Generate forecast using default dataset.
    """
    try:
        ensure_api_directories()

        with track_step("api_main_forecast_default", category="api"):
            input_dataframe = load_default_dataset()

            forecast_result = run_forecast_workflow(
                input_dataframe=input_dataframe,
                model_name=model_name,
                forecast_steps=forecast_steps,
                save_output=save_output,
            )

            summary = build_combined_summary(
                input_dataframe=input_dataframe,
                forecast_result=forecast_result,
            )

            if save_output:
                save_api_summary(summary)

        response_data = {
            "input_summary": summary.get("input_summary"),
            "forecast_summary": summary.get("forecast_summary"),
            "saved_path": summary.get("forecast_output_path"),
            "performance_summary": summary.get("performance_summary"),
        }

        if return_records:
            response_data["forecast_records"] = dataframe_to_json_records(
                forecast_result["forecast_dataframe"]
            )

        return build_success_response(
            message="Default dataset forecast generated successfully.",
            data=response_data,
        )

    except Exception as error:
        logger.error("Default forecast endpoint failed: %s", error, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=build_error_response(error),
        )


@app.get("/anomaly/default")
def anomaly_default(
    save_output: bool = Query(
        default=True,
        description="Whether to save anomaly output.",
    ),
    return_records: bool = Query(
        default=True,
        description="Whether to return anomaly records.",
    ),
    only_anomalies: bool = Query(
        default=False,
        description="Whether to return only anomaly records.",
    ),
) -> Dict[str, Any]:
    """
    Detect anomalies using default dataset.
    """
    try:
        ensure_api_directories()

        with track_step("api_main_anomaly_default", category="api"):
            input_dataframe = load_default_dataset()

            anomaly_result = run_anomaly_workflow(
                input_dataframe=input_dataframe,
                save_output=save_output,
            )

            summary = build_combined_summary(
                input_dataframe=input_dataframe,
                anomaly_result=anomaly_result,
            )

            if save_output:
                save_api_summary(summary)

        response_data = {
            "input_summary": summary.get("input_summary"),
            "anomaly_summary": summary.get("anomaly_summary"),
            "saved_path": summary.get("anomaly_output_path"),
            "performance_summary": summary.get("performance_summary"),
        }

        if return_records:
            filtered_anomalies = filter_anomaly_records(
                anomaly_result["anomaly_dataframe"],
                only_anomalies=only_anomalies,
            )

            response_data["anomaly_records"] = dataframe_to_json_records(
                filtered_anomalies
            )

        return build_success_response(
            message="Default dataset anomaly detection completed successfully.",
            data=response_data,
        )

    except Exception as error:
        logger.error("Default anomaly endpoint failed: %s", error, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=build_error_response(error),
        )


@app.post("/upload/predict")
async def upload_predict(
    file: UploadFile = File(...),
    model_name: str = Query(
        default="xgboost",
        description="Forecast model: xgboost, arima, prophet, lstm, or all.",
    ),
    forecast_steps: int = Query(
        default=30,
        ge=1,
        le=365,
        description="Number of future periods to forecast.",
    ),
    run_forecasting: bool = Query(
        default=True,
        description="Whether to run forecasting.",
    ),
    run_anomaly_detection: bool = Query(
        default=True,
        description="Whether to run anomaly detection.",
    ),
    save_output: bool = Query(
        default=True,
        description="Whether to save output files.",
    ),
    return_records: bool = Query(
        default=True,
        description="Whether to return records.",
    ),
    only_anomalies: bool = Query(
        default=False,
        description="Whether to return only anomaly records.",
    ),
) -> Dict[str, Any]:
    """
    Upload CSV and run forecasting/anomaly detection.

    CSV must contain one of:

    - date, asset, price
    - Date, Asset, Price
    """
    temp_path = None

    try:
        ensure_api_directories()

        if not file.filename.lower().endswith(".csv"):
            raise ValueError("Only CSV files are supported.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_file.write(await file.read())
            temp_path = Path(temp_file.name)

        with track_step("api_main_upload_predict", category="api"):
            input_dataframe = pd.read_csv(temp_path)
            input_dataframe = clean_input_data(input_dataframe)

            forecast_result = {}
            anomaly_result = {}

            if run_forecasting:
                forecast_result = run_forecast_workflow(
                    input_dataframe=input_dataframe,
                    model_name=model_name,
                    forecast_steps=forecast_steps,
                    save_output=save_output,
                )

            if run_anomaly_detection:
                anomaly_result = run_anomaly_workflow(
                    input_dataframe=input_dataframe,
                    save_output=save_output,
                )

            summary = build_combined_summary(
                input_dataframe=input_dataframe,
                forecast_result=forecast_result,
                anomaly_result=anomaly_result,
            )

            if save_output:
                save_api_summary(summary)

        response_data = {
            "uploaded_file": file.filename,
            "input_summary": summary.get("input_summary"),
            "forecast_summary": summary.get("forecast_summary"),
            "anomaly_summary": summary.get("anomaly_summary"),
            "forecast_saved_path": summary.get("forecast_output_path"),
            "anomaly_saved_path": summary.get("anomaly_output_path"),
            "performance_summary": summary.get("performance_summary"),
        }

        if return_records:
            if run_forecasting and forecast_result:
                response_data["forecast_records"] = dataframe_to_json_records(
                    forecast_result["forecast_dataframe"]
                )

            if run_anomaly_detection and anomaly_result:
                filtered_anomalies = filter_anomaly_records(
                    anomaly_result["anomaly_dataframe"],
                    only_anomalies=only_anomalies,
                )

                response_data["anomaly_records"] = dataframe_to_json_records(
                    filtered_anomalies
                )

        return build_success_response(
            message="Uploaded CSV prediction completed successfully.",
            data=response_data,
        )

    except Exception as error:
        logger.error("Upload predict endpoint failed: %s", error, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=build_error_response(error),
        )

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@app.get("/summary/latest")
def latest_summary() -> Dict[str, Any]:
    """
    Return latest saved API summary.
    """
    try:
        if not API_SUMMARY_JSON_PATH.exists():
            raise FileNotFoundError(
                f"No API summary found at: {API_SUMMARY_JSON_PATH}"
            )

        with API_SUMMARY_JSON_PATH.open("r", encoding="utf-8") as file:
            summary = json.load(file)

        return build_success_response(
            message="Latest API summary loaded successfully.",
            data=summary,
        )

    except Exception as error:
        logger.error("Latest summary endpoint failed: %s", error, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=build_error_response(error),
        )


# ============================================================
# Local Script Test
# ============================================================

def run_local_api_test() -> Dict[str, Any]:
    """
    Run a local API workflow test without starting the server.

    Returns:
        Dict[str, Any]: Test summary.
    """
    try:
        ensure_api_directories()

        input_dataframe = load_default_dataset()

        forecast_result = {}
        anomaly_result = {}

        try:
            forecast_result = run_forecast_workflow(
                input_dataframe=input_dataframe,
                model_name="xgboost",
                forecast_steps=30,
                save_output=True,
            )
        except Exception as forecast_error:
            logger.warning("Local forecast test skipped: %s", forecast_error)

        try:
            anomaly_result = run_anomaly_workflow(
                input_dataframe=input_dataframe,
                save_output=True,
            )
        except Exception as anomaly_error:
            logger.warning("Local anomaly test skipped: %s", anomaly_error)

        summary = build_combined_summary(
            input_dataframe=input_dataframe,
            forecast_result=forecast_result,
            anomaly_result=anomaly_result,
        )

        save_api_summary(summary)

        return summary

    except Exception as error:
        return {
            "status": "failed",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error_type": error.__class__.__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
        }


# ============================================================
# Script Entry Point
# ============================================================

if __name__ == "__main__":
    result = run_local_api_test()

    print("\n" + "=" * 80)
    print("API LOCAL TEST SUMMARY")
    print("=" * 80)

    print("Status:", result.get("status"))

    if result.get("status") == "success":
        input_summary = result.get("input_summary", {})
        forecast_summary = result.get("forecast_summary", {})
        anomaly_summary = result.get("anomaly_summary", {})

        print("\nInput Summary:")
        print("- Total Records:", input_summary.get("total_records"))
        print("- Total Assets:", input_summary.get("total_assets"))
        print("- Start Date:", input_summary.get("start_date"))
        print("- End Date:", input_summary.get("end_date"))

        print("\nForecast Summary:")
        print("- Forecast Rows:", forecast_summary.get("forecast_rows"))
        print("- Models Used:", forecast_summary.get("models_used"))
        print("- Assets Forecasted:", forecast_summary.get("assets_forecasted"))

        print("\nAnomaly Summary:")
        print("- Total Records:", anomaly_summary.get("total_records"))
        print("- Total Anomalies:", anomaly_summary.get("total_anomalies"))
        print("- Anomaly Rate %:", anomaly_summary.get("anomaly_rate_percent"))

        print("\nSaved Files:")
        print("- outputs/api_forecast_results.csv")
        print("- outputs/api_anomaly_results.csv")
        print("- reports/tables/api_summary.json")
        print("- reports/api_summary.md")

    else:
        print("\nAPI local test failed.")
        print("Error Type:", result.get("error_type"))
        print("Error Message:", result.get("error_message"))

    print("=" * 80)