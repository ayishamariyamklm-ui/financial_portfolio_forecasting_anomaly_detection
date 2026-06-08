"""
Forecast API routes for the Financial Portfolio Forecasting
and Anomaly Detection project.

This module contains only forecasting-related routes.

Routes included:
- GET  /forecast/health
- GET  /forecast/model-status
- POST /forecast/predict
- GET  /forecast/default
- POST /forecast/upload
- GET  /forecast/latest

Dataset expected columns:
- date
- asset
- price

This file is designed to be included in api/main.py using:

    from api.routes.forecast_routes import router as forecast_router
    app.include_router(forecast_router)

Run API server from project root:

    uvicorn api.main:app --reload
"""

import json
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
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
    clean_input_data,
    generate_forecasts,
    get_input_summary,
    summarize_forecast_results,
)
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
# Router
# ============================================================

router = APIRouter(
    prefix="/forecast",
    tags=["Forecasting"],
)


# ============================================================
# Paths
# ============================================================

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "financial_portfolio_data.csv"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
LOGS_DIR = PROJECT_ROOT / "logs"

API_FORECAST_OUTPUT_PATH = OUTPUTS_DIR / "api_forecast_results.csv"
API_FORECAST_SUMMARY_JSON_PATH = TABLES_DIR / "api_forecast_summary.json"
API_FORECAST_SUMMARY_MD_PATH = REPORTS_DIR / "api_forecast_summary.md"


# ============================================================
# Logger
# ============================================================

logger = get_project_logger()


# ============================================================
# Request Schemas
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
        description="Historical asset price records.",
    )
    model_name: str = Field(
        default="xgboost",
        description="Forecasting model: xgboost, arima, prophet, lstm, or all.",
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
    return_records: bool = Field(
        default=True,
        description="Whether to return forecast records in API response.",
    )


# ============================================================
# Directory Helpers
# ============================================================

def ensure_forecast_route_directories() -> None:
    """
    Create all required directories for forecast route outputs.
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
    Convert request records into cleaned dataframe.

    Args:
        records (List[PriceRecord]): List of asset price records.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    if not records:
        raise ValueError("No records provided.")

    dataframe = pd.DataFrame([record.model_dump() for record in records])

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
    Convert dataframe into JSON-safe records.

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


# ============================================================
# Status Helpers
# ============================================================

def get_forecast_model_status() -> Dict[str, Any]:
    """
    Get forecasting model artifact status.

    Returns:
        Dict[str, Any]: Forecasting model status.
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
    }


def validate_model_name(model_name: str) -> str:
    """
    Validate forecasting model name.

    Args:
        model_name (str): Model name.

    Returns:
        str: Cleaned model name.
    """
    allowed_models = {"xgboost", "arima", "prophet", "lstm", "all"}

    model_name = model_name.strip().lower()

    if model_name not in allowed_models:
        raise ValueError(
            f"Invalid model_name '{model_name}'. "
            f"Allowed values are: {sorted(allowed_models)}"
        )

    return model_name


# ============================================================
# Response Helpers
# ============================================================

def build_success_response(
    message: str,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build standard success response.

    Args:
        message (str): Success message.
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
        "traceback": traceback.format_exc(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ============================================================
# Save Helpers
# ============================================================

def save_forecast_results(
    forecast_dataframe: pd.DataFrame,
    file_path: Path = API_FORECAST_OUTPUT_PATH,
) -> None:
    """
    Save forecast results as CSV.

    Args:
        forecast_dataframe (pd.DataFrame): Forecast dataframe.
        file_path (Path): Output CSV path.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    forecast_dataframe.to_csv(file_path, index=False)

    logger.info("API forecast results saved at: %s", file_path)


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

    logger.info("API forecast summary JSON saved at: %s", file_path)


def save_markdown_summary(summary: Dict[str, Any], file_path: Path) -> None:
    """
    Save forecast summary as Markdown.

    Args:
        summary (Dict[str, Any]): Summary dictionary.
        file_path (Path): Output Markdown path.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    input_summary = summary.get("input_summary", {})
    forecast_summary = summary.get("forecast_summary", {})
    performance_summary = summary.get("performance_summary", {})

    lines = [
        "# API Forecast Summary",
        "",
        "## Project",
        "",
        "Financial Portfolio Forecasting & Anomaly Detection",
        "",
        "## Run Information",
        "",
        f"- Generated At: {summary.get('generated_at')}",
        f"- Status: {summary.get('status')}",
        f"- Model Name: {summary.get('model_name')}",
        f"- Forecast Steps: {summary.get('forecast_steps')}",
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
        f"- Minimum Forecasted Price: {forecast_summary.get('minimum_forecasted_price')}",
        f"- Maximum Forecasted Price: {forecast_summary.get('maximum_forecasted_price')}",
        "",
        "## Output Files",
        "",
        f"- Forecast CSV: `{summary.get('forecast_output_path')}`",
        f"- Summary JSON: `{API_FORECAST_SUMMARY_JSON_PATH}`",
        f"- Summary Markdown: `{API_FORECAST_SUMMARY_MD_PATH}`",
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
        "- Forecasting predicts future asset prices based on trained models.",
        "- XGBoost uses engineered time series features.",
        "- ARIMA is a statistical time series model.",
        "- Prophet models trend and seasonality.",
        "- LSTM is a deep learning sequence model.",
    ]

    file_path.write_text("\n".join(lines), encoding="utf-8")

    logger.info("API forecast summary Markdown saved at: %s", file_path)


def save_forecast_summary(summary: Dict[str, Any]) -> None:
    """
    Save forecast summary as JSON and Markdown.

    Args:
        summary (Dict[str, Any]): Forecast summary.
    """
    save_json_summary(summary, API_FORECAST_SUMMARY_JSON_PATH)
    save_markdown_summary(summary, API_FORECAST_SUMMARY_MD_PATH)


# ============================================================
# Core Forecast Workflow
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
        input_dataframe (pd.DataFrame): Input historical dataframe.
        model_name (str): Forecasting model name.
        forecast_steps (int): Forecast horizon.
        save_output (bool): Whether to save output files.

    Returns:
        Dict[str, Any]: Forecast workflow result.
    """
    model_name = validate_model_name(model_name)

    forecast_dataframe = generate_forecasts(
        dataframe=input_dataframe,
        model_name=model_name,
        forecast_steps=forecast_steps,
    )

    if save_output:
        save_forecast_results(forecast_dataframe)

    save_performance_reports()

    performance_summary = get_performance_summary()

    summary = {
        "status": "success",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "forecast_steps": forecast_steps,
        "input_summary": get_input_summary(input_dataframe),
        "forecast_summary": summarize_forecast_results(forecast_dataframe),
        "forecast_output_path": str(API_FORECAST_OUTPUT_PATH) if save_output else None,
        "summary_json_path": str(API_FORECAST_SUMMARY_JSON_PATH) if save_output else None,
        "summary_markdown_path": str(API_FORECAST_SUMMARY_MD_PATH) if save_output else None,
        "performance_summary": performance_summary,
    }

    if save_output:
        save_forecast_summary(summary)

    return {
        "forecast_dataframe": forecast_dataframe,
        "summary": summary,
    }


# ============================================================
# Routes
# ============================================================

@router.get("/health")
def forecast_health() -> Dict[str, Any]:
    """
    Forecast route health check.
    """
    ensure_forecast_route_directories()

    return build_success_response(
        message="Forecast routes are working.",
        data={
            "service": "forecast_routes",
            "project_root": str(PROJECT_ROOT),
            "default_data_available": DATA_PATH.exists(),
            "default_data_path": str(DATA_PATH),
            "model_status": get_forecast_model_status(),
        },
    )


@router.get("/model-status")
def forecast_model_status() -> Dict[str, Any]:
    """
    Return forecasting model artifact status.
    """
    return build_success_response(
        message="Forecast model status loaded.",
        data=get_forecast_model_status(),
    )


@router.post("/predict")
def forecast_predict(request: ForecastRequest) -> Dict[str, Any]:
    """
    Generate future forecasts from JSON records.
    """
    try:
        ensure_forecast_route_directories()

        with track_step("forecast_route_predict", category="api"):
            input_dataframe = records_to_dataframe(request.records)

            result = run_forecast_workflow(
                input_dataframe=input_dataframe,
                model_name=request.model_name,
                forecast_steps=request.forecast_steps,
                save_output=request.save_output,
            )

        forecast_dataframe = result["forecast_dataframe"]
        summary = result["summary"]

        response_data = {
            "input_summary": summary.get("input_summary"),
            "forecast_summary": summary.get("forecast_summary"),
            "saved_path": summary.get("forecast_output_path"),
            "summary_json_path": summary.get("summary_json_path"),
            "summary_markdown_path": summary.get("summary_markdown_path"),
            "performance_summary": summary.get("performance_summary"),
        }

        if request.return_records:
            response_data["forecast_records"] = dataframe_to_json_records(
                forecast_dataframe,
            )

        return build_success_response(
            message="Forecast generated successfully.",
            data=response_data,
        )

    except Exception as error:
        logger.error("Forecast predict route failed: %s", error, exc_info=True)

        raise HTTPException(
            status_code=500,
            detail=build_error_response(error),
        )


@router.get("/default")
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
    Generate forecasts using the default dataset from data/raw.
    """
    try:
        ensure_forecast_route_directories()

        with track_step("forecast_route_default", category="api"):
            input_dataframe = load_default_dataset()

            result = run_forecast_workflow(
                input_dataframe=input_dataframe,
                model_name=model_name,
                forecast_steps=forecast_steps,
                save_output=save_output,
            )

        forecast_dataframe = result["forecast_dataframe"]
        summary = result["summary"]

        response_data = {
            "input_summary": summary.get("input_summary"),
            "forecast_summary": summary.get("forecast_summary"),
            "saved_path": summary.get("forecast_output_path"),
            "summary_json_path": summary.get("summary_json_path"),
            "summary_markdown_path": summary.get("summary_markdown_path"),
            "performance_summary": summary.get("performance_summary"),
        }

        if return_records:
            response_data["forecast_records"] = dataframe_to_json_records(
                forecast_dataframe,
            )

        return build_success_response(
            message="Default dataset forecast generated successfully.",
            data=response_data,
        )

    except Exception as error:
        logger.error("Forecast default route failed: %s", error, exc_info=True)

        raise HTTPException(
            status_code=500,
            detail=build_error_response(error),
        )


@router.post("/upload")
async def forecast_upload(
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
        description="Whether to save forecast output.",
    ),
    return_records: bool = Query(
        default=True,
        description="Whether to return forecast records.",
    ),
) -> Dict[str, Any]:
    """
    Upload CSV file and generate future forecasts.

    CSV must contain:

    - date, asset, price

    or:

    - Date, Asset, Price
    """
    temp_path = None

    try:
        ensure_forecast_route_directories()

        if not file.filename.lower().endswith(".csv"):
            raise ValueError("Only CSV files are supported.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_file.write(await file.read())
            temp_path = Path(temp_file.name)

        with track_step("forecast_route_upload", category="api"):
            uploaded_dataframe = pd.read_csv(temp_path)
            input_dataframe = clean_input_data(uploaded_dataframe)

            result = run_forecast_workflow(
                input_dataframe=input_dataframe,
                model_name=model_name,
                forecast_steps=forecast_steps,
                save_output=save_output,
            )

        forecast_dataframe = result["forecast_dataframe"]
        summary = result["summary"]

        response_data = {
            "uploaded_file": file.filename,
            "input_summary": summary.get("input_summary"),
            "forecast_summary": summary.get("forecast_summary"),
            "saved_path": summary.get("forecast_output_path"),
            "summary_json_path": summary.get("summary_json_path"),
            "summary_markdown_path": summary.get("summary_markdown_path"),
            "performance_summary": summary.get("performance_summary"),
        }

        if return_records:
            response_data["forecast_records"] = dataframe_to_json_records(
                forecast_dataframe,
            )

        return build_success_response(
            message="Uploaded CSV forecast generated successfully.",
            data=response_data,
        )

    except Exception as error:
        logger.error("Forecast upload route failed: %s", error, exc_info=True)

        raise HTTPException(
            status_code=500,
            detail=build_error_response(error),
        )

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@router.get("/latest")
def latest_forecast_summary() -> Dict[str, Any]:
    """
    Return latest saved forecast summary.
    """
    try:
        if not API_FORECAST_SUMMARY_JSON_PATH.exists():
            raise FileNotFoundError(
                f"No forecast summary found at: {API_FORECAST_SUMMARY_JSON_PATH}"
            )

        with API_FORECAST_SUMMARY_JSON_PATH.open("r", encoding="utf-8") as file:
            summary = json.load(file)

        return build_success_response(
            message="Latest forecast summary loaded successfully.",
            data=summary,
        )

    except Exception as error:
        logger.error("Latest forecast summary route failed: %s", error, exc_info=True)

        raise HTTPException(
            status_code=500,
            detail=build_error_response(error),
        )


# ============================================================
# Local Test Runner
# ============================================================

def run_local_forecast_route_test() -> Dict[str, Any]:
    """
    Run local forecast route test without starting the API server.

    Returns:
        Dict[str, Any]: Local test summary.
    """
    try:
        ensure_forecast_route_directories()

        input_dataframe = load_default_dataset()

        result = run_forecast_workflow(
            input_dataframe=input_dataframe,
            model_name="xgboost",
            forecast_steps=30,
            save_output=True,
        )

        return result["summary"]

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
    local_result = run_local_forecast_route_test()

    print("\n" + "=" * 80)
    print("FORECAST ROUTES LOCAL TEST SUMMARY")
    print("=" * 80)

    print("Status:", local_result.get("status"))

    if local_result.get("status") == "success":
        input_summary = local_result.get("input_summary", {})
        forecast_summary = local_result.get("forecast_summary", {})

        print("\nInput Summary:")
        print("- Total Records:", input_summary.get("total_records"))
        print("- Total Assets:", input_summary.get("total_assets"))
        print("- Start Date:", input_summary.get("start_date"))
        print("- End Date:", input_summary.get("end_date"))

        print("\nForecast Summary:")
        print("- Forecast Rows:", forecast_summary.get("forecast_rows"))
        print("- Models Used:", forecast_summary.get("models_used"))
        print("- Assets Forecasted:", forecast_summary.get("assets_forecasted"))
        print("- Forecast Start Date:", forecast_summary.get("forecast_start_date"))
        print("- Forecast End Date:", forecast_summary.get("forecast_end_date"))

        print("\nSaved Files:")
        print("- outputs/api_forecast_results.csv")
        print("- reports/tables/api_forecast_summary.json")
        print("- reports/api_forecast_summary.md")

    else:
        print("\nForecast route local test failed.")
        print("Error Type:", local_result.get("error_type"))
        print("Error Message:", local_result.get("error_message"))

    print("=" * 80)