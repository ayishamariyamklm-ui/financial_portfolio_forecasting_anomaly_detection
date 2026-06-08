"""
Health API routes for the Financial Portfolio Forecasting
and Anomaly Detection project.

This module contains health, status, and project information routes.

Routes included:
- GET /health/
- GET /health/ping
- GET /health/project
- GET /health/system
- GET /health/data
- GET /health/models
- GET /health/artifacts
- GET /health/full

Dataset expected path:
- data/raw/financial_portfolio_data.csv

This file is designed to be included in api/main.py using:

    from api.routes.health_routes import router as health_router
    app.include_router(health_router)

Run API server from project root:

    uvicorn api.main:app --reload
"""

import json
import platform
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException


# ============================================================
# Add Project Root to Python Path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ============================================================
# Project Imports
# ============================================================

from src.models.anomaly_detector import ANOMALY_MODEL_PATH
from src.models.arima_model import ARIMA_MODEL_PATH
from src.models.lstm_model import LSTM_MODEL_DIR, LSTM_SCALER_PATH
from src.models.prophet_model import PROPHET_MODEL_PATH
from src.models.xgboost_model import XGBOOST_MODEL_PATH
from src.monitoring.logger import get_project_logger
from src.monitoring.performance_tracker import get_performance_summary
from src.utils.paths import create_required_directories


# ============================================================
# Router
# ============================================================

router = APIRouter(
    prefix="/health",
    tags=["Health"],
)


# ============================================================
# Paths
# ============================================================

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "financial_portfolio_data.csv"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"
API_DIR = PROJECT_ROOT / "api"

API_SUMMARY_JSON_PATH = TABLES_DIR / "api_summary.json"
FORECAST_SUMMARY_JSON_PATH = TABLES_DIR / "api_forecast_summary.json"
ANOMALY_SUMMARY_JSON_PATH = TABLES_DIR / "api_anomaly_summary.json"
EVALUATION_SUMMARY_JSON_PATH = TABLES_DIR / "evaluation_summary.json"
ALL_MODELS_SUMMARY_JSON_PATH = TABLES_DIR / "all_models_training_summary.json"


# ============================================================
# Logger
# ============================================================

logger = get_project_logger()


# ============================================================
# Directory Helpers
# ============================================================

def ensure_health_route_directories() -> None:
    """
    Create all core project directories used by the API.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

    create_required_directories()


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
        Dict[str, Any]: API error response.
    """
    return {
        "status": "failed",
        "error_type": error.__class__.__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ============================================================
# File and Directory Status Helpers
# ============================================================

def get_file_status(file_path: Path) -> Dict[str, Any]:
    """
    Return file existence, size, and modified time.

    Args:
        file_path (Path): File path.

    Returns:
        Dict[str, Any]: File status.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return {
            "exists": False,
            "path": str(file_path),
            "size_bytes": None,
            "modified_at": None,
        }

    return {
        "exists": True,
        "path": str(file_path),
        "size_bytes": int(file_path.stat().st_size),
        "modified_at": datetime.fromtimestamp(
            file_path.stat().st_mtime
        ).strftime("%Y-%m-%d %H:%M:%S"),
    }


def get_directory_status(directory_path: Path) -> Dict[str, Any]:
    """
    Return directory existence and item count.

    Args:
        directory_path (Path): Directory path.

    Returns:
        Dict[str, Any]: Directory status.
    """
    directory_path = Path(directory_path)

    if not directory_path.exists():
        return {
            "exists": False,
            "path": str(directory_path),
            "file_count": 0,
            "directory_count": 0,
        }

    files = [path for path in directory_path.iterdir() if path.is_file()]
    directories = [path for path in directory_path.iterdir() if path.is_dir()]

    return {
        "exists": True,
        "path": str(directory_path),
        "file_count": len(files),
        "directory_count": len(directories),
    }


def safe_load_json(file_path: Path) -> Dict[str, Any]:
    """
    Safely load JSON file.

    Args:
        file_path (Path): JSON file path.

    Returns:
        Dict[str, Any]: Loaded JSON data or empty dictionary.
    """
    if not file_path.exists():
        return {}

    try:
        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        return data if isinstance(data, dict) else {}

    except Exception:
        return {}


# ============================================================
# Project Status Helpers
# ============================================================

def get_project_info() -> Dict[str, Any]:
    """
    Return project metadata.

    Returns:
        Dict[str, Any]: Project information.
    """
    return {
        "project_name": "Financial Portfolio Forecasting & Anomaly Detection",
        "project_type": "Month 5 Practical",
        "description": (
            "End-to-end financial time series project for forecasting asset "
            "prices and detecting anomalies using statistical, machine learning, "
            "and deep learning methods."
        ),
        "expected_dataset_columns": ["date", "asset", "price"],
        "default_dataset_path": str(DATA_PATH),
        "project_root": str(PROJECT_ROOT),
        "api_version": "1.0.0",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
    }


def get_system_info() -> Dict[str, Any]:
    """
    Return system and Python runtime information.

    Returns:
        Dict[str, Any]: System information.
    """
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "operating_system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "current_working_directory": str(Path.cwd()),
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def get_directory_overview() -> Dict[str, Any]:
    """
    Return core directory status.

    Returns:
        Dict[str, Any]: Directory overview.
    """
    return {
        "data": get_directory_status(DATA_DIR),
        "data_raw": get_directory_status(RAW_DATA_DIR),
        "data_processed": get_directory_status(PROCESSED_DATA_DIR),
        "data_sample": get_directory_status(SAMPLE_DATA_DIR),
        "models": get_directory_status(MODELS_DIR),
        "outputs": get_directory_status(OUTPUTS_DIR),
        "reports": get_directory_status(REPORTS_DIR),
        "reports_tables": get_directory_status(TABLES_DIR),
        "reports_figures": get_directory_status(FIGURES_DIR),
        "logs": get_directory_status(LOGS_DIR),
        "config": get_directory_status(CONFIG_DIR),
        "notebooks": get_directory_status(NOTEBOOKS_DIR),
        "src": get_directory_status(SRC_DIR),
        "api": get_directory_status(API_DIR),
    }


def get_dataset_status() -> Dict[str, Any]:
    """
    Return dataset availability and basic validation information.

    Returns:
        Dict[str, Any]: Dataset status.
    """
    file_status = get_file_status(DATA_PATH)

    dataset_status = {
        "file": file_status,
        "available": file_status["exists"],
        "valid": False,
        "required_columns": ["date", "asset", "price"],
        "columns": [],
        "shape": None,
        "total_records": None,
        "total_assets": None,
        "date_range": {
            "start_date": None,
            "end_date": None,
        },
        "price_summary": {
            "average_price": None,
            "minimum_price": None,
            "maximum_price": None,
        },
        "message": None,
    }

    if not DATA_PATH.exists():
        dataset_status["message"] = (
            "Dataset not found. Place financial_portfolio_data.csv inside data/raw/."
        )
        return dataset_status

    try:
        dataframe = pd.read_csv(DATA_PATH)

        dataframe.columns = (
            dataframe.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
            .str.replace("-", "_", regex=False)
        )

        dataset_status["columns"] = dataframe.columns.tolist()
        dataset_status["shape"] = list(dataframe.shape)

        required_columns = ["date", "asset", "price"]

        missing_columns = [
            column for column in required_columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            dataset_status["valid"] = False
            dataset_status["message"] = f"Missing required columns: {missing_columns}"
            return dataset_status

        dataframe["date"] = pd.to_datetime(dataframe["date"], errors="coerce")
        dataframe["price"] = pd.to_numeric(dataframe["price"], errors="coerce")

        dataframe = dataframe.dropna(subset=["date", "asset", "price"])
        dataframe = dataframe[dataframe["price"] >= 0]

        dataset_status["valid"] = not dataframe.empty
        dataset_status["total_records"] = int(len(dataframe))
        dataset_status["total_assets"] = int(dataframe["asset"].nunique())
        dataset_status["date_range"] = {
            "start_date": str(dataframe["date"].min()),
            "end_date": str(dataframe["date"].max()),
        }
        dataset_status["price_summary"] = {
            "average_price": round(float(dataframe["price"].mean()), 4),
            "minimum_price": round(float(dataframe["price"].min()), 4),
            "maximum_price": round(float(dataframe["price"].max()), 4),
        }
        dataset_status["message"] = "Dataset is available and valid."

        return dataset_status

    except Exception as error:
        dataset_status["valid"] = False
        dataset_status["message"] = str(error)
        return dataset_status


def get_model_status() -> Dict[str, Any]:
    """
    Return trained model artifact status.

    Returns:
        Dict[str, Any]: Model status.
    """
    return {
        "forecasting_models": {
            "xgboost": {
                "available": XGBOOST_MODEL_PATH.exists(),
                "artifact": get_file_status(XGBOOST_MODEL_PATH),
            },
            "arima": {
                "available": ARIMA_MODEL_PATH.exists(),
                "artifact": get_file_status(ARIMA_MODEL_PATH),
            },
            "prophet": {
                "available": PROPHET_MODEL_PATH.exists(),
                "artifact": get_file_status(PROPHET_MODEL_PATH),
            },
            "lstm": {
                "available": LSTM_MODEL_DIR.exists() and LSTM_SCALER_PATH.exists(),
                "model_directory": get_directory_status(LSTM_MODEL_DIR),
                "scaler_artifact": get_file_status(LSTM_SCALER_PATH),
            },
        },
        "anomaly_detection": {
            "anomaly_detector": {
                "available": ANOMALY_MODEL_PATH.exists(),
                "artifact": get_file_status(ANOMALY_MODEL_PATH),
            },
        },
    }


def get_output_artifact_status() -> Dict[str, Any]:
    """
    Return output and report artifact status.

    Returns:
        Dict[str, Any]: Output artifact status.
    """
    output_files = {
        "api_summary": API_SUMMARY_JSON_PATH,
        "forecast_summary": FORECAST_SUMMARY_JSON_PATH,
        "anomaly_summary": ANOMALY_SUMMARY_JSON_PATH,
        "evaluation_summary": EVALUATION_SUMMARY_JSON_PATH,
        "all_models_training_summary": ALL_MODELS_SUMMARY_JSON_PATH,
        "api_forecast_results": OUTPUTS_DIR / "api_forecast_results.csv",
        "api_anomaly_results": OUTPUTS_DIR / "api_anomaly_results.csv",
        "inference_forecasts": OUTPUTS_DIR / "inference_forecasts.csv",
        "inference_anomaly_results": OUTPUTS_DIR / "inference_anomaly_results.csv",
        "forecasting_model_comparison": TABLES_DIR / "forecasting_model_comparison.csv",
        "model_comparison": TABLES_DIR / "model_comparison.csv",
        "anomaly_detection_results": OUTPUTS_DIR / "anomaly_detection_results.csv",
    }

    return {
        name: get_file_status(path)
        for name, path in output_files.items()
    }


def get_latest_summaries() -> Dict[str, Any]:
    """
    Return latest saved summary JSON contents if available.

    Returns:
        Dict[str, Any]: Latest summaries.
    """
    return {
        "api_summary": safe_load_json(API_SUMMARY_JSON_PATH),
        "forecast_summary": safe_load_json(FORECAST_SUMMARY_JSON_PATH),
        "anomaly_summary": safe_load_json(ANOMALY_SUMMARY_JSON_PATH),
        "evaluation_summary": safe_load_json(EVALUATION_SUMMARY_JSON_PATH),
        "all_models_training_summary": safe_load_json(ALL_MODELS_SUMMARY_JSON_PATH),
    }


def calculate_overall_health_status() -> Dict[str, Any]:
    """
    Calculate overall health status.

    Returns:
        Dict[str, Any]: Overall health summary.
    """
    dataset_status = get_dataset_status()
    model_status = get_model_status()

    xgboost_available = model_status["forecasting_models"]["xgboost"]["available"]
    arima_available = model_status["forecasting_models"]["arima"]["available"]
    prophet_available = model_status["forecasting_models"]["prophet"]["available"]
    lstm_available = model_status["forecasting_models"]["lstm"]["available"]
    anomaly_available = model_status["anomaly_detection"]["anomaly_detector"]["available"]

    forecasting_available = any(
        [
            xgboost_available,
            arima_available,
            prophet_available,
            lstm_available,
        ]
    )

    required_directories = get_directory_overview()
    missing_directories = [
        name for name, status in required_directories.items()
        if not status.get("exists")
    ]

    issues = []

    if not dataset_status.get("available"):
        issues.append("Dataset is missing.")

    if dataset_status.get("available") and not dataset_status.get("valid"):
        issues.append("Dataset exists but is not valid.")

    if not forecasting_available:
        issues.append("No trained forecasting model artifacts found.")

    if not anomaly_available:
        issues.append("Anomaly detection model artifact is missing.")

    if missing_directories:
        issues.append(f"Missing directories: {missing_directories}")

    if not issues:
        overall_status = "healthy"
    elif dataset_status.get("valid") and (forecasting_available or anomaly_available):
        overall_status = "partially_healthy"
    else:
        overall_status = "not_ready"

    return {
        "overall_status": overall_status,
        "dataset_available": dataset_status.get("available"),
        "dataset_valid": dataset_status.get("valid"),
        "forecasting_available": forecasting_available,
        "anomaly_detector_available": anomaly_available,
        "issues": issues,
    }


# ============================================================
# Routes
# ============================================================

@router.get("/")
def health_root() -> Dict[str, Any]:
    """
    Main health endpoint.
    """
    try:
        ensure_health_route_directories()

        health_status = calculate_overall_health_status()

        return build_success_response(
            message="Health check completed.",
            data=health_status,
        )

    except Exception as error:
        logger.error("Health root route failed: %s", error, exc_info=True)

        raise HTTPException(
            status_code=500,
            detail=build_error_response(error),
        )


@router.get("/ping")
def ping() -> Dict[str, Any]:
    """
    Lightweight ping endpoint.
    """
    return build_success_response(
        message="pong",
        data={
            "service": "health_routes",
            "project": "Financial Portfolio Forecasting & Anomaly Detection",
        },
    )


@router.get("/project")
def project_info() -> Dict[str, Any]:
    """
    Return project information.
    """
    try:
        return build_success_response(
            message="Project information loaded.",
            data=get_project_info(),
        )

    except Exception as error:
        logger.error("Project info route failed: %s", error, exc_info=True)

        raise HTTPException(
            status_code=500,
            detail=build_error_response(error),
        )


@router.get("/system")
def system_info() -> Dict[str, Any]:
    """
    Return system information.
    """
    try:
        return build_success_response(
            message="System information loaded.",
            data=get_system_info(),
        )

    except Exception as error:
        logger.error("System info route failed: %s", error, exc_info=True)

        raise HTTPException(
            status_code=500,
            detail=build_error_response(error),
        )


@router.get("/directories")
def directories_status() -> Dict[str, Any]:
    """
    Return project directory status.
    """
    try:
        ensure_health_route_directories()

        return build_success_response(
            message="Directory status loaded.",
            data=get_directory_overview(),
        )

    except Exception as error:
        logger.error("Directories status route failed: %s", error, exc_info=True)

        raise HTTPException(
            status_code=500,
            detail=build_error_response(error),
        )


@router.get("/data")
def data_status() -> Dict[str, Any]:
    """
    Return dataset status and validation information.
    """
    try:
        return build_success_response(
            message="Dataset status loaded.",
            data=get_dataset_status(),
        )

    except Exception as error:
        logger.error("Dataset status route failed: %s", error, exc_info=True)

        raise HTTPException(
            status_code=500,
            detail=build_error_response(error),
        )


@router.get("/models")
def models_status() -> Dict[str, Any]:
    """
    Return trained model artifact status.
    """
    try:
        return build_success_response(
            message="Model status loaded.",
            data=get_model_status(),
        )

    except Exception as error:
        logger.error("Models status route failed: %s", error, exc_info=True)

        raise HTTPException(
            status_code=500,
            detail=build_error_response(error),
        )


@router.get("/artifacts")
def artifacts_status() -> Dict[str, Any]:
    """
    Return generated output artifact status.
    """
    try:
        return build_success_response(
            message="Output artifact status loaded.",
            data=get_output_artifact_status(),
        )

    except Exception as error:
        logger.error("Artifacts status route failed: %s", error, exc_info=True)

        raise HTTPException(
            status_code=500,
            detail=build_error_response(error),
        )


@router.get("/performance")
def performance_status() -> Dict[str, Any]:
    """
    Return latest performance tracker summary.
    """
    try:
        return build_success_response(
            message="Performance summary loaded.",
            data=get_performance_summary(),
        )

    except Exception as error:
        logger.error("Performance status route failed: %s", error, exc_info=True)

        raise HTTPException(
            status_code=500,
            detail=build_error_response(error),
        )


@router.get("/summaries")
def latest_summaries() -> Dict[str, Any]:
    """
    Return latest saved JSON summaries.
    """
    try:
        return build_success_response(
            message="Latest summaries loaded.",
            data=get_latest_summaries(),
        )

    except Exception as error:
        logger.error("Latest summaries route failed: %s", error, exc_info=True)

        raise HTTPException(
            status_code=500,
            detail=build_error_response(error),
        )


@router.get("/full")
def full_health_check() -> Dict[str, Any]:
    """
    Return complete health report.
    """
    try:
        ensure_health_route_directories()

        full_report = {
            "health": calculate_overall_health_status(),
            "project": get_project_info(),
            "system": get_system_info(),
            "directories": get_directory_overview(),
            "dataset": get_dataset_status(),
            "models": get_model_status(),
            "artifacts": get_output_artifact_status(),
            "performance": get_performance_summary(),
            "latest_summaries": get_latest_summaries(),
        }

        return build_success_response(
            message="Full health check completed.",
            data=full_report,
        )

    except Exception as error:
        logger.error("Full health check route failed: %s", error, exc_info=True)

        raise HTTPException(
            status_code=500,
            detail=build_error_response(error),
        )


# ============================================================
# Local Test Runner
# ============================================================

def run_local_health_route_test() -> Dict[str, Any]:
    """
    Run local health route test without starting API server.

    Returns:
        Dict[str, Any]: Health summary.
    """
    try:
        ensure_health_route_directories()

        return {
            "status": "success",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "health": calculate_overall_health_status(),
            "project": get_project_info(),
            "dataset": get_dataset_status(),
            "models": get_model_status(),
            "artifacts": get_output_artifact_status(),
        }

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
    local_result = run_local_health_route_test()

    print("\n" + "=" * 80)
    print("HEALTH ROUTES LOCAL TEST SUMMARY")
    print("=" * 80)

    print("Status:", local_result.get("status"))

    if local_result.get("status") == "success":
        health = local_result.get("health", {})
        dataset = local_result.get("dataset", {})
        models = local_result.get("models", {})

        print("\nOverall Health:")
        print("- Overall Status:", health.get("overall_status"))
        print("- Dataset Available:", health.get("dataset_available"))
        print("- Dataset Valid:", health.get("dataset_valid"))
        print("- Forecasting Available:", health.get("forecasting_available"))
        print("- Anomaly Detector Available:", health.get("anomaly_detector_available"))

        print("\nIssues:")
        issues = health.get("issues", [])
        if issues:
            for issue in issues:
                print(f"- {issue}")
        else:
            print("- No issues found.")

        print("\nDataset:")
        print("- Available:", dataset.get("available"))
        print("- Valid:", dataset.get("valid"))
        print("- Total Records:", dataset.get("total_records"))
        print("- Total Assets:", dataset.get("total_assets"))

        print("\nModel Availability:")
        forecasting_models = models.get("forecasting_models", {})
        anomaly_models = models.get("anomaly_detection", {})

        for model_name, status in forecasting_models.items():
            print(f"- {model_name}: {status.get('available')}")

        for model_name, status in anomaly_models.items():
            print(f"- {model_name}: {status.get('available')}")

    else:
        print("\nHealth route local test failed.")
        print("Error Type:", local_result.get("error_type"))
        print("Error Message:", local_result.get("error_message"))

    print("=" * 80)