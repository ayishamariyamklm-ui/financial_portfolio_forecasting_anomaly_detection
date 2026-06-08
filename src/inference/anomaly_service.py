"""
Anomaly detection service for the Financial Portfolio Forecasting
and Anomaly Detection project.

This module provides a FastAPI service for:

1. Health check
2. Anomaly detection using saved model
3. Default dataset anomaly detection
4. CSV upload anomaly detection
5. Anomaly summary generation
6. API output saving

Dataset expected columns:
- date
- asset
- price

Run from project root:

    uvicorn src.inference.anomaly_service:app --reload
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
    clean_input_data,
    detect_anomalies_with_saved_model,
    get_input_summary,
    save_anomaly_results,
    summarize_anomaly_results,
)
from src.models.anomaly_detector import ANOMALY_MODEL_PATH
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

API_ANOMALY_OUTPUT_PATH = OUTPUTS_DIR / "api_anomaly_results.csv"
API_ANOMALY_SUMMARY_JSON_PATH = TABLES_DIR / "api_anomaly_summary.json"
API_ANOMALY_SUMMARY_MD_PATH = REPORTS_DIR / "api_anomaly_summary.md"


# ============================================================
# Logger
# ============================================================

logger = get_project_logger()


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="Financial Portfolio Anomaly Detection API",
    description=(
        "API service for detecting unusual financial asset price movements "
        "using IQR, Z-score, and Isolation Forest based anomaly detection."
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


class AnomalyDetectionRequest(BaseModel):
    """
    Request schema for anomaly detection.
    """

    records: List[PriceRecord] = Field(
        ...,
        description="Historical asset price records.",
    )
    save_output: bool = Field(
        default=True,
        description="Whether to save anomaly results as CSV.",
    )
    return_records: bool = Field(
        default=True,
        description="Whether to return anomaly records in API response.",
    )
    only_anomalies: bool = Field(
        default=False,
        description="Whether to return only anomaly records.",
    )


class BatchAnomalyRequest(BaseModel):
    """
    Request schema for batch anomaly detection.
    """

    records: List[PriceRecord] = Field(
        ...,
        description="Historical asset price records.",
    )
    save_output: bool = Field(
        default=True,
        description="Whether to save anomaly results.",
    )
    only_anomalies: bool = Field(
        default=True,
        description="Whether to return only anomaly records.",
    )


# ============================================================
# Directory Helpers
# ============================================================

def ensure_anomaly_service_directories() -> None:
    """
    Create required directories for anomaly service outputs.
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
    Convert API request records into cleaned dataframe.

    Args:
        records (List[PriceRecord]): List of price records.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    if not records:
        raise ValueError("No records provided.")

    dataframe = pd.DataFrame([record.dict() for record in records])

    return clean_input_data(dataframe)


def load_default_data() -> pd.DataFrame:
    """
    Load default project dataset.

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
    Filter dataframe to only anomaly records if requested.

    Args:
        dataframe (pd.DataFrame): Anomaly result dataframe.
        only_anomalies (bool): Whether to keep only anomalies.

    Returns:
        pd.DataFrame: Filtered dataframe.
    """
    if dataframe.empty:
        return dataframe

    if only_anomalies and "is_anomaly" in dataframe.columns:
        data = dataframe.copy()
        data["is_anomaly"] = data["is_anomaly"].astype(bool)
        return data[data["is_anomaly"]].copy()

    return dataframe


# ============================================================
# Summary Helpers
# ============================================================

def get_model_status() -> Dict[str, Any]:
    """
    Check anomaly model artifact status.

    Returns:
        Dict[str, Any]: Model status.
    """
    return {
        "anomaly_detector": {
            "available": ANOMALY_MODEL_PATH.exists(),
            "path": str(ANOMALY_MODEL_PATH),
        },
        "default_data": {
            "available": DATA_PATH.exists(),
            "path": str(DATA_PATH),
        },
    }


def get_method_summary(anomaly_dataframe: pd.DataFrame) -> Dict[str, int]:
    """
    Calculate method-wise anomaly count.

    Args:
        anomaly_dataframe (pd.DataFrame): Anomaly result dataframe.

    Returns:
        Dict[str, int]: Method summary.
    """
    method_columns = [
        "is_anomaly_iqr",
        "is_anomaly_zscore",
        "is_anomaly_isolation_forest",
    ]

    method_summary = {}

    for column in method_columns:
        if column in anomaly_dataframe.columns:
            method_summary[column] = int(
                anomaly_dataframe[column].astype(bool).sum()
            )

    return method_summary


def get_severity_summary(anomaly_dataframe: pd.DataFrame) -> Dict[str, int]:
    """
    Calculate anomaly severity distribution.

    Args:
        anomaly_dataframe (pd.DataFrame): Anomaly result dataframe.

    Returns:
        Dict[str, int]: Severity summary.
    """
    if "anomaly_severity" not in anomaly_dataframe.columns:
        return {}

    return {
        str(key): int(value)
        for key, value in anomaly_dataframe["anomaly_severity"]
        .value_counts()
        .to_dict()
        .items()
    }


def get_asset_anomaly_summary(anomaly_dataframe: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Generate asset-wise anomaly summary.

    Args:
        anomaly_dataframe (pd.DataFrame): Anomaly result dataframe.

    Returns:
        List[Dict[str, Any]]: Asset-wise summary records.
    """
    if anomaly_dataframe.empty:
        return []

    if "asset" not in anomaly_dataframe.columns or "is_anomaly" not in anomaly_dataframe.columns:
        return []

    data = anomaly_dataframe.copy()
    data["is_anomaly"] = data["is_anomaly"].astype(bool)

    aggregation_dict = {
        "total_records": ("is_anomaly", "count"),
        "anomaly_count": ("is_anomaly", "sum"),
    }

    if "price" in data.columns:
        aggregation_dict.update(
            {
                "average_price": ("price", "mean"),
                "min_price": ("price", "min"),
                "max_price": ("price", "max"),
            }
        )

    asset_summary = (
        data.groupby("asset")
        .agg(**aggregation_dict)
        .reset_index()
    )

    asset_summary["anomaly_rate_percent"] = (
        asset_summary["anomaly_count"]
        / asset_summary["total_records"]
        * 100
    )

    asset_summary = asset_summary.sort_values(
        "anomaly_count",
        ascending=False,
    )

    return asset_summary.to_dict(orient="records")


def get_top_anomaly_records(
    anomaly_dataframe: pd.DataFrame,
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """
    Get top anomaly records.

    Args:
        anomaly_dataframe (pd.DataFrame): Anomaly result dataframe.
        top_n (int): Number of top anomalies.

    Returns:
        List[Dict[str, Any]]: Top anomaly records.
    """
    if anomaly_dataframe.empty or "is_anomaly" not in anomaly_dataframe.columns:
        return []

    data = anomaly_dataframe.copy()
    data["is_anomaly"] = data["is_anomaly"].astype(bool)

    anomalies = data[data["is_anomaly"]].copy()

    if anomalies.empty:
        return []

    sort_columns = []
    ascending_values = []

    if "anomaly_method_count" in anomalies.columns:
        sort_columns.append("anomaly_method_count")
        ascending_values.append(False)

    if "anomaly_score" in anomalies.columns:
        sort_columns.append("anomaly_score")
        ascending_values.append(True)

    if sort_columns:
        anomalies = anomalies.sort_values(
            sort_columns,
            ascending=ascending_values,
        )

    return dataframe_to_json_records(anomalies.head(top_n))


def build_complete_anomaly_summary(
    input_dataframe: pd.DataFrame,
    anomaly_dataframe: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Build complete anomaly summary.

    Args:
        input_dataframe (pd.DataFrame): Input dataframe.
        anomaly_dataframe (pd.DataFrame): Anomaly result dataframe.

    Returns:
        Dict[str, Any]: Complete anomaly summary.
    """
    basic_summary = summarize_anomaly_results(anomaly_dataframe)

    complete_summary = {
        "input_summary": get_input_summary(input_dataframe),
        "anomaly_summary": basic_summary,
        "method_summary": get_method_summary(anomaly_dataframe),
        "severity_summary": get_severity_summary(anomaly_dataframe),
        "asset_summary": get_asset_anomaly_summary(anomaly_dataframe),
        "top_anomalies": get_top_anomaly_records(anomaly_dataframe, top_n=10),
    }

    return complete_summary


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
    Save anomaly summary as Markdown.

    Args:
        summary (Dict[str, Any]): Summary dictionary.
        file_path (Path): Output Markdown path.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    input_summary = summary.get("input_summary", {})
    anomaly_summary = summary.get("anomaly_summary", {})
    method_summary = summary.get("method_summary", {})
    severity_summary = summary.get("severity_summary", {})
    asset_summary = summary.get("asset_summary", [])
    performance_summary = summary.get("performance_summary", {})

    lines = [
        "# API Anomaly Detection Summary",
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
        "## Anomaly Summary",
        "",
        f"- Total Records Checked: {anomaly_summary.get('total_records')}",
        f"- Total Anomalies: {anomaly_summary.get('total_anomalies')}",
        f"- Normal Records: {anomaly_summary.get('normal_records')}",
        f"- Anomaly Rate: {anomaly_summary.get('anomaly_rate_percent')}%",
        "",
        "## Method Summary",
        "",
    ]

    if method_summary:
        for method_name, count in method_summary.items():
            lines.append(f"- {method_name}: {count}")
    else:
        lines.append("- No method summary available.")

    lines.extend(
        [
            "",
            "## Severity Summary",
            "",
        ]
    )

    if severity_summary:
        for severity, count in severity_summary.items():
            lines.append(f"- {severity}: {count}")
    else:
        lines.append("- No severity summary available.")

    lines.extend(
        [
            "",
            "## Asset Summary",
            "",
            "| Asset | Records | Anomalies | Anomaly Rate % |",
            "|---|---:|---:|---:|",
        ]
    )

    if asset_summary:
        for row in asset_summary:
            lines.append(
                "| {asset} | {records} | {anomalies} | {rate:.2f} |".format(
                    asset=row.get("asset"),
                    records=int(row.get("total_records", 0)),
                    anomalies=int(row.get("anomaly_count", 0)),
                    rate=float(row.get("anomaly_rate_percent", 0)),
                )
            )
    else:
        lines.append("| No asset summary available | - | - | - |")

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- Anomaly Results CSV: `{summary.get('anomaly_output_path')}`",
            f"- Summary JSON: `{API_ANOMALY_SUMMARY_JSON_PATH}`",
            f"- Summary Markdown: `{API_ANOMALY_SUMMARY_MD_PATH}`",
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
            "- IQR detects values outside the normal price spread.",
            "- Z-score detects values far from the asset average.",
            "- Isolation Forest detects unusual patterns using engineered features.",
            "- Combined anomaly results are useful for risk monitoring and data quality checks.",
        ]
    )

    file_path.write_text("\n".join(lines), encoding="utf-8")


def save_service_outputs(
    input_dataframe: pd.DataFrame,
    anomaly_dataframe: pd.DataFrame,
    save_output: bool = True,
) -> Dict[str, Any]:
    """
    Save anomaly service outputs.

    Args:
        input_dataframe (pd.DataFrame): Input dataframe.
        anomaly_dataframe (pd.DataFrame): Anomaly result dataframe.
        save_output (bool): Whether to save outputs.

    Returns:
        Dict[str, Any]: Output summary.
    """
    save_performance_reports()
    performance_summary = get_performance_summary()

    summary = build_complete_anomaly_summary(
        input_dataframe=input_dataframe,
        anomaly_dataframe=anomaly_dataframe,
    )

    full_summary = {
        "status": "success",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **summary,
        "anomaly_output_path": str(API_ANOMALY_OUTPUT_PATH) if save_output else None,
        "summary_json_path": str(API_ANOMALY_SUMMARY_JSON_PATH) if save_output else None,
        "summary_markdown_path": str(API_ANOMALY_SUMMARY_MD_PATH) if save_output else None,
        "performance_summary": performance_summary,
    }

    if save_output:
        save_anomaly_results(anomaly_dataframe, API_ANOMALY_OUTPUT_PATH)
        save_json_summary(full_summary, API_ANOMALY_SUMMARY_JSON_PATH)
        save_markdown_summary(full_summary, API_ANOMALY_SUMMARY_MD_PATH)

    return full_summary


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
# API Routes
# ============================================================

@app.get("/")
def root() -> Dict[str, Any]:
    """
    Root endpoint.
    """
    return build_success_response(
        message="Financial Portfolio Anomaly Detection API is running.",
        data={
            "project": "Financial Portfolio Forecasting & Anomaly Detection",
            "service": "anomaly_service",
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
    ensure_anomaly_service_directories()

    return build_success_response(
        message="Anomaly service health check completed.",
        data={
            "service": "anomaly_service",
            "project_root": str(PROJECT_ROOT),
            "model_status": get_model_status(),
        },
    )


@app.get("/model/status")
def model_status() -> Dict[str, Any]:
    """
    Return anomaly model status.
    """
    return build_success_response(
        message="Anomaly model status loaded.",
        data=get_model_status(),
    )


@app.post("/detect")
def detect_anomalies(request: AnomalyDetectionRequest) -> Dict[str, Any]:
    """
    Detect anomalies from JSON records.
    """
    try:
        ensure_anomaly_service_directories()

        with track_step("api_detect_anomalies", category="api"):
            input_dataframe = records_to_dataframe(request.records)

            anomaly_dataframe = detect_anomalies_with_saved_model(
                input_dataframe,
            )

            output_summary = save_service_outputs(
                input_dataframe=input_dataframe,
                anomaly_dataframe=anomaly_dataframe,
                save_output=request.save_output,
            )

        response_data = {
            "input_summary": output_summary.get("input_summary"),
            "anomaly_summary": output_summary.get("anomaly_summary"),
            "method_summary": output_summary.get("method_summary"),
            "severity_summary": output_summary.get("severity_summary"),
            "asset_summary": output_summary.get("asset_summary"),
            "top_anomalies": output_summary.get("top_anomalies"),
            "saved_path": output_summary.get("anomaly_output_path"),
            "performance_summary": output_summary.get("performance_summary"),
        }

        if request.return_records:
            response_records = filter_anomaly_records(
                anomaly_dataframe,
                only_anomalies=request.only_anomalies,
            )

            response_data["records"] = dataframe_to_json_records(response_records)

        return build_success_response(
            message="Anomaly detection completed successfully.",
            data=response_data,
        )

    except Exception as error:
        logger.error("Anomaly detection API failed: %s", error, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=build_error_response(error),
        )


@app.post("/batch-detect")
def batch_detect_anomalies(request: BatchAnomalyRequest) -> Dict[str, Any]:
    """
    Batch anomaly detection endpoint.

    This endpoint returns only anomaly records by default.
    """
    try:
        ensure_anomaly_service_directories()

        with track_step("api_batch_detect_anomalies", category="api"):
            input_dataframe = records_to_dataframe(request.records)

            anomaly_dataframe = detect_anomalies_with_saved_model(
                input_dataframe,
            )

            output_summary = save_service_outputs(
                input_dataframe=input_dataframe,
                anomaly_dataframe=anomaly_dataframe,
                save_output=request.save_output,
            )

            filtered_records = filter_anomaly_records(
                anomaly_dataframe,
                only_anomalies=request.only_anomalies,
            )

        return build_success_response(
            message="Batch anomaly detection completed successfully.",
            data={
                "input_summary": output_summary.get("input_summary"),
                "anomaly_summary": output_summary.get("anomaly_summary"),
                "method_summary": output_summary.get("method_summary"),
                "asset_summary": output_summary.get("asset_summary"),
                "top_anomalies": output_summary.get("top_anomalies"),
                "records": dataframe_to_json_records(filtered_records),
                "saved_path": output_summary.get("anomaly_output_path"),
                "performance_summary": output_summary.get("performance_summary"),
            },
        )

    except Exception as error:
        logger.error("Batch anomaly detection API failed: %s", error, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=build_error_response(error),
        )


@app.get("/detect/default")
def detect_default_data(
    save_output: bool = Query(
        default=True,
        description="Whether to save anomaly results.",
    ),
    return_records: bool = Query(
        default=True,
        description="Whether to return records in the response.",
    ),
    only_anomalies: bool = Query(
        default=False,
        description="Whether to return only anomaly records.",
    ),
) -> Dict[str, Any]:
    """
    Detect anomalies using default dataset from data/raw.
    """
    try:
        ensure_anomaly_service_directories()

        with track_step("api_detect_default_data", category="api"):
            input_dataframe = load_default_data()

            anomaly_dataframe = detect_anomalies_with_saved_model(
                input_dataframe,
            )

            output_summary = save_service_outputs(
                input_dataframe=input_dataframe,
                anomaly_dataframe=anomaly_dataframe,
                save_output=save_output,
            )

        response_data = {
            "input_summary": output_summary.get("input_summary"),
            "anomaly_summary": output_summary.get("anomaly_summary"),
            "method_summary": output_summary.get("method_summary"),
            "severity_summary": output_summary.get("severity_summary"),
            "asset_summary": output_summary.get("asset_summary"),
            "top_anomalies": output_summary.get("top_anomalies"),
            "saved_path": output_summary.get("anomaly_output_path"),
            "performance_summary": output_summary.get("performance_summary"),
        }

        if return_records:
            response_records = filter_anomaly_records(
                anomaly_dataframe,
                only_anomalies=only_anomalies,
            )

            response_data["records"] = dataframe_to_json_records(response_records)

        return build_success_response(
            message="Default data anomaly detection completed successfully.",
            data=response_data,
        )

    except Exception as error:
        logger.error("Default anomaly detection API failed: %s", error, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=build_error_response(error),
        )


@app.post("/upload")
async def upload_csv_detect_anomalies(
    file: UploadFile = File(...),
    save_output: bool = Query(
        default=True,
        description="Whether to save anomaly results.",
    ),
    return_records: bool = Query(
        default=True,
        description="Whether to return records in response.",
    ),
    only_anomalies: bool = Query(
        default=False,
        description="Whether to return only anomaly records.",
    ),
) -> Dict[str, Any]:
    """
    Upload CSV file and detect anomalies.

    CSV must contain:

    - date, asset, price

    or:

    - Date, Asset, Price
    """
    temp_path = None

    try:
        ensure_anomaly_service_directories()

        if not file.filename.lower().endswith(".csv"):
            raise ValueError("Only CSV files are supported.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_file.write(await file.read())
            temp_path = Path(temp_file.name)

        with track_step("api_upload_anomaly_detection", category="api"):
            input_dataframe = pd.read_csv(temp_path)
            input_dataframe = clean_input_data(input_dataframe)

            anomaly_dataframe = detect_anomalies_with_saved_model(
                input_dataframe,
            )

            output_summary = save_service_outputs(
                input_dataframe=input_dataframe,
                anomaly_dataframe=anomaly_dataframe,
                save_output=save_output,
            )

        response_data = {
            "uploaded_file": file.filename,
            "input_summary": output_summary.get("input_summary"),
            "anomaly_summary": output_summary.get("anomaly_summary"),
            "method_summary": output_summary.get("method_summary"),
            "severity_summary": output_summary.get("severity_summary"),
            "asset_summary": output_summary.get("asset_summary"),
            "top_anomalies": output_summary.get("top_anomalies"),
            "saved_path": output_summary.get("anomaly_output_path"),
            "performance_summary": output_summary.get("performance_summary"),
        }

        if return_records:
            response_records = filter_anomaly_records(
                anomaly_dataframe,
                only_anomalies=only_anomalies,
            )

            response_data["records"] = dataframe_to_json_records(response_records)

        return build_success_response(
            message="Uploaded CSV anomaly detection completed successfully.",
            data=response_data,
        )

    except Exception as error:
        logger.error("Upload anomaly detection API failed: %s", error, exc_info=True)
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
    Return latest saved API anomaly summary.
    """
    try:
        if not API_ANOMALY_SUMMARY_JSON_PATH.exists():
            raise FileNotFoundError(
                f"No API anomaly summary found at: {API_ANOMALY_SUMMARY_JSON_PATH}"
            )

        with API_ANOMALY_SUMMARY_JSON_PATH.open("r", encoding="utf-8") as file:
            summary = json.load(file)

        return build_success_response(
            message="Latest anomaly summary loaded successfully.",
            data=summary,
        )

    except Exception as error:
        logger.error("Latest summary API failed: %s", error, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=build_error_response(error),
        )


# ============================================================
# Local Script Runner
# ============================================================

def run_local_anomaly_service_test() -> Dict[str, Any]:
    """
    Run local anomaly service test without starting API server.

    Returns:
        Dict[str, Any]: Test summary.
    """
    try:
        ensure_anomaly_service_directories()

        input_dataframe = load_default_data()

        anomaly_dataframe = detect_anomalies_with_saved_model(input_dataframe)

        output_summary = save_service_outputs(
            input_dataframe=input_dataframe,
            anomaly_dataframe=anomaly_dataframe,
            save_output=True,
        )

        return {
            "status": "success",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **output_summary,
        }

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
    result = run_local_anomaly_service_test()

    print("\n" + "=" * 80)
    print("ANOMALY SERVICE LOCAL TEST SUMMARY")
    print("=" * 80)

    print("Status:", result.get("status"))

    if result.get("status") == "success":
        input_summary = result.get("input_summary", {})
        anomaly_summary = result.get("anomaly_summary", {})

        print("\nInput Summary:")
        print("- Total Records:", input_summary.get("total_records"))
        print("- Total Assets:", input_summary.get("total_assets"))
        print("- Start Date:", input_summary.get("start_date"))
        print("- End Date:", input_summary.get("end_date"))

        print("\nAnomaly Summary:")
        print("- Total Records:", anomaly_summary.get("total_records"))
        print("- Total Anomalies:", anomaly_summary.get("total_anomalies"))
        print("- Normal Records:", anomaly_summary.get("normal_records"))
        print("- Anomaly Rate %:", anomaly_summary.get("anomaly_rate_percent"))

        print("\nSaved Files:")
        print("- outputs/api_anomaly_results.csv")
        print("- reports/tables/api_anomaly_summary.json")
        print("- reports/api_anomaly_summary.md")

    else:
        print("\nLocal anomaly service test failed.")
        print("Error Type:", result.get("error_type"))
        print("Error Message:", result.get("error_message"))

    print("=" * 80)