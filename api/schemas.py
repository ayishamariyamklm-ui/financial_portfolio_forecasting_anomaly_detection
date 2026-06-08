"""
Pydantic schemas for the Financial Portfolio Forecasting
and Anomaly Detection API.

This file contains reusable request and response models for:

1. Forecasting endpoints
2. Anomaly detection endpoints
3. Combined prediction endpoints
4. CSV upload responses
5. Health check responses
6. Error responses

Dataset expected columns:
- date
- asset
- price
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ============================================================
# Base Schemas
# ============================================================

class PriceRecord(BaseModel):
    """
    Single financial asset price record.

    Example:
        {
            "date": "2024-01-01",
            "asset": "Stock A",
            "price": 120.50
        }
    """

    date: str = Field(
        ...,
        description="Date of the asset price record. Example: 2024-01-01",
        examples=["2024-01-01"],
    )
    asset: str = Field(
        ...,
        min_length=1,
        description="Asset name. Example: Stock A",
        examples=["Stock A"],
    )
    price: float = Field(
        ...,
        ge=0,
        description="Asset price. Must be non-negative.",
        examples=[120.50],
    )

    @field_validator("asset")
    @classmethod
    def validate_asset(cls, value: str) -> str:
        """
        Validate asset name.

        Args:
            value (str): Asset name.

        Returns:
            str: Cleaned asset name.
        """
        value = value.strip()

        if not value:
            raise ValueError("Asset name cannot be empty.")

        return value

    @field_validator("date")
    @classmethod
    def validate_date(cls, value: str) -> str:
        """
        Validate date string.

        Args:
            value (str): Date value.

        Returns:
            str: Date string.
        """
        value = value.strip()

        if not value:
            raise ValueError("Date cannot be empty.")

        return value


# ============================================================
# Forecast Request Schemas
# ============================================================

class ForecastRequest(BaseModel):
    """
    Request schema for forecasting endpoint.
    """

    records: List[PriceRecord] = Field(
        ...,
        min_length=1,
        description="Historical asset price records.",
    )
    model_name: str = Field(
        default="xgboost",
        description="Forecasting model: xgboost, arima, prophet, lstm, or all.",
        examples=["xgboost"],
    )
    forecast_steps: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Number of future periods to forecast.",
        examples=[30],
    )
    save_output: bool = Field(
        default=True,
        description="Whether to save forecast output as CSV.",
    )
    return_records: bool = Field(
        default=True,
        description="Whether to return forecast records in API response.",
    )

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        """
        Validate forecasting model name.

        Args:
            value (str): Model name.

        Returns:
            str: Lowercase model name.
        """
        allowed_models = {"xgboost", "arima", "prophet", "lstm", "all"}

        value = value.strip().lower()

        if value not in allowed_models:
            raise ValueError(
                f"Invalid model_name '{value}'. "
                f"Allowed values are: {sorted(allowed_models)}"
            )

        return value


class ForecastDefaultRequest(BaseModel):
    """
    Optional schema for default dataset forecasting configuration.
    """

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
        description="Whether to save output files.",
    )
    return_records: bool = Field(
        default=True,
        description="Whether to return forecast records.",
    )

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        """
        Validate forecasting model name.

        Args:
            value (str): Model name.

        Returns:
            str: Lowercase model name.
        """
        allowed_models = {"xgboost", "arima", "prophet", "lstm", "all"}

        value = value.strip().lower()

        if value not in allowed_models:
            raise ValueError(
                f"Invalid model_name '{value}'. "
                f"Allowed values are: {sorted(allowed_models)}"
            )

        return value


# ============================================================
# Anomaly Request Schemas
# ============================================================

class AnomalyRequest(BaseModel):
    """
    Request schema for anomaly detection endpoint.
    """

    records: List[PriceRecord] = Field(
        ...,
        min_length=1,
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


class BatchAnomalyRequest(BaseModel):
    """
    Request schema for batch anomaly detection.

    This schema is useful when the response should usually return
    only anomaly records.
    """

    records: List[PriceRecord] = Field(
        ...,
        min_length=1,
        description="Historical asset price records.",
    )
    save_output: bool = Field(
        default=True,
        description="Whether to save anomaly output as CSV.",
    )
    only_anomalies: bool = Field(
        default=True,
        description="Whether to return only anomaly records.",
    )


# ============================================================
# Combined Prediction Request Schema
# ============================================================

class CombinedPredictionRequest(BaseModel):
    """
    Request schema for combined forecasting and anomaly detection.
    """

    records: List[PriceRecord] = Field(
        ...,
        min_length=1,
        description="Historical asset price records.",
    )
    model_name: str = Field(
        default="xgboost",
        description="Forecasting model: xgboost, arima, prophet, lstm, or all.",
        examples=["xgboost"],
    )
    forecast_steps: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Number of future periods to forecast.",
        examples=[30],
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
        description="Whether to return forecast/anomaly records.",
    )
    only_anomalies: bool = Field(
        default=False,
        description="Whether anomaly response should return only anomaly records.",
    )

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        """
        Validate forecasting model name.

        Args:
            value (str): Model name.

        Returns:
            str: Lowercase model name.
        """
        allowed_models = {"xgboost", "arima", "prophet", "lstm", "all"}

        value = value.strip().lower()

        if value not in allowed_models:
            raise ValueError(
                f"Invalid model_name '{value}'. "
                f"Allowed values are: {sorted(allowed_models)}"
            )

        return value


# ============================================================
# Upload Request Configuration Schemas
# ============================================================

class UploadPredictionConfig(BaseModel):
    """
    Configuration schema for uploaded CSV prediction workflow.

    Note:
        In FastAPI, file upload itself is handled using UploadFile.
        This schema is useful for documenting/query configuration.
    """

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
        description="Whether to save output files.",
    )
    return_records: bool = Field(
        default=True,
        description="Whether to return output records.",
    )
    only_anomalies: bool = Field(
        default=False,
        description="Whether to return only anomaly records.",
    )

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        """
        Validate forecasting model name.

        Args:
            value (str): Model name.

        Returns:
            str: Lowercase model name.
        """
        allowed_models = {"xgboost", "arima", "prophet", "lstm", "all"}

        value = value.strip().lower()

        if value not in allowed_models:
            raise ValueError(
                f"Invalid model_name '{value}'. "
                f"Allowed values are: {sorted(allowed_models)}"
            )

        return value


# ============================================================
# Summary Schemas
# ============================================================

class InputSummary(BaseModel):
    """
    Input data summary schema.
    """

    total_records: Optional[int] = None
    total_assets: Optional[int] = None
    assets: List[str] = Field(default_factory=list)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    average_price: Optional[float] = None
    minimum_price: Optional[float] = None
    maximum_price: Optional[float] = None


class ForecastSummary(BaseModel):
    """
    Forecast result summary schema.
    """

    forecast_rows: Optional[int] = None
    models_used: List[str] = Field(default_factory=list)
    assets_forecasted: List[str] = Field(default_factory=list)
    forecast_start_date: Optional[str] = None
    forecast_end_date: Optional[str] = None
    average_forecasted_price: Optional[float] = None
    minimum_forecasted_price: Optional[float] = None
    maximum_forecasted_price: Optional[float] = None


class AnomalySummary(BaseModel):
    """
    Anomaly result summary schema.
    """

    total_records: Optional[int] = None
    total_anomalies: Optional[int] = None
    normal_records: Optional[int] = None
    anomaly_rate_percent: Optional[float] = None
    method_summary: Dict[str, int] = Field(default_factory=dict)
    asset_summary: List[Dict[str, Any]] = Field(default_factory=list)


class PerformanceSummary(BaseModel):
    """
    Performance tracking summary schema.
    """

    total_records: Optional[int] = None
    total_execution_time_seconds: Optional[float] = None
    successful_steps: Optional[int] = None
    failed_steps: Optional[int] = None


# ============================================================
# Record Output Schemas
# ============================================================

class ForecastRecord(BaseModel):
    """
    Single forecast output record.
    """

    date: Optional[str] = None
    asset: Optional[str] = None
    forecasted_price: Optional[float] = None
    forecast_lower_bound: Optional[float] = None
    forecast_upper_bound: Optional[float] = None
    model_name: Optional[str] = None


class AnomalyRecord(BaseModel):
    """
    Single anomaly output record.
    """

    date: Optional[str] = None
    asset: Optional[str] = None
    price: Optional[float] = None
    daily_return: Optional[float] = None
    price_z_score: Optional[float] = None
    is_anomaly_iqr: Optional[bool] = None
    is_anomaly_zscore: Optional[bool] = None
    is_anomaly_isolation_forest: Optional[bool] = None
    anomaly_method_count: Optional[int] = None
    is_anomaly: Optional[bool] = None
    anomaly_method: Optional[str] = None
    anomaly_severity: Optional[str] = None
    anomaly_score: Optional[float] = None


# ============================================================
# Standard API Response Schemas
# ============================================================

class StandardResponse(BaseModel):
    """
    Standard API response schema.
    """

    status: str = Field(
        default="success",
        description="Response status.",
    )
    message: str = Field(
        ...,
        description="Response message.",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        description="Response timestamp.",
    )
    data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Response payload.",
    )


class ErrorResponse(BaseModel):
    """
    Standard API error response schema.
    """

    status: str = Field(
        default="failed",
        description="Error status.",
    )
    error_type: str = Field(
        ...,
        description="Exception type.",
    )
    error_message: str = Field(
        ...,
        description="Error message.",
    )
    traceback: Optional[str] = Field(
        default=None,
        description="Traceback details.",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        description="Error timestamp.",
    )


# ============================================================
# Detailed Response Payload Schemas
# ============================================================

class ForecastResponseData(BaseModel):
    """
    Forecast endpoint response data.
    """

    input_summary: Optional[Dict[str, Any]] = None
    forecast_summary: Optional[Dict[str, Any]] = None
    saved_path: Optional[str] = None
    performance_summary: Optional[Dict[str, Any]] = None
    forecast_records: List[Dict[str, Any]] = Field(default_factory=list)


class AnomalyResponseData(BaseModel):
    """
    Anomaly endpoint response data.
    """

    input_summary: Optional[Dict[str, Any]] = None
    anomaly_summary: Optional[Dict[str, Any]] = None
    saved_path: Optional[str] = None
    performance_summary: Optional[Dict[str, Any]] = None
    anomaly_records: List[Dict[str, Any]] = Field(default_factory=list)


class CombinedPredictionResponseData(BaseModel):
    """
    Combined prediction endpoint response data.
    """

    input_summary: Optional[Dict[str, Any]] = None
    forecast_summary: Optional[Dict[str, Any]] = None
    anomaly_summary: Optional[Dict[str, Any]] = None
    forecast_saved_path: Optional[str] = None
    anomaly_saved_path: Optional[str] = None
    performance_summary: Optional[Dict[str, Any]] = None
    forecast_records: List[Dict[str, Any]] = Field(default_factory=list)
    anomaly_records: List[Dict[str, Any]] = Field(default_factory=list)


class HealthResponseData(BaseModel):
    """
    Health endpoint response data.
    """

    service: Optional[str] = None
    project_root: Optional[str] = None
    default_data_available: Optional[bool] = None
    default_data_path: Optional[str] = None
    model_artifacts: Dict[str, Any] = Field(default_factory=dict)


class ModelArtifactStatus(BaseModel):
    """
    Model artifact status response schema.
    """

    xgboost: Dict[str, Any] = Field(default_factory=dict)
    arima: Dict[str, Any] = Field(default_factory=dict)
    prophet: Dict[str, Any] = Field(default_factory=dict)
    lstm: Dict[str, Any] = Field(default_factory=dict)
    anomaly_detector: Dict[str, Any] = Field(default_factory=dict)


# ============================================================
# Latest Summary Schema
# ============================================================

class LatestSummaryResponseData(BaseModel):
    """
    Latest saved API summary response data.
    """

    status: Optional[str] = None
    generated_at: Optional[str] = None
    input_summary: Optional[Dict[str, Any]] = None
    forecast_summary: Optional[Dict[str, Any]] = None
    anomaly_summary: Optional[Dict[str, Any]] = None
    forecast_output_path: Optional[str] = None
    anomaly_output_path: Optional[str] = None
    performance_summary: Optional[Dict[str, Any]] = None


# ============================================================
# Example Payloads
# ============================================================

EXAMPLE_PRICE_RECORDS = [
    {
        "date": "2024-01-01",
        "asset": "Stock A",
        "price": 120.50,
    },
    {
        "date": "2024-01-02",
        "asset": "Stock A",
        "price": 121.75,
    },
    {
        "date": "2024-01-03",
        "asset": "Stock A",
        "price": 119.90,
    },
    {
        "date": "2024-01-01",
        "asset": "Bond A",
        "price": 95.10,
    },
    {
        "date": "2024-01-02",
        "asset": "Bond A",
        "price": 96.25,
    },
]


EXAMPLE_FORECAST_REQUEST = {
    "records": EXAMPLE_PRICE_RECORDS,
    "model_name": "xgboost",
    "forecast_steps": 30,
    "save_output": True,
    "return_records": True,
}


EXAMPLE_ANOMALY_REQUEST = {
    "records": EXAMPLE_PRICE_RECORDS,
    "save_output": True,
    "return_records": True,
    "only_anomalies": False,
}


EXAMPLE_COMBINED_REQUEST = {
    "records": EXAMPLE_PRICE_RECORDS,
    "model_name": "xgboost",
    "forecast_steps": 30,
    "run_forecasting": True,
    "run_anomaly_detection": True,
    "save_output": True,
    "return_records": True,
    "only_anomalies": False,
}


# ============================================================
# Utility Schema Helpers
# ============================================================

def get_allowed_forecasting_models() -> List[str]:
    """
    Return allowed forecasting model names.

    Returns:
        List[str]: Allowed model names.
    """
    return ["xgboost", "arima", "prophet", "lstm", "all"]


def build_success_response_schema(
    message: str,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build standard success response dictionary.

    Args:
        message (str): Response message.
        data (Optional[Dict[str, Any]]): Response data.

    Returns:
        Dict[str, Any]: Standard response.
    """
    response = StandardResponse(
        message=message,
        data=data,
    )

    return response.model_dump()


def build_error_response_schema(
    error_type: str,
    error_message: str,
    traceback_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build standard error response dictionary.

    Args:
        error_type (str): Error type.
        error_message (str): Error message.
        traceback_text (Optional[str]): Traceback text.

    Returns:
        Dict[str, Any]: Error response.
    """
    response = ErrorResponse(
        error_type=error_type,
        error_message=error_message,
        traceback=traceback_text,
    )

    return response.model_dump()


# ============================================================
# Script Usage Test
# ============================================================

if __name__ == "__main__":
    print("API schemas loaded successfully.")

    sample_forecast_request = ForecastRequest(**EXAMPLE_FORECAST_REQUEST)
    sample_anomaly_request = AnomalyRequest(**EXAMPLE_ANOMALY_REQUEST)
    sample_combined_request = CombinedPredictionRequest(**EXAMPLE_COMBINED_REQUEST)

    print("\nAllowed forecasting models:")
    print(get_allowed_forecasting_models())

    print("\nSample Forecast Request:")
    print(sample_forecast_request.model_dump())

    print("\nSample Anomaly Request:")
    print(sample_anomaly_request.model_dump())

    print("\nSample Combined Request:")
    print(sample_combined_request.model_dump())

    print("\nSample Success Response:")
    print(
        build_success_response_schema(
            message="Schemas tested successfully.",
            data={"example": True},
        )
    )