"""
Custom exception handling utilities for the Financial Portfolio Forecasting
and Anomaly Detection project.

This module provides:
- Project-specific custom exceptions
- Error message formatting
- Traceback logging
- Safe exception handling decorators
- Validation helpers for files, columns, and model artifacts
"""

import functools
import logging
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, TypeVar, Union


# ============================================================
# Type Variable for Decorators
# ============================================================

F = TypeVar("F", bound=Callable[..., Any])


# ============================================================
# Base Project Exception
# ============================================================

class FinancialPortfolioException(Exception):
    """
    Base exception class for the Financial Portfolio Forecasting
    and Anomaly Detection project.
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> None:
        """
        Initialize the base project exception.

        Args:
            message (str): Human-readable error message.
            error_code (Optional[str]): Optional project-specific error code.
            details (Optional[dict]): Additional error details.
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "PROJECT_ERROR"
        self.details = details or {}

    def __str__(self) -> str:
        """
        Return a readable string representation of the exception.

        Returns:
            str: Formatted exception message.
        """
        if self.details:
            return (
                f"[{self.error_code}] {self.message} | "
                f"Details: {self.details}"
            )

        return f"[{self.error_code}] {self.message}"


# ============================================================
# Configuration Exceptions
# ============================================================

class ConfigurationError(FinancialPortfolioException):
    """Raised when configuration loading or validation fails."""

    def __init__(
        self,
        message: str,
        details: Optional[dict] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details=details,
        )


class MissingConfigurationError(ConfigurationError):
    """Raised when a required configuration key or file is missing."""

    def __init__(
        self,
        message: str,
        details: Optional[dict] = None,
    ) -> None:
        super().__init__(message=message, details=details)
        self.error_code = "MISSING_CONFIGURATION_ERROR"


# ============================================================
# Data Exceptions
# ============================================================

class DataError(FinancialPortfolioException):
    """Raised when a general data-related error occurs."""

    def __init__(
        self,
        message: str,
        details: Optional[dict] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="DATA_ERROR",
            details=details,
        )


class DataValidationError(DataError):
    """Raised when dataset validation fails."""

    def __init__(
        self,
        message: str,
        details: Optional[dict] = None,
    ) -> None:
        super().__init__(message=message, details=details)
        self.error_code = "DATA_VALIDATION_ERROR"


class MissingColumnError(DataValidationError):
    """Raised when required columns are missing from the dataset."""

    def __init__(
        self,
        missing_columns: Iterable[str],
        available_columns: Optional[Iterable[str]] = None,
    ) -> None:
        missing_columns_list = list(missing_columns)
        details = {"missing_columns": missing_columns_list}

        if available_columns is not None:
            details["available_columns"] = list(available_columns)

        super().__init__(
            message="Required columns are missing from the dataset.",
            details=details,
        )
        self.error_code = "MISSING_COLUMN_ERROR"


class EmptyDataFrameError(DataValidationError):
    """Raised when a dataframe is empty."""

    def __init__(self, message: str = "The dataframe is empty.") -> None:
        super().__init__(message=message)
        self.error_code = "EMPTY_DATAFRAME_ERROR"


class InvalidDataTypeError(DataValidationError):
    """Raised when a column has an invalid data type."""

    def __init__(
        self,
        column: str,
        expected_type: str,
        actual_type: str,
    ) -> None:
        super().__init__(
            message=f"Invalid data type found for column '{column}'.",
            details={
                "column": column,
                "expected_type": expected_type,
                "actual_type": actual_type,
            },
        )
        self.error_code = "INVALID_DATA_TYPE_ERROR"


# ============================================================
# File and Path Exceptions
# ============================================================

class FilePathError(FinancialPortfolioException):
    """Raised when a file or path-related error occurs."""

    def __init__(
        self,
        message: str,
        path: Optional[Union[str, Path]] = None,
        details: Optional[dict] = None,
    ) -> None:
        error_details = details or {}

        if path is not None:
            error_details["path"] = str(path)

        super().__init__(
            message=message,
            error_code="FILE_PATH_ERROR",
            details=error_details,
        )


class FileNotFoundProjectError(FilePathError):
    """Raised when a required project file is not found."""

    def __init__(self, path: Union[str, Path]) -> None:
        super().__init__(
            message="Required file was not found.",
            path=path,
        )
        self.error_code = "FILE_NOT_FOUND_ERROR"


class DirectoryNotFoundError(FilePathError):
    """Raised when a required directory is not found."""

    def __init__(self, path: Union[str, Path]) -> None:
        super().__init__(
            message="Required directory was not found.",
            path=path,
        )
        self.error_code = "DIRECTORY_NOT_FOUND_ERROR"


# ============================================================
# Feature Engineering Exceptions
# ============================================================

class FeatureEngineeringError(FinancialPortfolioException):
    """Raised when feature engineering fails."""

    def __init__(
        self,
        message: str,
        details: Optional[dict] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="FEATURE_ENGINEERING_ERROR",
            details=details,
        )


# ============================================================
# Model Exceptions
# ============================================================

class ModelError(FinancialPortfolioException):
    """Raised when model training, loading, or prediction fails."""

    def __init__(
        self,
        message: str,
        details: Optional[dict] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="MODEL_ERROR",
            details=details,
        )


class ModelTrainingError(ModelError):
    """Raised when model training fails."""

    def __init__(
        self,
        message: str = "Model training failed.",
        details: Optional[dict] = None,
    ) -> None:
        super().__init__(message=message, details=details)
        self.error_code = "MODEL_TRAINING_ERROR"


class ModelPredictionError(ModelError):
    """Raised when model prediction fails."""

    def __init__(
        self,
        message: str = "Model prediction failed.",
        details: Optional[dict] = None,
    ) -> None:
        super().__init__(message=message, details=details)
        self.error_code = "MODEL_PREDICTION_ERROR"


class ModelArtifactError(ModelError):
    """Raised when model artifacts are missing or invalid."""

    def __init__(
        self,
        message: str = "Model artifact error.",
        details: Optional[dict] = None,
    ) -> None:
        super().__init__(message=message, details=details)
        self.error_code = "MODEL_ARTIFACT_ERROR"


# ============================================================
# Forecasting and Anomaly Detection Exceptions
# ============================================================

class ForecastingError(FinancialPortfolioException):
    """Raised when forecasting fails."""

    def __init__(
        self,
        message: str = "Forecasting failed.",
        details: Optional[dict] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="FORECASTING_ERROR",
            details=details,
        )


class AnomalyDetectionError(FinancialPortfolioException):
    """Raised when anomaly detection fails."""

    def __init__(
        self,
        message: str = "Anomaly detection failed.",
        details: Optional[dict] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="ANOMALY_DETECTION_ERROR",
            details=details,
        )


# ============================================================
# API and Dashboard Exceptions
# ============================================================

class APIError(FinancialPortfolioException):
    """Raised when API execution fails."""

    def __init__(
        self,
        message: str = "API error occurred.",
        details: Optional[dict] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="API_ERROR",
            details=details,
        )


class DashboardError(FinancialPortfolioException):
    """Raised when dashboard execution fails."""

    def __init__(
        self,
        message: str = "Dashboard error occurred.",
        details: Optional[dict] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="DASHBOARD_ERROR",
            details=details,
        )


# ============================================================
# Error Formatting Utilities
# ============================================================

def get_error_details(error: Exception) -> dict:
    """
    Extract useful details from an exception.

    Args:
        error (Exception): Exception object.

    Returns:
        dict: Error details dictionary.
    """
    return {
        "error_type": error.__class__.__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
    }


def format_exception_message(error: Exception) -> str:
    """
    Format an exception into a readable error message.

    Args:
        error (Exception): Exception object.

    Returns:
        str: Formatted error message.
    """
    return (
        f"Error Type: {error.__class__.__name__}\n"
        f"Error Message: {str(error)}"
    )


def log_exception(
    logger: logging.Logger,
    error: Exception,
    message: str = "An error occurred",
) -> None:
    """
    Log an exception with traceback.

    Args:
        logger (logging.Logger): Logger object.
        error (Exception): Exception object.
        message (str): Custom log message.
    """
    logger.error(
        "%s: %s",
        message,
        str(error),
        exc_info=True,
    )


# ============================================================
# Validation Helper Functions
# ============================================================

def validate_file_exists(file_path: Union[str, Path]) -> Path:
    """
    Validate that a file exists.

    Args:
        file_path (Union[str, Path]): File path to validate.

    Returns:
        Path: Validated file path.

    Raises:
        FileNotFoundProjectError: If file does not exist.
    """
    path = Path(file_path)

    if not path.exists() or not path.is_file():
        raise FileNotFoundProjectError(path)

    return path


def validate_directory_exists(directory_path: Union[str, Path]) -> Path:
    """
    Validate that a directory exists.

    Args:
        directory_path (Union[str, Path]): Directory path to validate.

    Returns:
        Path: Validated directory path.

    Raises:
        DirectoryNotFoundError: If directory does not exist.
    """
    path = Path(directory_path)

    if not path.exists() or not path.is_dir():
        raise DirectoryNotFoundError(path)

    return path


def validate_required_columns(
    available_columns: Iterable[str],
    required_columns: Iterable[str],
) -> None:
    """
    Validate that all required columns exist.

    Args:
        available_columns (Iterable[str]): Columns available in dataset.
        required_columns (Iterable[str]): Columns required by the project.

    Raises:
        MissingColumnError: If required columns are missing.
    """
    available_set = set(available_columns)
    required_set = set(required_columns)

    missing_columns = sorted(required_set - available_set)

    if missing_columns:
        raise MissingColumnError(
            missing_columns=missing_columns,
            available_columns=sorted(available_set),
        )


def validate_not_empty(data: Any, name: str = "data") -> None:
    """
    Validate that an object is not empty.

    Supports pandas DataFrames, lists, tuples, sets, and dictionaries.

    Args:
        data (Any): Data object to validate.
        name (str): Name of the object for error messages.

    Raises:
        EmptyDataFrameError: If the object is empty.
    """
    if data is None:
        raise EmptyDataFrameError(f"{name} is None.")

    if hasattr(data, "empty") and data.empty:
        raise EmptyDataFrameError(f"{name} is empty.")

    if isinstance(data, (list, tuple, set, dict)) and len(data) == 0:
        raise EmptyDataFrameError(f"{name} is empty.")


def validate_model_artifact_exists(model_path: Union[str, Path]) -> Path:
    """
    Validate that a model artifact exists.

    Args:
        model_path (Union[str, Path]): Model artifact path.

    Returns:
        Path: Validated model artifact path.

    Raises:
        ModelArtifactError: If model artifact does not exist.
    """
    path = Path(model_path)

    if not path.exists() or not path.is_file():
        raise ModelArtifactError(
            message="Model artifact file was not found.",
            details={"model_path": str(path)},
        )

    return path


# ============================================================
# Safe Execution Decorators
# ============================================================

def handle_exceptions(
    logger: Optional[logging.Logger] = None,
    default_return: Optional[Any] = None,
    raise_error: bool = True,
) -> Callable[[F], F]:
    """
    Decorator to handle exceptions safely.

    Args:
        logger (Optional[logging.Logger]): Logger for exception logging.
        default_return (Optional[Any]): Value returned if raise_error is False.
        raise_error (bool): Whether to re-raise the exception.

    Returns:
        Callable[[F], F]: Decorated function.

    Example:
        @handle_exceptions(logger=my_logger, raise_error=True)
        def train_model():
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)

            except FinancialPortfolioException as error:
                if logger:
                    log_exception(
                        logger=logger,
                        error=error,
                        message=f"Project error in {func.__name__}",
                    )

                if raise_error:
                    raise

                return default_return

            except Exception as error:
                if logger:
                    log_exception(
                        logger=logger,
                        error=error,
                        message=f"Unexpected error in {func.__name__}",
                    )

                if raise_error:
                    raise FinancialPortfolioException(
                        message=f"Unexpected error in {func.__name__}: {error}",
                        error_code="UNEXPECTED_ERROR",
                        details=get_error_details(error),
                    ) from error

                return default_return

        return wrapper  # type: ignore[return-value]

    return decorator


def safe_run(
    function: Callable[..., Any],
    *args: Any,
    logger: Optional[logging.Logger] = None,
    default_return: Optional[Any] = None,
    raise_error: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Run a function safely with exception handling.

    Args:
        function (Callable[..., Any]): Function to execute.
        *args (Any): Positional arguments.
        logger (Optional[logging.Logger]): Logger object.
        default_return (Optional[Any]): Return value if error occurs.
        raise_error (bool): Whether to raise error after catching it.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: Function result or default return value.
    """
    try:
        return function(*args, **kwargs)

    except Exception as error:
        if logger:
            log_exception(
                logger=logger,
                error=error,
                message=f"Error while running {function.__name__}",
            )

        if raise_error:
            raise

        return default_return


# ============================================================
# API-Friendly Error Response
# ============================================================

def build_error_response(
    error: Exception,
    status: str = "error",
) -> dict:
    """
    Build a dictionary error response for API or dashboard use.

    Args:
        error (Exception): Exception object.
        status (str): Response status.

    Returns:
        dict: Error response dictionary.
    """
    if isinstance(error, FinancialPortfolioException):
        return {
            "status": status,
            "error_code": error.error_code,
            "message": error.message,
            "details": error.details,
        }

    return {
        "status": status,
        "error_code": "UNEXPECTED_ERROR",
        "message": str(error),
        "details": {
            "error_type": error.__class__.__name__,
        },
    }


# ============================================================
# Script Usage Example
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_logger = logging.getLogger("exception_handler_test")

    try:
        validate_file_exists("data/raw/financial_portfolio_data.csv")
    except Exception as exc:
        log_exception(test_logger, exc, "Validation test failed")
        print(build_error_response(exc))