"""
Configuration utilities for the Financial Portfolio Forecasting
and Anomaly Detection project.

This module handles:
- Loading YAML configuration files
- Accessing project, data, model, API, and dashboard settings
- Providing safe default values
- Supporting environment variable overrides
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# ============================================================
# Project Root and Config Files
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = PROJECT_ROOT / "config"
MAIN_CONFIG_FILE = CONFIG_DIR / "config.yaml"
PATHS_CONFIG_FILE = CONFIG_DIR / "paths.yaml"
LOGGING_CONFIG_FILE = CONFIG_DIR / "logging.yaml"


# ============================================================
# YAML Utilities
# ============================================================

def load_yaml(file_path: Path) -> Dict[str, Any]:
    """
    Load a YAML file safely.

    Args:
        file_path (Path): Path to YAML file.

    Returns:
        Dict[str, Any]: Parsed YAML data.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    return data if data is not None else {}


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load the main project configuration from config/config.yaml.

    Args:
        config_path (Optional[Path]): Optional custom config file path.

    Returns:
        Dict[str, Any]: Main configuration dictionary.
    """
    path = config_path or MAIN_CONFIG_FILE
    return load_yaml(path)


def load_paths_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load path configuration from config/paths.yaml.

    Args:
        config_path (Optional[Path]): Optional custom paths config file.

    Returns:
        Dict[str, Any]: Paths configuration dictionary.
    """
    path = config_path or PATHS_CONFIG_FILE
    return load_yaml(path)


def load_logging_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load logging configuration from config/logging.yaml.

    Args:
        config_path (Optional[Path]): Optional custom logging config file.

    Returns:
        Dict[str, Any]: Logging configuration dictionary.
    """
    path = config_path or LOGGING_CONFIG_FILE
    return load_yaml(path)


# ============================================================
# Nested Config Access
# ============================================================

def get_nested_value(
    config: Dict[str, Any],
    keys: str,
    default: Optional[Any] = None,
) -> Any:
    """
    Get a nested value from a dictionary using dot notation.

    Example:
        get_nested_value(config, "forecasting.forecast_horizon_days")

    Args:
        config (Dict[str, Any]): Configuration dictionary.
        keys (str): Dot-separated nested key path.
        default (Optional[Any]): Default value if key is not found.

    Returns:
        Any: Retrieved value or default.
    """
    current_value: Any = config

    for key in keys.split("."):
        if isinstance(current_value, dict) and key in current_value:
            current_value = current_value[key]
        else:
            return default

    return current_value


def get_config_value(
    keys: str,
    default: Optional[Any] = None,
    config_path: Optional[Path] = None,
) -> Any:
    """
    Get a value from config/config.yaml using dot notation.

    Example:
        get_config_value("project.random_state")

    Args:
        keys (str): Dot-separated nested key path.
        default (Optional[Any]): Default value if key is missing.
        config_path (Optional[Path]): Optional custom config file path.

    Returns:
        Any: Retrieved configuration value or default.
    """
    config = load_config(config_path)
    return get_nested_value(config, keys, default)


# ============================================================
# Environment Variable Helpers
# ============================================================

def get_env_value(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get an environment variable value.

    Args:
        key (str): Environment variable name.
        default (Optional[str]): Default value if variable is missing.

    Returns:
        Optional[str]: Environment variable value or default.
    """
    return os.getenv(key, default)


def get_env_bool(key: str, default: bool = False) -> bool:
    """
    Get a boolean environment variable.

    Accepted true values:
    - true
    - 1
    - yes
    - y

    Args:
        key (str): Environment variable name.
        default (bool): Default value.

    Returns:
        bool: Parsed boolean value.
    """
    value = os.getenv(key)

    if value is None:
        return default

    return value.lower() in {"true", "1", "yes", "y"}


def get_env_int(key: str, default: int) -> int:
    """
    Get an integer environment variable.

    Args:
        key (str): Environment variable name.
        default (int): Default value if parsing fails.

    Returns:
        int: Parsed integer value.
    """
    value = os.getenv(key)

    if value is None:
        return default

    try:
        return int(value)
    except ValueError:
        return default


def get_env_float(key: str, default: float) -> float:
    """
    Get a float environment variable.

    Args:
        key (str): Environment variable name.
        default (float): Default value if parsing fails.

    Returns:
        float: Parsed float value.
    """
    value = os.getenv(key)

    if value is None:
        return default

    try:
        return float(value)
    except ValueError:
        return default


# ============================================================
# Project Configuration
# ============================================================

def get_project_config() -> Dict[str, Any]:
    """
    Get project-level configuration.

    Returns:
        Dict[str, Any]: Project configuration.
    """
    return load_config().get("project", {})


def get_project_name() -> str:
    """
    Get project name.

    Returns:
        str: Project name.
    """
    return str(
        get_env_value(
            "PROJECT_NAME",
            get_config_value(
                "project.name",
                "financial-portfolio-forecasting-anomaly-detection",
            ),
        )
    )


def get_project_version() -> str:
    """
    Get project version.

    Returns:
        str: Project version.
    """
    return str(
        get_env_value(
            "PROJECT_VERSION",
            get_config_value("project.version", "0.1.0"),
        )
    )


def get_environment() -> str:
    """
    Get project environment.

    Returns:
        str: Environment name.
    """
    return str(
        get_env_value(
            "ENVIRONMENT",
            get_config_value("project.environment", "development"),
        )
    )


def get_random_state() -> int:
    """
    Get random state value.

    Returns:
        int: Random seed.
    """
    return get_env_int(
        "RANDOM_STATE",
        int(get_config_value("project.random_state", 42)),
    )


# ============================================================
# Data Configuration
# ============================================================

def get_data_config() -> Dict[str, Any]:
    """
    Get data configuration.

    Returns:
        Dict[str, Any]: Data configuration.
    """
    return load_config().get("data", {})


def get_date_column() -> str:
    """
    Get date column name.

    Returns:
        str: Date column name.
    """
    return str(
        get_env_value(
            "DATE_COLUMN",
            get_config_value("data.date_column", "date"),
        )
    )


def get_target_column() -> str:
    """
    Get target column name for forecasting.

    Returns:
        str: Target column name.
    """
    return str(
        get_env_value(
            "TARGET_COLUMN",
            get_config_value("data.target_column", "portfolio_value"),
        )
    )


def get_required_columns() -> list:
    """
    Get required dataset columns.

    Returns:
        list: Required column names.
    """
    return list(get_config_value("data.required_columns", []))


def get_numerical_columns() -> list:
    """
    Get numerical columns.

    Returns:
        list: Numerical column names.
    """
    return list(get_config_value("data.numerical_columns", []))


def get_categorical_columns() -> list:
    """
    Get categorical columns.

    Returns:
        list: Categorical column names.
    """
    return list(get_config_value("data.categorical_columns", []))


# ============================================================
# Cleaning Configuration
# ============================================================

def get_cleaning_config() -> Dict[str, Any]:
    """
    Get data cleaning configuration.

    Returns:
        Dict[str, Any]: Cleaning configuration.
    """
    return load_config().get("cleaning", {})


def should_remove_duplicates() -> bool:
    """
    Check whether duplicate rows should be removed.

    Returns:
        bool: True if duplicate removal is enabled.
    """
    return bool(get_config_value("cleaning.remove_duplicates", True))


def should_handle_missing_values() -> bool:
    """
    Check whether missing values should be handled.

    Returns:
        bool: True if missing value handling is enabled.
    """
    return bool(get_config_value("cleaning.handle_missing_values", True))


def get_numeric_missing_strategy() -> str:
    """
    Get numeric missing value strategy.

    Returns:
        str: Missing value strategy.
    """
    return str(get_config_value("cleaning.missing_strategy_numeric", "median"))


def get_categorical_missing_strategy() -> str:
    """
    Get categorical missing value strategy.

    Returns:
        str: Missing value strategy.
    """
    return str(get_config_value("cleaning.missing_strategy_categorical", "mode"))


# ============================================================
# Feature Engineering Configuration
# ============================================================

def get_feature_config() -> Dict[str, Any]:
    """
    Get feature engineering configuration.

    Returns:
        Dict[str, Any]: Feature configuration.
    """
    return load_config().get("features", {})


def get_rolling_windows() -> list:
    """
    Get rolling window periods.

    Returns:
        list: Rolling window values.
    """
    return list(get_config_value("features.rolling_windows", [7, 14, 30]))


def get_lag_periods() -> list:
    """
    Get lag periods.

    Returns:
        list: Lag period values.
    """
    return list(get_config_value("features.lag_periods", [1, 3, 7]))


def get_time_features() -> list:
    """
    Get time-based features.

    Returns:
        list: Time feature names.
    """
    return list(
        get_config_value(
            "features.time_features",
            [
                "day",
                "month",
                "quarter",
                "year",
                "day_of_week",
                "is_month_start",
                "is_month_end",
            ],
        )
    )


# ============================================================
# Split Configuration
# ============================================================

def get_split_config() -> Dict[str, Any]:
    """
    Get train-test split configuration.

    Returns:
        Dict[str, Any]: Split configuration.
    """
    return load_config().get("split", {})


def get_test_size() -> float:
    """
    Get test size.

    Returns:
        float: Test data size.
    """
    return get_env_float(
        "TEST_SIZE",
        float(get_config_value("split.test_size", 0.2)),
    )


def get_validation_size() -> float:
    """
    Get validation size.

    Returns:
        float: Validation data size.
    """
    return float(get_config_value("split.validation_size", 0.1))


def should_shuffle_split() -> bool:
    """
    Check whether train-test split should be shuffled.

    Returns:
        bool: True if shuffle is enabled.
    """
    return bool(get_config_value("split.shuffle", False))


# ============================================================
# Forecasting Configuration
# ============================================================

def get_forecasting_config() -> Dict[str, Any]:
    """
    Get forecasting configuration.

    Returns:
        Dict[str, Any]: Forecasting configuration.
    """
    return load_config().get("forecasting", {})


def get_forecast_horizon_days() -> int:
    """
    Get forecast horizon in days.

    Returns:
        int: Number of forecast days.
    """
    return get_env_int(
        "FORECAST_HORIZON_DAYS",
        int(get_config_value("forecasting.forecast_horizon_days", 30)),
    )


def get_forecasting_models_config() -> Dict[str, Any]:
    """
    Get forecasting model configurations.

    Returns:
        Dict[str, Any]: Model configuration dictionary.
    """
    return dict(get_config_value("forecasting.models", {}))


def get_evaluation_metrics() -> list:
    """
    Get model evaluation metrics.

    Returns:
        list: Evaluation metric names.
    """
    return list(
        get_config_value(
            "forecasting.evaluation_metrics",
            ["mae", "mse", "rmse", "mape", "r2_score"],
        )
    )


def get_best_model_metric() -> str:
    """
    Get metric used to select the best model.

    Returns:
        str: Metric name.
    """
    return str(get_config_value("forecasting.best_model_metric", "rmse"))


def is_lower_metric_better() -> bool:
    """
    Check whether lower metric value is better.

    Returns:
        bool: True if lower value is better.
    """
    return bool(get_config_value("forecasting.lower_is_better", True))


# ============================================================
# Anomaly Detection Configuration
# ============================================================

def get_anomaly_config() -> Dict[str, Any]:
    """
    Get anomaly detection configuration.

    Returns:
        Dict[str, Any]: Anomaly detection configuration.
    """
    return load_config().get("anomaly_detection", {})


def get_anomaly_features() -> list:
    """
    Get anomaly detection feature columns.

    Returns:
        list: Feature column names.
    """
    return list(
        get_config_value(
            "anomaly_detection.anomaly_features",
            ["portfolio_value", "daily_return", "asset_value"],
        )
    )


def get_anomaly_contamination() -> float:
    """
    Get Isolation Forest contamination value.

    Returns:
        float: Contamination rate.
    """
    return get_env_float(
        "ANOMALY_CONTAMINATION",
        float(
            get_config_value(
                "anomaly_detection.methods.isolation_forest.contamination",
                0.05,
            )
        ),
    )


def get_z_score_threshold() -> float:
    """
    Get Z-score anomaly threshold.

    Returns:
        float: Z-score threshold.
    """
    return float(
        get_config_value(
            "anomaly_detection.methods.z_score.threshold",
            3.0,
        )
    )


def get_iqr_multiplier() -> float:
    """
    Get IQR anomaly multiplier.

    Returns:
        float: IQR multiplier.
    """
    return float(
        get_config_value(
            "anomaly_detection.methods.iqr.multiplier",
            1.5,
        )
    )


# ============================================================
# API Configuration
# ============================================================

def get_api_config() -> Dict[str, Any]:
    """
    Get API configuration.

    Returns:
        Dict[str, Any]: API configuration.
    """
    return load_config().get("api", {})


def get_api_host() -> str:
    """
    Get API host.

    Returns:
        str: API host.
    """
    return str(
        get_env_value(
            "API_HOST",
            get_config_value("api.host", "0.0.0.0"),
        )
    )


def get_api_port() -> int:
    """
    Get API port.

    Returns:
        int: API port.
    """
    return get_env_int(
        "API_PORT",
        int(get_config_value("api.port", 8000)),
    )


def get_api_reload() -> bool:
    """
    Get API reload setting.

    Returns:
        bool: True if reload is enabled.
    """
    return get_env_bool(
        "API_RELOAD",
        bool(get_config_value("api.reload", True)),
    )


# ============================================================
# Dashboard Configuration
# ============================================================

def get_dashboard_config() -> Dict[str, Any]:
    """
    Get dashboard configuration.

    Returns:
        Dict[str, Any]: Dashboard configuration.
    """
    return load_config().get("dashboard", {})


def get_dashboard_host() -> str:
    """
    Get dashboard host.

    Returns:
        str: Streamlit host.
    """
    return str(
        get_env_value(
            "STREAMLIT_HOST",
            get_config_value("dashboard.host", "0.0.0.0"),
        )
    )


def get_dashboard_port() -> int:
    """
    Get dashboard port.

    Returns:
        int: Streamlit port.
    """
    return get_env_int(
        "STREAMLIT_PORT",
        int(get_config_value("dashboard.port", 8501)),
    )


# ============================================================
# Model Configuration
# ============================================================

def get_model_config() -> Dict[str, Any]:
    """
    Get model saving configuration.

    Returns:
        Dict[str, Any]: Model configuration.
    """
    return load_config().get("model", {})


def should_save_model() -> bool:
    """
    Check whether trained models should be saved.

    Returns:
        bool: True if model saving is enabled.
    """
    return bool(get_config_value("model.save_model", True))


def get_model_format() -> str:
    """
    Get model serialization format.

    Returns:
        str: Model format.
    """
    return str(get_config_value("model.model_format", "pickle"))


# ============================================================
# Report Configuration
# ============================================================

def get_reports_config() -> Dict[str, Any]:
    """
    Get report configuration.

    Returns:
        Dict[str, Any]: Reports configuration.
    """
    return load_config().get("reports", {})


def should_generate_model_report() -> bool:
    """
    Check whether model report generation is enabled.

    Returns:
        bool: True if enabled.
    """
    return bool(get_config_value("reports.generate_model_report", True))


def should_generate_anomaly_report() -> bool:
    """
    Check whether anomaly report generation is enabled.

    Returns:
        bool: True if enabled.
    """
    return bool(get_config_value("reports.generate_anomaly_report", True))


def should_generate_forecast_report() -> bool:
    """
    Check whether forecast report generation is enabled.

    Returns:
        bool: True if enabled.
    """
    return bool(get_config_value("reports.generate_forecast_report", True))


# ============================================================
# MLflow Configuration
# ============================================================

def get_mlflow_config() -> Dict[str, Any]:
    """
    Get MLflow configuration.

    Returns:
        Dict[str, Any]: MLflow configuration.
    """
    return load_config().get("mlflow", {})


def is_mlflow_enabled() -> bool:
    """
    Check whether MLflow tracking is enabled.

    Returns:
        bool: True if MLflow is enabled.
    """
    return bool(get_config_value("mlflow.enabled", False))


# ============================================================
# Full Configuration Bundle
# ============================================================

def get_all_configs() -> Dict[str, Dict[str, Any]]:
    """
    Load all project configuration files.

    Returns:
        Dict[str, Dict[str, Any]]: Main, paths, and logging configs.
    """
    return {
        "config": load_config(),
        "paths": load_paths_config(),
        "logging": load_logging_config(),
    }


# ============================================================
# Script Usage
# ============================================================

if __name__ == "__main__":
    print("Configuration loaded successfully.")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Project name: {get_project_name()}")
    print(f"Environment: {get_environment()}")
    print(f"Target column: {get_target_column()}")
    print(f"Forecast horizon: {get_forecast_horizon_days()} days")
    print(f"API: {get_api_host()}:{get_api_port()}")
    print(f"Dashboard: {get_dashboard_host()}:{get_dashboard_port()}")