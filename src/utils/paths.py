"""
Path utilities for the Financial Portfolio Forecasting
and Anomaly Detection project.

This module handles:
- Loading paths from config/paths.yaml
- Creating required project directories
- Resolving paths from project root
- Safely checking file and folder existence
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# ============================================================
# Project Root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# Default Config Paths
# ============================================================

CONFIG_DIR = PROJECT_ROOT / "config"
PATHS_CONFIG_FILE = CONFIG_DIR / "paths.yaml"


# ============================================================
# YAML Loading
# ============================================================

def load_yaml(file_path: Path) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        file_path (Path): Path to the YAML file.

    Returns:
        Dict[str, Any]: YAML content as a dictionary.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If the YAML file cannot be parsed.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        return {}

    return data


def load_paths_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load the project paths configuration from config/paths.yaml.

    Args:
        config_path (Optional[Path]): Custom path to paths.yaml.

    Returns:
        Dict[str, Any]: Loaded paths configuration.
    """
    path = config_path or PATHS_CONFIG_FILE
    return load_yaml(path)


# ============================================================
# Path Resolving
# ============================================================

def resolve_path(path_value: str) -> Path:
    """
    Resolve a relative path from the project root.

    Args:
        path_value (str): Relative or absolute path as a string.

    Returns:
        Path: Resolved absolute path.
    """
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def get_path(section: str, key: str, config_path: Optional[Path] = None) -> Path:
    """
    Get a specific path from config/paths.yaml.

    Example:
        get_path("data", "raw")

    Args:
        section (str): Main section name in paths.yaml.
        key (str): Path key inside the section.
        config_path (Optional[Path]): Optional custom config file path.

    Returns:
        Path: Resolved absolute project path.

    Raises:
        KeyError: If the section or key does not exist.
    """
    paths_config = load_paths_config(config_path)

    if section not in paths_config:
        raise KeyError(f"Section '{section}' not found in paths.yaml")

    if key not in paths_config[section]:
        raise KeyError(
            f"Key '{key}' not found under section '{section}' in paths.yaml"
        )

    return resolve_path(paths_config[section][key])


def get_project_root() -> Path:
    """
    Return the project root directory.

    Returns:
        Path: Project root path.
    """
    return PROJECT_ROOT


# ============================================================
# Directory Creation
# ============================================================

def create_directory(directory_path: Path) -> None:
    """
    Create a directory if it does not already exist.

    Args:
        directory_path (Path): Directory path to create.
    """
    directory_path.mkdir(parents=True, exist_ok=True)


def create_required_directories(config_path: Optional[Path] = None) -> None:
    """
    Create all required project directories from config/paths.yaml.

    This function checks common directory keys ending with:
    - dir
    - _dir

    Args:
        config_path (Optional[Path]): Optional custom config file path.
    """
    paths_config = load_paths_config(config_path)

    for section_data in paths_config.values():
        if isinstance(section_data, dict):
            for key, value in section_data.items():
                if isinstance(value, str) and (
                    key == "dir" or key.endswith("_dir")
                ):
                    create_directory(resolve_path(value))


def create_parent_directory(file_path: Path) -> None:
    """
    Create the parent directory of a file path.

    Args:
        file_path (Path): File path whose parent directory should be created.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# File and Directory Checks
# ============================================================

def path_exists(path_value: str) -> bool:
    """
    Check if a path exists.

    Args:
        path_value (str): Relative or absolute path.

    Returns:
        bool: True if path exists, otherwise False.
    """
    return resolve_path(path_value).exists()


def file_exists(path_value: str) -> bool:
    """
    Check if a file exists.

    Args:
        path_value (str): Relative or absolute file path.

    Returns:
        bool: True if file exists, otherwise False.
    """
    return resolve_path(path_value).is_file()


def directory_exists(path_value: str) -> bool:
    """
    Check if a directory exists.

    Args:
        path_value (str): Relative or absolute directory path.

    Returns:
        bool: True if directory exists, otherwise False.
    """
    return resolve_path(path_value).is_dir()


# ============================================================
# Common Project Paths
# ============================================================

def get_raw_data_path() -> Path:
    """
    Get raw financial portfolio dataset path.

    Returns:
        Path: Raw data CSV path.
    """
    return get_path("data", "raw")


def get_cleaned_data_path() -> Path:
    """
    Get cleaned financial data path.

    Returns:
        Path: Cleaned data CSV path.
    """
    return get_path("data", "cleaned")


def get_train_data_path() -> Path:
    """
    Get training data path.

    Returns:
        Path: Training data CSV path.
    """
    return get_path("data", "train")


def get_validation_data_path() -> Path:
    """
    Get validation data path.

    Returns:
        Path: Validation data CSV path.
    """
    return get_path("data", "validation")


def get_test_data_path() -> Path:
    """
    Get test data path.

    Returns:
        Path: Test data CSV path.
    """
    return get_path("data", "test")


def get_anomaly_results_path() -> Path:
    """
    Get anomaly results data path.

    Returns:
        Path: Anomaly results CSV path.
    """
    return get_path("data", "anomaly_results")


def get_forecast_results_path() -> Path:
    """
    Get forecast results data path.

    Returns:
        Path: Forecast results CSV path.
    """
    return get_path("data", "forecast_results")


def get_model_dir() -> Path:
    """
    Get models directory path.

    Returns:
        Path: Models directory path.
    """
    return get_path("models", "dir")


def get_forecasting_model_path() -> Path:
    """
    Get forecasting model artifact path.

    Returns:
        Path: Forecasting model path.
    """
    return get_path("models", "forecasting_model")


def get_anomaly_model_path() -> Path:
    """
    Get anomaly detection model artifact path.

    Returns:
        Path: Anomaly model path.
    """
    return get_path("models", "anomaly_model")


def get_scaler_path() -> Path:
    """
    Get scaler artifact path.

    Returns:
        Path: Scaler artifact path.
    """
    return get_path("models", "scaler")


def get_feature_columns_path() -> Path:
    """
    Get feature columns JSON path.

    Returns:
        Path: Feature columns path.
    """
    return get_path("models", "feature_columns")


def get_logs_dir() -> Path:
    """
    Get logs directory path.

    Returns:
        Path: Logs directory path.
    """
    return get_path("logs", "dir")


def get_reports_dir() -> Path:
    """
    Get reports directory path.

    Returns:
        Path: Reports directory path.
    """
    return get_path("reports", "dir")


def get_outputs_dir() -> Path:
    """
    Get outputs directory path.

    Returns:
        Path: Outputs directory path.
    """
    return get_path("outputs", "dir")


# ============================================================
# Script Usage
# ============================================================

if __name__ == "__main__":
    create_required_directories()
    print("Required project directories created successfully.")
    print(f"Project root: {PROJECT_ROOT}")