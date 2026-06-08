"""
Logging setup for the Financial Portfolio Forecasting
and Anomaly Detection project.

This module provides:
- YAML-based logging configuration
- Automatic log directory creation
- Console and file logging
- Reusable logger function for all modules
"""

import logging
import logging.config
from pathlib import Path
from typing import Optional

import yaml


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGGING_CONFIG_PATH = PROJECT_ROOT / "config" / "logging.yaml"
LOG_DIR = PROJECT_ROOT / "logs"


# ============================================================
# Directory Setup
# ============================================================

def create_log_directory() -> None:
    """
    Create the logs directory if it does not already exist.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# YAML Loader
# ============================================================

def load_logging_yaml(config_path: Path = LOGGING_CONFIG_PATH) -> dict:
    """
    Load logging configuration from a YAML file.

    Args:
        config_path (Path): Path to logging.yaml.

    Returns:
        dict: Logging configuration dictionary.

    Raises:
        FileNotFoundError: If logging.yaml does not exist.
        yaml.YAMLError: If logging.yaml has invalid YAML syntax.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Logging configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config if config is not None else {}


# ============================================================
# Logging Setup
# ============================================================

def setup_logging(
    config_path: Optional[Path] = None,
    default_level: int = logging.INFO,
) -> None:
    """
    Set up logging for the project.

    This function first tries to load logging settings from
    config/logging.yaml. If the file is missing or invalid,
    it falls back to a simple basic logging configuration.

    Args:
        config_path (Optional[Path]): Optional custom logging config path.
        default_level (int): Default logging level.
    """
    create_log_directory()

    path = config_path or LOGGING_CONFIG_PATH

    try:
        logging_config = load_logging_yaml(path)

        logging.config.dictConfig(logging_config)

        logging.getLogger(__name__).info(
            "Logging configured successfully using %s",
            path,
        )

    except Exception as error:
        logging.basicConfig(
            level=default_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        logging.getLogger(__name__).warning(
            "Failed to load logging configuration from %s. "
            "Using basic logging instead. Error: %s",
            path,
            error,
        )


# ============================================================
# Logger Getter
# ============================================================

def get_logger(name: str = "financial_portfolio") -> logging.Logger:
    """
    Get a configured logger.

    Args:
        name (str): Logger name.

    Returns:
        logging.Logger: Configured logger instance.

    Example:
        logger = get_logger(__name__)
        logger.info("Pipeline started")
    """
    if not logging.getLogger().handlers:
        setup_logging()

    return logging.getLogger(name)


# ============================================================
# Named Project Loggers
# ============================================================

def get_project_logger() -> logging.Logger:
    """
    Get the main project logger.

    Returns:
        logging.Logger: Main project logger.
    """
    return get_logger("financial_portfolio")


def get_training_logger() -> logging.Logger:
    """
    Get the training logger.

    Returns:
        logging.Logger: Training logger.
    """
    return get_logger("training")


def get_evaluation_logger() -> logging.Logger:
    """
    Get the evaluation logger.

    Returns:
        logging.Logger: Evaluation logger.
    """
    return get_logger("evaluation")


def get_api_logger() -> logging.Logger:
    """
    Get the API logger.

    Returns:
        logging.Logger: API logger.
    """
    return get_logger("api")


def get_dashboard_logger() -> logging.Logger:
    """
    Get the dashboard logger.

    Returns:
        logging.Logger: Dashboard logger.
    """
    return get_logger("financial_portfolio.dashboard")


def get_anomaly_logger() -> logging.Logger:
    """
    Get the anomaly detection logger.

    Returns:
        logging.Logger: Anomaly detection logger.
    """
    return get_logger("financial_portfolio.anomaly_detection")


def get_forecasting_logger() -> logging.Logger:
    """
    Get the forecasting logger.

    Returns:
        logging.Logger: Forecasting logger.
    """
    return get_logger("financial_portfolio.forecasting")


# ============================================================
# Logging Helper Functions
# ============================================================

def log_section(logger: logging.Logger, title: str) -> None:
    """
    Log a formatted section title.

    Args:
        logger (logging.Logger): Logger instance.
        title (str): Section title.
    """
    separator = "=" * 70
    logger.info(separator)
    logger.info(title)
    logger.info(separator)


def log_subsection(logger: logging.Logger, title: str) -> None:
    """
    Log a formatted subsection title.

    Args:
        logger (logging.Logger): Logger instance.
        title (str): Subsection title.
    """
    separator = "-" * 70
    logger.info(separator)
    logger.info(title)
    logger.info(separator)


def log_dataframe_info(
    logger: logging.Logger,
    dataframe,
    dataframe_name: str = "DataFrame",
) -> None:
    """
    Log basic information about a pandas DataFrame.

    Args:
        logger (logging.Logger): Logger instance.
        dataframe: pandas DataFrame.
        dataframe_name (str): Name of the DataFrame.
    """
    try:
        logger.info("%s shape: %s", dataframe_name, dataframe.shape)
        logger.info("%s columns: %s", dataframe_name, list(dataframe.columns))
        logger.info(
            "%s missing values: %s",
            dataframe_name,
            int(dataframe.isnull().sum().sum()),
        )
    except Exception as error:
        logger.warning(
            "Could not log dataframe information for %s. Error: %s",
            dataframe_name,
            error,
        )


def log_model_metrics(
    logger: logging.Logger,
    metrics: dict,
    model_name: str = "Model",
) -> None:
    """
    Log model evaluation metrics.

    Args:
        logger (logging.Logger): Logger instance.
        metrics (dict): Dictionary of metric names and values.
        model_name (str): Model name.
    """
    logger.info("%s evaluation metrics:", model_name)

    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            logger.info("%s: %.6f", metric_name, metric_value)
        else:
            logger.info("%s: %s", metric_name, metric_value)


def log_file_saved(
    logger: logging.Logger,
    file_path: Path,
    description: str = "File",
) -> None:
    """
    Log that a file has been saved.

    Args:
        logger (logging.Logger): Logger instance.
        file_path (Path): Saved file path.
        description (str): Description of saved file.
    """
    logger.info("%s saved successfully at: %s", description, file_path)


def log_file_loaded(
    logger: logging.Logger,
    file_path: Path,
    description: str = "File",
) -> None:
    """
    Log that a file has been loaded.

    Args:
        logger (logging.Logger): Logger instance.
        file_path (Path): Loaded file path.
        description (str): Description of loaded file.
    """
    logger.info("%s loaded successfully from: %s", description, file_path)


def log_step_start(logger: logging.Logger, step_name: str) -> None:
    """
    Log the start of a pipeline step.

    Args:
        logger (logging.Logger): Logger instance.
        step_name (str): Step name.
    """
    logger.info("Started step: %s", step_name)


def log_step_end(logger: logging.Logger, step_name: str) -> None:
    """
    Log the successful completion of a pipeline step.

    Args:
        logger (logging.Logger): Logger instance.
        step_name (str): Step name.
    """
    logger.info("Completed step: %s", step_name)


def log_step_error(
    logger: logging.Logger,
    step_name: str,
    error: Exception,
) -> None:
    """
    Log an error from a pipeline step.

    Args:
        logger (logging.Logger): Logger instance.
        step_name (str): Step name.
        error (Exception): Exception object.
    """
    logger.error(
        "Error in step '%s': %s",
        step_name,
        error,
        exc_info=True,
    )


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    setup_logging()

    logger = get_project_logger()

    log_section(logger, "Financial Portfolio Logging Test")
    logger.info("Logger is working successfully.")
    logger.warning("This is a sample warning message.")
    logger.error("This is a sample error message.")