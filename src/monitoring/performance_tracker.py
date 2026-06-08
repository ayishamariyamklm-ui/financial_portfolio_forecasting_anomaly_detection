"""
Performance tracking utilities for the Financial Portfolio Forecasting
and Anomaly Detection project.

This module provides:
- Function execution time tracking
- Pipeline step duration tracking
- Memory usage tracking
- Model metric tracking
- API request tracking
- Performance report saving as JSON and CSV
"""

import csv
import functools
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar

try:
    import psutil
except ImportError:
    psutil = None


try:
    from src.monitoring.logger import get_project_logger
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)
    get_project_logger = lambda: logging.getLogger("performance_tracker")


# ============================================================
# Type Variable
# ============================================================

F = TypeVar("F", bound=Callable[..., Any])


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
LOGS_DIR = PROJECT_ROOT / "logs"

PERFORMANCE_JSON_PATH = TABLES_DIR / "performance_metrics.json"
PERFORMANCE_CSV_PATH = TABLES_DIR / "performance_metrics.csv"


# ============================================================
# Logger
# ============================================================

logger = get_project_logger()


# ============================================================
# Helper Functions
# ============================================================

def ensure_performance_dirs() -> None:
    """
    Create folders required for saving performance reports.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def get_current_timestamp() -> str:
    """
    Get the current timestamp as a formatted string.

    Returns:
        str: Current timestamp.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_process_memory_mb() -> float:
    """
    Get current process memory usage in MB.

    Returns:
        float: Memory usage in megabytes.
    """
    if psutil is None:
        return 0.0

    process = psutil.Process(os.getpid())
    memory_bytes = process.memory_info().rss

    return round(memory_bytes / (1024 * 1024), 4)


def get_system_cpu_percent() -> float:
    """
    Get current system CPU usage percentage.

    Returns:
        float: CPU usage percentage.
    """
    if psutil is None:
        return 0.0

    return float(psutil.cpu_percent(interval=0.1))


def safe_round(value: Any, digits: int = 6) -> Any:
    """
    Round numeric values safely.

    Args:
        value (Any): Value to round.
        digits (int): Number of decimal places.

    Returns:
        Any: Rounded value if numeric, otherwise original value.
    """
    if isinstance(value, (int, float)):
        return round(value, digits)

    return value


# ============================================================
# Performance Tracker Class
# ============================================================

class PerformanceTracker:
    """
    Track performance metrics for scripts, pipelines, models, and API calls.
    """

    def __init__(self) -> None:
        """
        Initialize the performance tracker.
        """
        ensure_performance_dirs()
        self.records: List[Dict[str, Any]] = []

    def add_record(
        self,
        step_name: str,
        execution_time_seconds: float,
        status: str = "success",
        category: str = "general",
        memory_usage_mb: Optional[float] = None,
        cpu_percent: Optional[float] = None,
        extra_metrics: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Add a performance record.

        Args:
            step_name (str): Name of the tracked step.
            execution_time_seconds (float): Duration in seconds.
            status (str): Execution status.
            category (str): Record category.
            memory_usage_mb (Optional[float]): Memory usage in MB.
            cpu_percent (Optional[float]): CPU usage percentage.
            extra_metrics (Optional[Dict[str, Any]]): Extra metric values.
            error_message (Optional[str]): Error message if execution failed.

        Returns:
            Dict[str, Any]: Created performance record.
        """
        record = {
            "timestamp": get_current_timestamp(),
            "category": category,
            "step_name": step_name,
            "status": status,
            "execution_time_seconds": round(execution_time_seconds, 6),
            "memory_usage_mb": (
                memory_usage_mb
                if memory_usage_mb is not None
                else get_process_memory_mb()
            ),
            "cpu_percent": (
                cpu_percent
                if cpu_percent is not None
                else get_system_cpu_percent()
            ),
            "error_message": error_message,
        }

        if extra_metrics:
            for key, value in extra_metrics.items():
                record[key] = safe_round(value)

        self.records.append(record)

        logger.info(
            "Performance tracked | step=%s | status=%s | time=%.4fs | memory=%.2fMB",
            step_name,
            status,
            execution_time_seconds,
            record["memory_usage_mb"],
        )

        return record

    def track_metric(
        self,
        metric_name: str,
        metric_value: Any,
        category: str = "metric",
        step_name: str = "model_metric",
    ) -> Dict[str, Any]:
        """
        Track a single metric value.

        Args:
            metric_name (str): Metric name.
            metric_value (Any): Metric value.
            category (str): Metric category.
            step_name (str): Step name.

        Returns:
            Dict[str, Any]: Created metric record.
        """
        return self.add_record(
            step_name=step_name,
            execution_time_seconds=0.0,
            category=category,
            extra_metrics={metric_name: metric_value},
        )

    def track_model_metrics(
        self,
        model_name: str,
        metrics: Dict[str, Any],
        category: str = "model",
    ) -> Dict[str, Any]:
        """
        Track evaluation metrics for a model.

        Args:
            model_name (str): Model name.
            metrics (Dict[str, Any]): Model metrics dictionary.
            category (str): Metric category.

        Returns:
            Dict[str, Any]: Created model metric record.
        """
        clean_metrics = {
            key: safe_round(value)
            for key, value in metrics.items()
        }

        return self.add_record(
            step_name=model_name,
            execution_time_seconds=0.0,
            category=category,
            extra_metrics=clean_metrics,
        )

    def track_api_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        execution_time_seconds: float,
        request_count: int = 1,
        error_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Track API request performance.

        Args:
            endpoint (str): API endpoint.
            method (str): HTTP method.
            status_code (int): HTTP response status code.
            execution_time_seconds (float): Request duration.
            request_count (int): Number of requests.
            error_message (Optional[str]): Error message if request failed.

        Returns:
            Dict[str, Any]: Created API performance record.
        """
        status = "success" if 200 <= status_code < 400 else "failed"

        return self.add_record(
            step_name=endpoint,
            execution_time_seconds=execution_time_seconds,
            status=status,
            category="api",
            extra_metrics={
                "method": method,
                "status_code": status_code,
                "request_count": request_count,
            },
            error_message=error_message,
        )

    def get_records(self) -> List[Dict[str, Any]]:
        """
        Get all performance records.

        Returns:
            List[Dict[str, Any]]: List of performance records.
        """
        return self.records

    def get_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of tracked performance records.

        Returns:
            Dict[str, Any]: Performance summary.
        """
        if not self.records:
            return {
                "total_records": 0,
                "total_execution_time_seconds": 0.0,
                "average_execution_time_seconds": 0.0,
                "successful_steps": 0,
                "failed_steps": 0,
            }

        total_time = sum(
            float(record.get("execution_time_seconds", 0.0))
            for record in self.records
        )

        successful_steps = sum(
            1 for record in self.records if record.get("status") == "success"
        )

        failed_steps = sum(
            1 for record in self.records if record.get("status") == "failed"
        )

        return {
            "total_records": len(self.records),
            "total_execution_time_seconds": round(total_time, 6),
            "average_execution_time_seconds": round(
                total_time / len(self.records),
                6,
            ),
            "successful_steps": successful_steps,
            "failed_steps": failed_steps,
            "categories": sorted(
                set(str(record.get("category", "general")) for record in self.records)
            ),
            "generated_at": get_current_timestamp(),
        }

    def save_json(
        self,
        file_path: Path = PERFORMANCE_JSON_PATH,
        include_summary: bool = True,
    ) -> None:
        """
        Save performance records to a JSON file.

        Args:
            file_path (Path): Output JSON path.
            include_summary (bool): Whether to include performance summary.
        """
        ensure_performance_dirs()
        file_path.parent.mkdir(parents=True, exist_ok=True)

        output = {
            "records": self.records,
        }

        if include_summary:
            output["summary"] = self.get_summary()

        with file_path.open("w", encoding="utf-8") as file:
            json.dump(output, file, indent=4)

        logger.info("Performance JSON report saved at: %s", file_path)

    def save_csv(
        self,
        file_path: Path = PERFORMANCE_CSV_PATH,
    ) -> None:
        """
        Save performance records to a CSV file.

        Args:
            file_path (Path): Output CSV path.
        """
        ensure_performance_dirs()
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.records:
            logger.warning("No performance records available to save as CSV.")
            return

        all_fieldnames = sorted(
            {
                key
                for record in self.records
                for key in record.keys()
            }
        )

        preferred_order = [
            "timestamp",
            "category",
            "step_name",
            "status",
            "execution_time_seconds",
            "memory_usage_mb",
            "cpu_percent",
            "error_message",
        ]

        fieldnames = preferred_order + [
            field
            for field in all_fieldnames
            if field not in preferred_order
        ]

        with file_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)

        logger.info("Performance CSV report saved at: %s", file_path)

    def save_reports(self) -> None:
        """
        Save both JSON and CSV performance reports.
        """
        self.save_json()
        self.save_csv()


# ============================================================
# Global Tracker
# ============================================================

performance_tracker = PerformanceTracker()


# ============================================================
# Decorators and Context Managers
# ============================================================

def track_execution_time(
    step_name: Optional[str] = None,
    category: str = "function",
    tracker: PerformanceTracker = performance_tracker,
    raise_error: bool = True,
) -> Callable[[F], F]:
    """
    Decorator to track execution time of a function.

    Args:
        step_name (Optional[str]): Custom step name.
        category (str): Category name.
        tracker (PerformanceTracker): Performance tracker instance.
        raise_error (bool): Whether to raise exceptions after tracking.

    Returns:
        Callable[[F], F]: Decorated function.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            name = step_name or func.__name__
            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                end_time = time.perf_counter()

                tracker.add_record(
                    step_name=name,
                    execution_time_seconds=end_time - start_time,
                    status="success",
                    category=category,
                )

                return result

            except Exception as error:
                end_time = time.perf_counter()

                tracker.add_record(
                    step_name=name,
                    execution_time_seconds=end_time - start_time,
                    status="failed",
                    category=category,
                    error_message=str(error),
                )

                logger.error(
                    "Execution failed for %s: %s",
                    name,
                    error,
                    exc_info=True,
                )

                if raise_error:
                    raise

                return None

        return wrapper  # type: ignore[return-value]

    return decorator


@contextmanager
def track_step(
    step_name: str,
    category: str = "pipeline",
    tracker: PerformanceTracker = performance_tracker,
    raise_error: bool = True,
) -> Generator[None, None, None]:
    """
    Context manager to track execution time of a code block.

    Args:
        step_name (str): Step name.
        category (str): Category name.
        tracker (PerformanceTracker): Performance tracker instance.
        raise_error (bool): Whether to raise exceptions after tracking.

    Yields:
        None

    Example:
        with track_step("data_cleaning"):
            clean_data()
    """
    start_time = time.perf_counter()

    try:
        yield

        end_time = time.perf_counter()

        tracker.add_record(
            step_name=step_name,
            execution_time_seconds=end_time - start_time,
            status="success",
            category=category,
        )

    except Exception as error:
        end_time = time.perf_counter()

        tracker.add_record(
            step_name=step_name,
            execution_time_seconds=end_time - start_time,
            status="failed",
            category=category,
            error_message=str(error),
        )

        logger.error(
            "Step failed: %s | Error: %s",
            step_name,
            error,
            exc_info=True,
        )

        if raise_error:
            raise


# ============================================================
# Convenience Functions
# ============================================================

def add_performance_record(
    step_name: str,
    execution_time_seconds: float,
    status: str = "success",
    category: str = "general",
    extra_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Add a performance record to the global tracker.

    Args:
        step_name (str): Step name.
        execution_time_seconds (float): Execution duration.
        status (str): Execution status.
        category (str): Category.
        extra_metrics (Optional[Dict[str, Any]]): Extra metrics.

    Returns:
        Dict[str, Any]: Created performance record.
    """
    return performance_tracker.add_record(
        step_name=step_name,
        execution_time_seconds=execution_time_seconds,
        status=status,
        category=category,
        extra_metrics=extra_metrics,
    )


def track_model_performance(
    model_name: str,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Track model performance metrics.

    Args:
        model_name (str): Model name.
        metrics (Dict[str, Any]): Model metrics.

    Returns:
        Dict[str, Any]: Created model performance record.
    """
    return performance_tracker.track_model_metrics(
        model_name=model_name,
        metrics=metrics,
    )


def save_performance_reports() -> None:
    """
    Save global performance reports as JSON and CSV.
    """
    performance_tracker.save_reports()


def get_performance_summary() -> Dict[str, Any]:
    """
    Get summary from the global performance tracker.

    Returns:
        Dict[str, Any]: Performance summary.
    """
    return performance_tracker.get_summary()


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    logger.info("Testing performance tracker...")

    @track_execution_time(step_name="sample_function", category="test")
    def sample_function() -> int:
        time.sleep(1)
        return 100

    with track_step("sample_pipeline_step", category="test"):
        time.sleep(0.5)

    result = sample_function()

    track_model_performance(
        model_name="sample_forecasting_model",
        metrics={
            "mae": 125.45,
            "rmse": 230.78,
            "r2_score": 0.91,
        },
    )

    logger.info("Sample function result: %s", result)
    logger.info("Performance summary: %s", get_performance_summary())

    save_performance_reports()