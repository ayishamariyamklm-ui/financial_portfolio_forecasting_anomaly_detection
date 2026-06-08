"""
Evaluate trained forecasting and anomaly detection models for the
Financial Portfolio Forecasting & Anomaly Detection project.

This script evaluates:

1. Forecasting model outputs
   - ARIMA
   - XGBoost
   - Prophet
   - LSTM

2. Anomaly detection outputs
   - IQR
   - Z-score
   - Isolation Forest

3. Model comparison
   - Best model selection using RMSE

4. Visualization generation
   - Model comparison chart
   - Anomaly charts
   - Basic time-series charts

Run from project root:

    python scripts/evaluate_models.py
"""

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


# ============================================================
# Add Project Root to Python Path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ============================================================
# Project Imports
# ============================================================

from src.evaluation.metrics import (
    evaluate_anomaly_dataframe,
    evaluate_forecast_dataframe,
    save_anomaly_metrics,
    save_forecast_metrics,
)

from src.evaluation.model_comparison import run_model_comparison
from src.evaluation.visualization import (
    generate_anomaly_visualizations,
    generate_basic_visualizations,
    plot_actual_vs_predicted,
    plot_model_comparison,
)
from src.monitoring.logger import (
    get_project_logger,
    log_section,
    log_step_end,
    log_step_error,
    log_step_start,
)
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

REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

ARIMA_METRICS_PATH = TABLES_DIR / "arima_metrics.json"
XGBOOST_METRICS_PATH = TABLES_DIR / "xgboost_metrics.json"
PROPHET_METRICS_PATH = TABLES_DIR / "prophet_metrics.json"
LSTM_METRICS_PATH = TABLES_DIR / "lstm_metrics.json"

XGBOOST_PREDICTIONS_PATH = OUTPUTS_DIR / "xgboost_predictions.csv"
PROPHET_PREDICTIONS_PATH = OUTPUTS_DIR / "prophet_test_predictions.csv"
LSTM_PREDICTIONS_PATH = OUTPUTS_DIR / "lstm_test_predictions.csv"

ANOMALY_RESULTS_PATH = OUTPUTS_DIR / "anomaly_detection_results.csv"

MODEL_COMPARISON_PATH = TABLES_DIR / "model_comparison.csv"
FORECASTING_MODEL_COMPARISON_PATH = TABLES_DIR / "forecasting_model_comparison.csv"

EVALUATION_SUMMARY_JSON_PATH = TABLES_DIR / "evaluation_summary.json"
EVALUATION_SUMMARY_MD_PATH = REPORTS_DIR / "evaluation_summary.md"
FORECAST_EVALUATION_COMBINED_PATH = TABLES_DIR / "forecast_evaluation_combined.csv"


# ============================================================
# Logger
# ============================================================

logger = get_project_logger()


# ============================================================
# Helper Functions
# ============================================================

def ensure_evaluation_directories() -> None:
    """
    Create all required directories for evaluation outputs.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def load_csv_if_exists(file_path: Path) -> Optional[pd.DataFrame]:
    """
    Load a CSV file if it exists.

    Args:
        file_path (Path): CSV file path.

    Returns:
        Optional[pd.DataFrame]: Loaded dataframe or None.
    """
    if not file_path.exists():
        logger.warning("CSV file not found: %s", file_path)
        return None

    try:
        dataframe = pd.read_csv(file_path)
        logger.info("Loaded CSV file: %s with shape %s", file_path, dataframe.shape)
        return dataframe

    except Exception as error:
        logger.error(
            "Failed to load CSV file %s: %s",
            file_path,
            error,
            exc_info=True,
        )
        return None


def load_json_if_exists(file_path: Path) -> Dict[str, Any]:
    """
    Load a JSON file if it exists.

    Args:
        file_path (Path): JSON file path.

    Returns:
        Dict[str, Any]: Loaded JSON dictionary or empty dictionary.
    """
    if not file_path.exists():
        logger.warning("JSON file not found: %s", file_path)
        return {}

    try:
        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        logger.info("Loaded JSON file: %s", file_path)

        return data if isinstance(data, dict) else {}

    except Exception as error:
        logger.error(
            "Failed to load JSON file %s: %s",
            file_path,
            error,
            exc_info=True,
        )
        return {}


def save_json(data: Dict[str, Any], file_path: Path) -> None:
    """
    Save dictionary as JSON.

    Args:
        data (Dict[str, Any]): Data to save.
        file_path (Path): Output JSON path.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, default=str)

    logger.info("JSON saved at: %s", file_path)


def clean_raw_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw financial dataset for visualization/evaluation.

    Args:
        dataframe (pd.DataFrame): Raw dataframe.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    data = dataframe.copy()

    data.columns = (
        data.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    required_columns = ["date", "asset", "price"]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns in raw data: {missing_columns}")

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["price"] = pd.to_numeric(data["price"], errors="coerce")

    data = data.dropna(subset=["date", "asset", "price"])
    data = data[data["price"] >= 0]
    data = data.sort_values(["asset", "date"]).reset_index(drop=True)

    return data


def get_dataset_summary(dataframe: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """
    Generate dataset summary.

    Args:
        dataframe (Optional[pd.DataFrame]): Input dataframe.

    Returns:
        Dict[str, Any]: Dataset summary.
    """
    if dataframe is None or dataframe.empty:
        return {
            "available": False,
            "message": "Raw dataset not available.",
        }

    data = clean_raw_data(dataframe)

    return {
        "available": True,
        "total_records": int(len(data)),
        "total_assets": int(data["asset"].nunique()),
        "assets": sorted(data["asset"].unique().tolist()),
        "start_date": str(data["date"].min()),
        "end_date": str(data["date"].max()),
        "average_price": round(float(data["price"].mean()), 4),
        "minimum_price": round(float(data["price"].min()), 4),
        "maximum_price": round(float(data["price"].max()), 4),
    }


def normalize_prediction_columns(
    dataframe: pd.DataFrame,
    model_name: str,
) -> Optional[pd.DataFrame]:
    """
    Normalize prediction dataframe column names for evaluation.

    Expected final columns:
    - date
    - asset
    - actual_price
    - predicted_price

    Args:
        dataframe (pd.DataFrame): Prediction dataframe.
        model_name (str): Model name.

    Returns:
        Optional[pd.DataFrame]: Normalized dataframe or None.
    """
    if dataframe is None or dataframe.empty:
        return None

    data = dataframe.copy()

    data.columns = (
        data.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    if "actual_price" not in data.columns:
        if "price" in data.columns:
            data = data.rename(columns={"price": "actual_price"})
        elif "y" in data.columns:
            data = data.rename(columns={"y": "actual_price"})

    if "predicted_price" not in data.columns:
        if "forecasted_price" in data.columns:
            data = data.rename(columns={"forecasted_price": "predicted_price"})
        elif "yhat" in data.columns:
            data = data.rename(columns={"yhat": "predicted_price"})

    if "date" not in data.columns:
        if "ds" in data.columns:
            data = data.rename(columns={"ds": "date"})

    if "asset" not in data.columns:
        if "asset_encoded_reference" in data.columns:
            data = data.rename(columns={"asset_encoded_reference": "asset"})
        else:
            data["asset"] = model_name

    required_columns = ["date", "asset", "actual_price", "predicted_price"]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        logger.warning(
            "Skipping %s predictions because required columns are missing: %s",
            model_name,
            missing_columns,
        )
        return None

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["actual_price"] = pd.to_numeric(data["actual_price"], errors="coerce")
    data["predicted_price"] = pd.to_numeric(data["predicted_price"], errors="coerce")

    data = data.dropna(subset=["date", "actual_price", "predicted_price"])

    if data.empty:
        logger.warning("Skipping %s predictions because data is empty after cleaning.", model_name)
        return None

    data["model_name"] = model_name
    data["absolute_error"] = (
        data["actual_price"] - data["predicted_price"]
    ).abs()

    return data


def evaluate_prediction_file(
    file_path: Path,
    model_name: str,
) -> Dict[str, Any]:
    """
    Evaluate one prediction CSV file.

    Args:
        file_path (Path): Prediction CSV path.
        model_name (str): Model name.

    Returns:
        Dict[str, Any]: Evaluation result.
    """
    prediction_dataframe = load_csv_if_exists(file_path)

    if prediction_dataframe is None:
        return {
            "status": "missing",
            "model_name": model_name,
            "file_path": str(file_path),
            "message": "Prediction file not found.",
        }

    normalized_dataframe = normalize_prediction_columns(
        dataframe=prediction_dataframe,
        model_name=model_name,
    )

    if normalized_dataframe is None:
        return {
            "status": "invalid",
            "model_name": model_name,
            "file_path": str(file_path),
            "message": "Prediction file does not contain required columns.",
        }

    metrics = evaluate_forecast_dataframe(
        dataframe=normalized_dataframe,
        actual_column="actual_price",
        predicted_column="predicted_price",
        asset_column="asset",
    )

    model_metrics_path = TABLES_DIR / f"{model_name.lower()}_forecast_evaluation.json"
    save_forecast_metrics(
        metrics=metrics,
        json_path=model_metrics_path,
        csv_path=TABLES_DIR / f"{model_name.lower()}_forecast_evaluation.csv",
    )

    figure_path = FIGURES_DIR / f"{model_name.lower()}_actual_vs_predicted.png"

    try:
        plot_actual_vs_predicted(
            dataframe=normalized_dataframe,
            date_column="date",
            asset_column="asset",
            actual_column="actual_price",
            predicted_column="predicted_price",
            save_path=figure_path,
            show=False,
        )
    except Exception as error:
        logger.warning(
            "Could not create actual vs predicted plot for %s: %s",
            model_name,
            error,
        )

    return {
        "status": "success",
        "model_name": model_name,
        "file_path": str(file_path),
        "record_count": int(len(normalized_dataframe)),
        "overall_metrics": metrics.get("overall_metrics", {}),
        "asset_metrics": metrics.get("asset_metrics", {}),
        "metrics_path": str(model_metrics_path),
        "figure_path": str(figure_path),
    }


def build_forecast_evaluation_table(
    forecast_results: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """
    Build combined forecast evaluation dataframe.

    Args:
        forecast_results (Dict[str, Dict[str, Any]]): Forecast evaluation results.

    Returns:
        pd.DataFrame: Combined evaluation dataframe.
    """
    rows = []

    for model_name, result in forecast_results.items():
        overall_metrics = result.get("overall_metrics", {})

        rows.append(
            {
                "model_name": model_name,
                "status": result.get("status"),
                "record_count": result.get("record_count", 0),
                "mae": overall_metrics.get("mae"),
                "mse": overall_metrics.get("mse"),
                "rmse": overall_metrics.get("rmse"),
                "mape": overall_metrics.get("mape"),
                "smape": overall_metrics.get("smape"),
                "r2_score": overall_metrics.get("r2_score"),
                "file_path": result.get("file_path"),
            }
        )

    evaluation_dataframe = pd.DataFrame(rows)

    if not evaluation_dataframe.empty and "rmse" in evaluation_dataframe.columns:
        evaluation_dataframe["rmse"] = pd.to_numeric(
            evaluation_dataframe["rmse"],
            errors="coerce",
        )

        evaluation_dataframe = evaluation_dataframe.sort_values(
            by="rmse",
            ascending=True,
            na_position="last",
        )

    evaluation_dataframe.to_csv(FORECAST_EVALUATION_COMBINED_PATH, index=False)

    logger.info(
        "Combined forecast evaluation table saved at: %s",
        FORECAST_EVALUATION_COMBINED_PATH,
    )

    return evaluation_dataframe


def evaluate_anomaly_results() -> Dict[str, Any]:
    """
    Evaluate anomaly detection results.

    Returns:
        Dict[str, Any]: Anomaly evaluation summary.
    """
    anomaly_dataframe = load_csv_if_exists(ANOMALY_RESULTS_PATH)

    if anomaly_dataframe is None:
        return {
            "status": "missing",
            "file_path": str(ANOMALY_RESULTS_PATH),
            "message": "Anomaly results file not found.",
        }

    if anomaly_dataframe.empty:
        return {
            "status": "invalid",
            "file_path": str(ANOMALY_RESULTS_PATH),
            "message": "Anomaly results file is empty.",
        }

    metrics = evaluate_anomaly_dataframe(
        dataframe=anomaly_dataframe,
        anomaly_column="is_anomaly",
        asset_column="asset",
        price_column="price",
        top_n=10,
    )

    save_anomaly_metrics(metrics)

    figure_paths = {}

    try:
        figure_paths = generate_anomaly_visualizations(
            anomaly_dataframe=anomaly_dataframe,
            show=False,
        )
    except Exception as error:
        logger.warning("Could not generate anomaly visualizations: %s", error)

    return {
        "status": "success",
        "file_path": str(ANOMALY_RESULTS_PATH),
        "summary": metrics.get("summary", {}),
        "method_summary": metrics.get("method_summary", {}),
        "severity_summary": metrics.get("severity_summary", {}),
        "top_anomalies": metrics.get("top_anomalies", []),
        "metrics_json": str(TABLES_DIR / "anomaly_metrics.json"),
        "metrics_csv": str(TABLES_DIR / "anomaly_metrics.csv"),
        "figure_paths": figure_paths,
    }


def generate_model_comparison_visuals(
    model_comparison_summary: Dict[str, Any],
) -> Optional[str]:
    """
    Generate model comparison visualization.

    Args:
        model_comparison_summary (Dict[str, Any]): Model comparison summary.

    Returns:
        Optional[str]: Saved figure path or None.
    """
    comparison_dataframe = None

    if MODEL_COMPARISON_PATH.exists():
        comparison_dataframe = pd.read_csv(MODEL_COMPARISON_PATH)
    elif FORECASTING_MODEL_COMPARISON_PATH.exists():
        comparison_dataframe = pd.read_csv(FORECASTING_MODEL_COMPARISON_PATH)

    if comparison_dataframe is None or comparison_dataframe.empty:
        records = model_comparison_summary.get("comparison_records", [])

        if records:
            comparison_dataframe = pd.DataFrame(records)

    if comparison_dataframe is None or comparison_dataframe.empty:
        logger.warning("No model comparison data available for visualization.")
        return None

    figure_path = FIGURES_DIR / "evaluation_model_comparison_rmse.png"

    try:
        plot_model_comparison(
            comparison_dataframe=comparison_dataframe,
            metric_column="rmse",
            model_column="model_name",
            save_path=figure_path,
            show=False,
        )

        return str(figure_path)

    except Exception as error:
        logger.warning("Could not generate model comparison plot: %s", error)
        return None


def save_markdown_summary(summary: Dict[str, Any]) -> None:
    """
    Save evaluation summary as Markdown.

    Args:
        summary (Dict[str, Any]): Evaluation summary.
    """
    dataset_summary = summary.get("dataset_summary", {})
    forecast_results = summary.get("forecast_evaluation", {})
    anomaly_results = summary.get("anomaly_evaluation", {})
    comparison_summary = summary.get("model_comparison", {})
    performance_summary = summary.get("performance_summary", {})

    best_model = comparison_summary.get("best_model", {})

    lines = [
        "# Model Evaluation Summary",
        "",
        "## Project",
        "",
        "Financial Portfolio Forecasting & Anomaly Detection",
        "",
        "## Evaluation Run Information",
        "",
        f"- Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Dataset Path: `{DATA_PATH}`",
        f"- Overall Status: `{summary.get('status')}`",
        "",
        "## Dataset Summary",
        "",
        f"- Dataset Available: {dataset_summary.get('available')}",
        f"- Total Records: {dataset_summary.get('total_records')}",
        f"- Total Assets: {dataset_summary.get('total_assets')}",
        f"- Assets: {', '.join(dataset_summary.get('assets', [])) if dataset_summary.get('assets') else 'N/A'}",
        f"- Start Date: {dataset_summary.get('start_date')}",
        f"- End Date: {dataset_summary.get('end_date')}",
        "",
        "## Forecasting Evaluation",
        "",
        "| Model | Status | Records | MAE | RMSE | MAPE | SMAPE | R2 Score |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    if forecast_results:
        for model_name, result in forecast_results.items():
            metrics = result.get("overall_metrics", {})
            lines.append(
                "| {model} | {status} | {records} | {mae} | {rmse} | {mape} | {smape} | {r2} |".format(
                    model=model_name,
                    status=result.get("status"),
                    records=result.get("record_count", 0),
                    mae=metrics.get("mae"),
                    rmse=metrics.get("rmse"),
                    mape=metrics.get("mape"),
                    smape=metrics.get("smape"),
                    r2=metrics.get("r2_score"),
                )
            )
    else:
        lines.append("| No forecast evaluation available | - | - | - | - | - | - | - |")

    lines.extend(
        [
            "",
            "## Best Forecasting Model",
            "",
            f"- Best Model: {best_model.get('best_model')}",
            f"- Selection Metric: {best_model.get('selection_metric')}",
            f"- Metric Value: {best_model.get('metric_value')}",
            f"- Reason: {best_model.get('reason')}",
            "",
            "## Anomaly Evaluation",
            "",
            f"- Status: {anomaly_results.get('status')}",
            f"- Total Records: {anomaly_results.get('summary', {}).get('total_records')}",
            f"- Total Anomalies: {anomaly_results.get('summary', {}).get('total_anomalies')}",
            f"- Anomaly Rate: {anomaly_results.get('summary', {}).get('anomaly_rate_percent')}%",
            "",
            "## Anomaly Method Summary",
            "",
        ]
    )

    method_summary = anomaly_results.get("method_summary", {})

    if method_summary:
        for method_name, count in method_summary.items():
            lines.append(f"- {method_name}: {count}")
    else:
        lines.append("- No anomaly method summary available.")

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- Evaluation Summary JSON: `reports/tables/evaluation_summary.json`",
            "- Evaluation Summary Markdown: `reports/evaluation_summary.md`",
            "- Forecast Evaluation Combined CSV: `reports/tables/forecast_evaluation_combined.csv`",
            "- Forecast Metrics JSON/CSV: `reports/tables/*_forecast_evaluation.*`",
            "- Anomaly Metrics JSON: `reports/tables/anomaly_metrics.json`",
            "- Anomaly Metrics CSV: `reports/tables/anomaly_metrics.csv`",
            "- Model Comparison Report: `reports/model_comparison_report.md`",
            "- Evaluation Figures: `reports/figures/`",
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
            "- Lower MAE and RMSE indicate better forecasting accuracy.",
            "- MAPE and SMAPE show percentage-based forecasting error.",
            "- R2 score shows how well the model explains price variation.",
            "- Anomaly rate shows how many records were marked as unusual.",
            "- Model comparison selects the best forecasting model using RMSE.",
        ]
    )

    EVALUATION_SUMMARY_MD_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    logger.info("Evaluation Markdown summary saved at: %s", EVALUATION_SUMMARY_MD_PATH)


# ============================================================
# Evaluation Runner
# ============================================================

class ModelEvaluationRunner:
    """
    Run full model evaluation workflow.
    """

    def __init__(
        self,
        generate_visuals: bool = True,
        evaluate_forecasts: bool = True,
        evaluate_anomalies: bool = True,
        compare_models: bool = True,
    ) -> None:
        """
        Initialize ModelEvaluationRunner.

        Args:
            generate_visuals (bool): Whether to generate visualizations.
            evaluate_forecasts (bool): Whether to evaluate forecasting predictions.
            evaluate_anomalies (bool): Whether to evaluate anomaly results.
            compare_models (bool): Whether to compare trained models.
        """
        self.generate_visuals = generate_visuals
        self.evaluate_forecasts = evaluate_forecasts
        self.evaluate_anomalies = evaluate_anomalies
        self.compare_models = compare_models

    def run_basic_visualizations(
        self,
        raw_dataframe: Optional[pd.DataFrame],
    ) -> Dict[str, str]:
        """
        Generate basic dataset visualizations.

        Args:
            raw_dataframe (Optional[pd.DataFrame]): Raw dataframe.

        Returns:
            Dict[str, str]: Saved figure paths.
        """
        if not self.generate_visuals:
            return {}

        if raw_dataframe is None or raw_dataframe.empty:
            return {}

        try:
            clean_data = clean_raw_data(raw_dataframe)

            return generate_basic_visualizations(
                dataframe=clean_data,
                show=False,
            )

        except Exception as error:
            logger.warning("Could not generate basic visualizations: %s", error)
            return {}

    def run_forecast_evaluation(self) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate available forecast prediction files.

        Returns:
            Dict[str, Dict[str, Any]]: Forecast evaluation results.
        """
        if not self.evaluate_forecasts:
            return {}

        prediction_files = {
            "XGBoost": XGBOOST_PREDICTIONS_PATH,
            "Prophet": PROPHET_PREDICTIONS_PATH,
            "LSTM": LSTM_PREDICTIONS_PATH,
        }

        forecast_results = {}

        for model_name, file_path in prediction_files.items():
            log_step_start(logger, f"Evaluate {model_name} predictions")

            try:
                with track_step(
                    f"evaluate_{model_name.lower()}_predictions",
                    category="model_evaluation",
                ):
                    result = evaluate_prediction_file(
                        file_path=file_path,
                        model_name=model_name,
                    )

                forecast_results[model_name] = result

                log_step_end(logger, f"Evaluate {model_name} predictions")

            except Exception as error:
                log_step_error(logger, f"Evaluate {model_name} predictions", error)

                forecast_results[model_name] = {
                    "status": "failed",
                    "model_name": model_name,
                    "file_path": str(file_path),
                    "error_message": str(error),
                }

        build_forecast_evaluation_table(forecast_results)

        return forecast_results

    def run_anomaly_evaluation(self) -> Dict[str, Any]:
        """
        Evaluate anomaly detection results.

        Returns:
            Dict[str, Any]: Anomaly evaluation result.
        """
        if not self.evaluate_anomalies:
            return {}

        log_step_start(logger, "Evaluate anomaly detection results")

        try:
            with track_step(
                "evaluate_anomaly_results",
                category="model_evaluation",
            ):
                result = evaluate_anomaly_results()

            log_step_end(logger, "Evaluate anomaly detection results")

            return result

        except Exception as error:
            log_step_error(logger, "Evaluate anomaly detection results", error)

            return {
                "status": "failed",
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            }

    def run_model_comparison_workflow(self) -> Dict[str, Any]:
        """
        Run model comparison.

        Returns:
            Dict[str, Any]: Model comparison summary.
        """
        if not self.compare_models:
            return {}

        log_step_start(logger, "Run model comparison")

        try:
            with track_step("run_model_comparison", category="model_evaluation"):
                result = run_model_comparison(
                    selection_metric="rmse",
                    lower_is_better=True,
                    save_outputs=True,
                )

            if self.generate_visuals:
                figure_path = generate_model_comparison_visuals(result)
                result["comparison_figure_path"] = figure_path

            log_step_end(logger, "Run model comparison")

            return result

        except Exception as error:
            log_step_error(logger, "Run model comparison", error)

            return {
                "status": "failed",
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            }

    def run(self) -> Dict[str, Any]:
        """
        Run full evaluation workflow.

        Returns:
            Dict[str, Any]: Evaluation summary.
        """
        log_section(logger, "Starting Model Evaluation Workflow")

        try:
            ensure_evaluation_directories()
            create_required_directories()

            with track_step("load_raw_dataset_for_evaluation", category="model_evaluation"):
                raw_dataframe = load_csv_if_exists(DATA_PATH)

            dataset_summary = get_dataset_summary(raw_dataframe)

            basic_figure_paths = self.run_basic_visualizations(raw_dataframe)

            forecast_results = self.run_forecast_evaluation()

            anomaly_results = self.run_anomaly_evaluation()

            model_comparison_summary = self.run_model_comparison_workflow()

            save_performance_reports()
            performance_summary = get_performance_summary()

            status = "success"

            if (
                not forecast_results
                and not anomaly_results
                and not model_comparison_summary
            ):
                status = "partial_success"

            summary = {
                "status": status,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataset_summary": dataset_summary,
                "forecast_evaluation": forecast_results,
                "anomaly_evaluation": anomaly_results,
                "model_comparison": model_comparison_summary,
                "basic_visualizations": basic_figure_paths,
                "performance_summary": performance_summary,
                "output_paths": {
                    "evaluation_summary_json": str(EVALUATION_SUMMARY_JSON_PATH),
                    "evaluation_summary_markdown": str(EVALUATION_SUMMARY_MD_PATH),
                    "forecast_evaluation_combined": str(FORECAST_EVALUATION_COMBINED_PATH),
                    "figures_dir": str(FIGURES_DIR),
                },
            }

            save_json(summary, EVALUATION_SUMMARY_JSON_PATH)
            save_markdown_summary(summary)

            log_section(logger, "Model Evaluation Workflow Completed")

            return summary

        except Exception as error:
            log_step_error(logger, "Model evaluation workflow", error)

            save_performance_reports()

            summary = {
                "status": "failed",
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error_type": error.__class__.__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            }

            save_json(summary, EVALUATION_SUMMARY_JSON_PATH)

            return summary


# ============================================================
# Main Function
# ============================================================

def main() -> None:
    """
    Main function for model evaluation.
    """
    runner = ModelEvaluationRunner(
        generate_visuals=True,
        evaluate_forecasts=True,
        evaluate_anomalies=True,
        compare_models=True,
    )

    summary = runner.run()

    print("\n" + "=" * 80)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 80)

    print("Status:", summary.get("status"))

    if summary.get("status") in ["success", "partial_success"]:
        dataset_summary = summary.get("dataset_summary", {})
        model_comparison = summary.get("model_comparison", {})
        anomaly_evaluation = summary.get("anomaly_evaluation", {})

        print("\nDataset Summary:")
        print("- Dataset Available:", dataset_summary.get("available"))
        print("- Total Records:", dataset_summary.get("total_records"))
        print("- Total Assets:", dataset_summary.get("total_assets"))
        print("- Start Date:", dataset_summary.get("start_date"))
        print("- End Date:", dataset_summary.get("end_date"))

        print("\nBest Model:")
        best_model = model_comparison.get("best_model", {})
        print("- Model:", best_model.get("best_model"))
        print("- Metric:", best_model.get("selection_metric"))
        print("- Value:", best_model.get("metric_value"))

        print("\nAnomaly Evaluation:")
        anomaly_summary = anomaly_evaluation.get("summary", {})
        print("- Status:", anomaly_evaluation.get("status"))
        print("- Total Records:", anomaly_summary.get("total_records"))
        print("- Total Anomalies:", anomaly_summary.get("total_anomalies"))
        print("- Anomaly Rate %:", anomaly_summary.get("anomaly_rate_percent"))

        print("\nSaved Files:")
        for name, path in summary.get("output_paths", {}).items():
            print(f"- {name}: {path}")

    else:
        print("\nEvaluation failed.")
        print("Error Type:", summary.get("error_type"))
        print("Error Message:", summary.get("error_message"))
        print("\nCheck logs/app.log or logs/training.log for details.")

    print("=" * 80)


# ============================================================
# Script Entry Point
# ============================================================

if __name__ == "__main__":
    main()