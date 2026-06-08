"""
Run all models for the Financial Portfolio Forecasting
and Anomaly Detection project.

This script runs the complete model workflow:

1. Train forecasting models
   - ARIMA
   - XGBoost
   - Prophet
   - LSTM

2. Train anomaly detection model
   - IQR
   - Z-score
   - Isolation Forest

3. Save final project summary

Dataset expected columns:
- date
- asset
- price

Raw dataset expected path:
data/raw/financial_portfolio_data.csv

Run from project root:

    python scripts/run_all_models.py
"""

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


# ============================================================
# Add Project Root to Python Path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ============================================================
# Project Imports
# ============================================================

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
from src.training.train_anomaly_model import AnomalyTrainingRunner
from src.training.train_forecasting_models import ForecastingModelTrainer
from src.utils.paths import create_required_directories


# ============================================================
# Paths
# ============================================================

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "financial_portfolio_data.csv"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = REPORTS_DIR / "figures"
LOGS_DIR = PROJECT_ROOT / "logs"

FINAL_SUMMARY_JSON_PATH = TABLES_DIR / "all_models_training_summary.json"
FINAL_SUMMARY_MD_PATH = REPORTS_DIR / "all_models_training_summary.md"


# ============================================================
# Logger
# ============================================================

logger = get_project_logger()


# ============================================================
# Helper Functions
# ============================================================

def ensure_all_directories() -> None:
    """
    Create all required project directories.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def check_dataset_exists() -> None:
    """
    Check whether the raw dataset exists.

    Raises:
        FileNotFoundError: If dataset is missing.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}\n\n"
            "Please make sure your CSV file is placed here:\n"
            "data/raw/financial_portfolio_data.csv\n\n"
            "Your CSV should contain these columns:\n"
            "- Date\n"
            "- Asset\n"
            "- Price"
        )


def save_json_summary(summary: Dict[str, Any]) -> None:
    """
    Save final all-model training summary as JSON.

    Args:
        summary (Dict[str, Any]): Final summary dictionary.
    """
    with FINAL_SUMMARY_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, default=str)

    logger.info("Final all-model summary JSON saved at: %s", FINAL_SUMMARY_JSON_PATH)


def save_markdown_summary(summary: Dict[str, Any]) -> None:
    """
    Save final all-model training summary as Markdown.

    Args:
        summary (Dict[str, Any]): Final summary dictionary.
    """
    forecasting_summary = summary.get("forecasting_summary", {})
    anomaly_summary = summary.get("anomaly_summary", {})
    performance_summary = summary.get("performance_summary", {})

    forecasting_status = forecasting_summary.get("status")
    anomaly_status = anomaly_summary.get("status")

    best_model = forecasting_summary.get("best_model", {})

    anomaly_details = anomaly_summary.get("anomaly_summary", {})

    lines = [
        "# All Models Training Summary",
        "",
        "## Project",
        "",
        "Financial Portfolio Forecasting & Anomaly Detection",
        "",
        "## Run Information",
        "",
        f"- Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Dataset Path: `{DATA_PATH}`",
        f"- Overall Status: `{summary.get('status')}`",
        "",
        "## Workflow Status",
        "",
        f"- Forecasting Models Status: `{forecasting_status}`",
        f"- Anomaly Detection Status: `{anomaly_status}`",
        "",
        "## Forecasting Models",
        "",
        "The following forecasting models were trained:",
        "",
        "- ARIMA",
        "- XGBoost",
        "- Prophet",
        "- LSTM",
        "",
        "### Best Forecasting Model",
        "",
        f"- Best Model: {best_model.get('best_model')}",
        f"- Selection Metric: {best_model.get('best_metric')}",
        f"- Best Metric Value: {best_model.get('best_metric_value')}",
        "",
        "### Forecasting Output Files",
        "",
        "- Model Comparison CSV: `reports/tables/forecasting_model_comparison.csv`",
        "- Model Comparison JSON: `reports/tables/forecasting_model_comparison.json`",
        "- Forecasting Training Summary: `reports/forecasting_training_summary.md`",
        "- ARIMA Forecasts: `outputs/arima_forecast_results.csv`",
        "- XGBoost Forecasts: `outputs/xgboost_forecast_results.csv`",
        "- Prophet Forecasts: `outputs/prophet_forecast_results.csv`",
        "- LSTM Forecasts: `outputs/lstm_forecast_results.csv`",
        "",
        "## Anomaly Detection",
        "",
        "The following anomaly detection methods were used:",
        "",
        "- IQR",
        "- Z-score",
        "- Isolation Forest",
        "",
        "### Anomaly Detection Summary",
        "",
        f"- Total Records Checked: {anomaly_details.get('total_records')}",
        f"- Total Anomalies Detected: {anomaly_details.get('total_anomalies')}",
        f"- Anomaly Rate: {anomaly_details.get('anomaly_rate_percent')}%",
        f"- Asset Count: {anomaly_details.get('asset_count')}",
        "",
        "### Anomaly Output Files",
        "",
        "- Anomaly Model: `models/anomaly_detection_model.pkl`",
        "- Anomaly Results: `outputs/anomaly_detection_results.csv`",
        "- Anomaly Summary JSON: `reports/tables/anomaly_summary.json`",
        "- Anomaly Summary CSV: `reports/tables/anomaly_summary.csv`",
        "- Anomaly Report: `reports/anomaly_report.md`",
        "- Anomaly Training Summary: `reports/anomaly_training_summary.md`",
        "",
        "## Performance Summary",
        "",
        f"- Total Records Tracked: {performance_summary.get('total_records')}",
        f"- Total Execution Time: {performance_summary.get('total_execution_time_seconds')} seconds",
        f"- Successful Steps: {performance_summary.get('successful_steps')}",
        f"- Failed Steps: {performance_summary.get('failed_steps')}",
        "",
        "## Final Output Files",
        "",
        f"- Final Summary JSON: `{FINAL_SUMMARY_JSON_PATH}`",
        f"- Final Summary Markdown: `{FINAL_SUMMARY_MD_PATH}`",
        "- Performance Metrics JSON: `reports/tables/performance_metrics.json`",
        "- Performance Metrics CSV: `reports/tables/performance_metrics.csv`",
        "",
        "## Interpretation",
        "",
        "- Forecasting models predict future asset prices.",
        "- The best forecasting model is selected using lowest RMSE.",
        "- Anomaly detection identifies unusual asset price movements.",
        "- The final outputs can be used in dashboards, reports, and portfolio documentation.",
        "",
        "## Next Steps",
        "",
        "- Review model comparison metrics.",
        "- Review anomaly report.",
        "- Use the best forecasting model in the dashboard or API.",
        "- Add charts and screenshots to the final project documentation.",
    ]

    FINAL_SUMMARY_MD_PATH.write_text("\n".join(lines), encoding="utf-8")

    logger.info(
        "Final all-model summary Markdown saved at: %s",
        FINAL_SUMMARY_MD_PATH,
    )


# ============================================================
# All Models Runner
# ============================================================

class AllModelsRunner:
    """
    Run all forecasting and anomaly detection models.
    """

    def __init__(
        self,
        forecast_steps: int = 30,
        run_arima: bool = True,
        run_xgboost: bool = True,
        run_prophet: bool = True,
        run_lstm: bool = True,
        lstm_epochs: int = 10,
        iqr_multiplier: float = 1.5,
        z_score_threshold: float = 3.0,
        contamination: float = 0.05,
    ) -> None:
        """
        Initialize AllModelsRunner.

        Args:
            forecast_steps (int): Forecast horizon.
            run_arima (bool): Whether to train ARIMA.
            run_xgboost (bool): Whether to train XGBoost.
            run_prophet (bool): Whether to train Prophet.
            run_lstm (bool): Whether to train LSTM.
            lstm_epochs (int): Number of LSTM epochs.
            iqr_multiplier (float): IQR anomaly multiplier.
            z_score_threshold (float): Z-score threshold.
            contamination (float): Isolation Forest contamination.
        """
        self.forecast_steps = forecast_steps
        self.run_arima = run_arima
        self.run_xgboost = run_xgboost
        self.run_prophet = run_prophet
        self.run_lstm = run_lstm
        self.lstm_epochs = lstm_epochs
        self.iqr_multiplier = iqr_multiplier
        self.z_score_threshold = z_score_threshold
        self.contamination = contamination

    def run_forecasting_models(self) -> Dict[str, Any]:
        """
        Train all selected forecasting models.

        Returns:
            Dict[str, Any]: Forecasting training summary.
        """
        log_step_start(logger, "Run forecasting models")

        forecasting_trainer = ForecastingModelTrainer(
            forecast_steps=self.forecast_steps,
            run_arima=self.run_arima,
            run_xgboost=self.run_xgboost,
            run_prophet=self.run_prophet,
            run_lstm=self.run_lstm,
            lstm_epochs=self.lstm_epochs,
        )

        with track_step("run_forecasting_models", category="all_models"):
            forecasting_summary = forecasting_trainer.run()

        log_step_end(logger, "Run forecasting models")

        return forecasting_summary

    def run_anomaly_model(self) -> Dict[str, Any]:
        """
        Train anomaly detection model.

        Returns:
            Dict[str, Any]: Anomaly training summary.
        """
        log_step_start(logger, "Run anomaly detection model")

        anomaly_runner = AnomalyTrainingRunner(
            iqr_multiplier=self.iqr_multiplier,
            z_score_threshold=self.z_score_threshold,
            contamination=self.contamination,
            random_state=42,
        )

        with track_step("run_anomaly_model", category="all_models"):
            anomaly_summary = anomaly_runner.run()

        log_step_end(logger, "Run anomaly detection model")

        return anomaly_summary

    def run(self) -> Dict[str, Any]:
        """
        Run complete all-model workflow.

        Returns:
            Dict[str, Any]: Final workflow summary.
        """
        log_section(logger, "Starting All Models Workflow")

        try:
            ensure_all_directories()
            create_required_directories()
            check_dataset_exists()

            forecasting_summary = self.run_forecasting_models()
            anomaly_summary = self.run_anomaly_model()

            save_performance_reports()
            performance_summary = get_performance_summary()

            overall_status = (
                "success"
                if forecasting_summary.get("status") == "success"
                and anomaly_summary.get("status") == "success"
                else "partial_success"
            )

            final_summary = {
                "status": overall_status,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataset_path": str(DATA_PATH),
                "configuration": {
                    "forecast_steps": self.forecast_steps,
                    "run_arima": self.run_arima,
                    "run_xgboost": self.run_xgboost,
                    "run_prophet": self.run_prophet,
                    "run_lstm": self.run_lstm,
                    "lstm_epochs": self.lstm_epochs,
                    "iqr_multiplier": self.iqr_multiplier,
                    "z_score_threshold": self.z_score_threshold,
                    "contamination": self.contamination,
                },
                "forecasting_summary": forecasting_summary,
                "anomaly_summary": anomaly_summary,
                "performance_summary": performance_summary,
                "output_paths": {
                    "final_summary_json": str(FINAL_SUMMARY_JSON_PATH),
                    "final_summary_markdown": str(FINAL_SUMMARY_MD_PATH),
                    "forecasting_comparison": "reports/tables/forecasting_model_comparison.csv",
                    "forecasting_summary": "reports/forecasting_training_summary.md",
                    "anomaly_results": "outputs/anomaly_detection_results.csv",
                    "anomaly_report": "reports/anomaly_report.md",
                    "performance_json": "reports/tables/performance_metrics.json",
                    "performance_csv": "reports/tables/performance_metrics.csv",
                },
            }

            save_json_summary(final_summary)
            save_markdown_summary(final_summary)

            log_section(logger, "All Models Workflow Completed")

            return final_summary

        except Exception as error:
            log_step_error(logger, "All models workflow", error)

            save_performance_reports()

            final_summary = {
                "status": "failed",
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataset_path": str(DATA_PATH),
                "error_type": error.__class__.__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            }

            save_json_summary(final_summary)

            return final_summary


# ============================================================
# Main Function
# ============================================================

def main() -> None:
    """
    Main entry point for running all models.
    """
    runner = AllModelsRunner(
        forecast_steps=30,
        run_arima=True,
        run_xgboost=True,
        run_prophet=True,
        run_lstm=True,
        lstm_epochs=10,
        iqr_multiplier=1.5,
        z_score_threshold=3.0,
        contamination=0.05,
    )

    summary = runner.run()

    print("\n" + "=" * 80)
    print("ALL MODELS TRAINING SUMMARY")
    print("=" * 80)

    print("Status:", summary.get("status"))

    if summary.get("status") in ["success", "partial_success"]:
        print("\nConfiguration:")
        for key, value in summary.get("configuration", {}).items():
            print(f"- {key}: {value}")

        forecasting_summary = summary.get("forecasting_summary", {})
        anomaly_summary = summary.get("anomaly_summary", {})

        print("\nForecasting Status:", forecasting_summary.get("status"))

        best_model = forecasting_summary.get("best_model", {})
        if best_model:
            print("Best Forecasting Model:", best_model.get("best_model"))
            print("Best Metric:", best_model.get("best_metric"))
            print("Best Metric Value:", best_model.get("best_metric_value"))

        print("\nAnomaly Status:", anomaly_summary.get("status"))

        anomaly_details = anomaly_summary.get("anomaly_summary", {})
        if anomaly_details:
            print("Total Anomalies:", anomaly_details.get("total_anomalies"))
            print("Anomaly Rate %:", anomaly_details.get("anomaly_rate_percent"))

        print("\nSaved Files:")
        for name, path in summary.get("output_paths", {}).items():
            print(f"- {name}: {path}")

    else:
        print("\nWorkflow failed.")
        print("Error Type:", summary.get("error_type"))
        print("Error Message:", summary.get("error_message"))
        print("\nCheck logs/app.log or logs/training.log for details.")

    print("=" * 80)


# ============================================================
# Script Entry Point
# ============================================================

if __name__ == "__main__":
    main()