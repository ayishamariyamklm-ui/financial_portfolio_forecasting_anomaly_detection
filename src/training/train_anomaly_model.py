"""
Train anomaly detection model for the Financial Portfolio Forecasting
and Anomaly Detection project.

This script trains and saves the anomaly detection workflow using:

1. IQR anomaly detection
2. Z-score anomaly detection
3. Isolation Forest anomaly detection

Dataset expected columns:
- date
- asset
- price

Raw dataset expected path:
data/raw/financial_portfolio_data.csv

Run from project root:

    python src/training/train_anomaly_model.py
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ============================================================
# Project Imports
# ============================================================

from src.models.anomaly_detector import FinancialAnomalyDetector
from src.monitoring.logger import (
    get_training_logger,
    log_dataframe_info,
    log_section,
    log_step_error,
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

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = REPORTS_DIR / "figures"
LOGS_DIR = PROJECT_ROOT / "logs"

ANOMALY_TRAINING_SUMMARY_JSON_PATH = TABLES_DIR / "anomaly_training_summary.json"
ANOMALY_TRAINING_SUMMARY_MD_PATH = REPORTS_DIR / "anomaly_training_summary.md"
ANOMALY_ASSET_SUMMARY_PATH = TABLES_DIR / "anomaly_asset_summary.csv"
ANOMALY_TOP_RECORDS_PATH = TABLES_DIR / "top_anomaly_records.csv"


# ============================================================
# Logger
# ============================================================

logger = get_training_logger()


# ============================================================
# Helper Functions
# ============================================================

def ensure_training_directories() -> None:
    """
    Create all folders required for anomaly model training outputs.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def load_financial_data(file_path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load and clean financial portfolio dataset.

    Args:
        file_path (Path): CSV file path.

    Returns:
        pd.DataFrame: Cleaned dataframe.

    Raises:
        FileNotFoundError: If dataset is missing.
        ValueError: If required columns are missing or data is empty.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {file_path}\n"
            "Please place your CSV file at data/raw/financial_portfolio_data.csv"
        )

    dataframe = pd.read_csv(file_path)

    dataframe.columns = (
        dataframe.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    required_columns = ["date", "asset", "price"]

    missing_columns = [
        column for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            "Expected columns are: date, asset, price."
        )

    dataframe["date"] = pd.to_datetime(dataframe["date"], errors="coerce")
    dataframe["price"] = pd.to_numeric(dataframe["price"], errors="coerce")

    dataframe = dataframe.dropna(subset=["date", "asset", "price"])
    dataframe = dataframe[dataframe["price"] >= 0]
    dataframe = dataframe.sort_values(["asset", "date"]).reset_index(drop=True)

    if dataframe.empty:
        raise ValueError("Dataset is empty after cleaning.")

    logger.info("Dataset loaded successfully with shape: %s", dataframe.shape)

    return dataframe


def get_dataset_summary(dataframe: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate dataset summary.

    Args:
        dataframe (pd.DataFrame): Input dataframe.

    Returns:
        Dict[str, Any]: Dataset summary dictionary.
    """
    asset_summary = (
        dataframe.groupby("asset")
        .agg(
            record_count=("price", "count"),
            start_date=("date", "min"),
            end_date=("date", "max"),
            average_price=("price", "mean"),
            min_price=("price", "min"),
            max_price=("price", "max"),
            price_std=("price", "std"),
        )
        .reset_index()
    )

    return {
        "total_records": int(len(dataframe)),
        "total_assets": int(dataframe["asset"].nunique()),
        "assets": sorted(dataframe["asset"].unique().tolist()),
        "start_date": str(dataframe["date"].min()),
        "end_date": str(dataframe["date"].max()),
        "average_price": round(float(dataframe["price"].mean()), 4),
        "minimum_price": round(float(dataframe["price"].min()), 4),
        "maximum_price": round(float(dataframe["price"].max()), 4),
        "asset_summary": asset_summary.to_dict(orient="records"),
    }


def save_training_json_summary(summary: Dict[str, Any]) -> None:
    """
    Save anomaly training summary as JSON.

    Args:
        summary (Dict[str, Any]): Training summary.
    """
    with ANOMALY_TRAINING_SUMMARY_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, default=str)

    logger.info(
        "Anomaly training summary JSON saved at: %s",
        ANOMALY_TRAINING_SUMMARY_JSON_PATH,
    )


def save_asset_summary(detector_summary: Dict[str, Any]) -> pd.DataFrame:
    """
    Save asset-level anomaly summary as CSV.

    Args:
        detector_summary (Dict[str, Any]): Detector summary.

    Returns:
        pd.DataFrame: Asset summary dataframe.
    """
    asset_summary = detector_summary.get("asset_summary", [])

    asset_summary_dataframe = pd.DataFrame(asset_summary)

    if not asset_summary_dataframe.empty:
        asset_summary_dataframe.to_csv(ANOMALY_ASSET_SUMMARY_PATH, index=False)

        logger.info(
            "Anomaly asset summary saved at: %s",
            ANOMALY_ASSET_SUMMARY_PATH,
        )

    return asset_summary_dataframe


def save_top_anomaly_records(detector_summary: Dict[str, Any]) -> pd.DataFrame:
    """
    Save top anomaly records as CSV.

    Args:
        detector_summary (Dict[str, Any]): Detector summary.

    Returns:
        pd.DataFrame: Top anomalies dataframe.
    """
    top_anomalies = detector_summary.get("top_anomalies", [])

    top_anomalies_dataframe = pd.DataFrame(top_anomalies)

    if not top_anomalies_dataframe.empty:
        top_anomalies_dataframe.to_csv(ANOMALY_TOP_RECORDS_PATH, index=False)

        logger.info(
            "Top anomaly records saved at: %s",
            ANOMALY_TOP_RECORDS_PATH,
        )

    return top_anomalies_dataframe


def save_markdown_training_summary(
    summary: Dict[str, Any],
    asset_summary_dataframe: pd.DataFrame,
    top_anomalies_dataframe: pd.DataFrame,
) -> None:
    """
    Save anomaly model training summary as Markdown.

    Args:
        summary (Dict[str, Any]): Training summary.
        asset_summary_dataframe (pd.DataFrame): Asset summary dataframe.
        top_anomalies_dataframe (pd.DataFrame): Top anomalies dataframe.
    """
    dataset_summary = summary.get("dataset_summary", {})
    anomaly_summary = summary.get("anomaly_summary", {})
    model_config = summary.get("model_config", {})
    output_paths = summary.get("output_paths", {})
    performance_summary = summary.get("performance_summary", {})

    method_summary = anomaly_summary.get("method_summary", {})
    severity_summary = anomaly_summary.get("severity_summary", {})

    lines = [
        "# Anomaly Model Training Summary",
        "",
        "## Project",
        "",
        "Financial Portfolio Forecasting & Anomaly Detection",
        "",
        "## Training Run Information",
        "",
        f"- Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Dataset Path: `{DATA_PATH}`",
        "",
        "## Dataset Summary",
        "",
        f"- Total Records: {dataset_summary.get('total_records')}",
        f"- Total Assets: {dataset_summary.get('total_assets')}",
        f"- Assets: {', '.join(dataset_summary.get('assets', []))}",
        f"- Start Date: {dataset_summary.get('start_date')}",
        f"- End Date: {dataset_summary.get('end_date')}",
        f"- Average Price: {dataset_summary.get('average_price')}",
        f"- Minimum Price: {dataset_summary.get('minimum_price')}",
        f"- Maximum Price: {dataset_summary.get('maximum_price')}",
        "",
        "## Anomaly Model Configuration",
        "",
        f"- Methods Used: {', '.join(model_config.get('methods', []))}",
        f"- IQR Multiplier: {model_config.get('iqr_multiplier')}",
        f"- Z-Score Threshold: {model_config.get('z_score_threshold')}",
        f"- Isolation Forest Contamination: {model_config.get('contamination')}",
        f"- Random State: {model_config.get('random_state')}",
        "",
        "## Overall Anomaly Summary",
        "",
        f"- Total Records Checked: {anomaly_summary.get('total_records')}",
        f"- Total Anomalies Detected: {anomaly_summary.get('total_anomalies')}",
        f"- Anomaly Rate: {anomaly_summary.get('anomaly_rate_percent')}%",
        f"- Asset Count: {anomaly_summary.get('asset_count')}",
        "",
        "## Method-wise Anomaly Count",
        "",
        f"- IQR Anomalies: {method_summary.get('iqr_anomalies', 0)}",
        f"- Z-Score Anomalies: {method_summary.get('zscore_anomalies', 0)}",
        f"- Isolation Forest Anomalies: {method_summary.get('isolation_forest_anomalies', 0)}",
        "",
        "## Severity Summary",
        "",
    ]

    if severity_summary:
        for severity, count in severity_summary.items():
            lines.append(f"- {severity}: {count}")
    else:
        lines.append("- No severity summary available.")

    lines.extend(
        [
            "",
            "## Asset-Level Anomaly Summary",
            "",
            "| Asset | Records | Anomalies | Anomaly Rate % | IQR | Z-Score | Isolation Forest | Avg Price |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    if not asset_summary_dataframe.empty:
        for _, row in asset_summary_dataframe.iterrows():
            lines.append(
                "| {asset} | {records} | {anomalies} | {rate:.2f} | {iqr} | {zscore} | {iso} | {avg:.2f} |".format(
                    asset=row.get("asset"),
                    records=int(row.get("total_records", 0)),
                    anomalies=int(row.get("anomaly_count", 0)),
                    rate=float(row.get("anomaly_rate_percent", 0)),
                    iqr=int(row.get("iqr_anomaly_count", 0)),
                    zscore=int(row.get("zscore_anomaly_count", 0)),
                    iso=int(row.get("isolation_forest_anomaly_count", 0)),
                    avg=float(row.get("average_price", 0)),
                )
            )
    else:
        lines.append("| No asset summary available | - | - | - | - | - | - | - |")

    lines.extend(
        [
            "",
            "## Top Anomaly Records",
            "",
            "| Date | Asset | Price | Method | Severity | Anomaly Score |",
            "|---|---|---:|---|---|---:|",
        ]
    )

    if not top_anomalies_dataframe.empty:
        for _, row in top_anomalies_dataframe.iterrows():
            lines.append(
                "| {date} | {asset} | {price:.2f} | {method} | {severity} | {score:.6f} |".format(
                    date=row.get("date"),
                    asset=row.get("asset"),
                    price=float(row.get("price", 0)),
                    method=row.get("anomaly_method"),
                    severity=row.get("anomaly_severity"),
                    score=float(row.get("anomaly_score", 0)),
                )
            )
    else:
        lines.append("| No top anomalies available | - | - | - | - | - |")

    lines.extend(
        [
            "",
            "## Output Files",
            "",
        ]
    )

    for name, path in output_paths.items():
        lines.append(f"- {name}: `{path}`")

    lines.extend(
        [
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
            "- IQR detects price values outside the normal interquartile range.",
            "- Z-score detects prices far away from the asset's average price.",
            "- Isolation Forest detects unusual patterns using price, return, lag, rolling, volatility, and calendar features.",
            "- If the same record is detected by multiple methods, it is considered more severe.",
            "",
            "## Business Use",
            "",
            "This anomaly model can help detect unusual asset price movements, possible data quality issues, sudden market changes, and risky price behavior.",
        ]
    )

    ANOMALY_TRAINING_SUMMARY_MD_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    logger.info(
        "Anomaly training summary Markdown saved at: %s",
        ANOMALY_TRAINING_SUMMARY_MD_PATH,
    )


# ============================================================
# Anomaly Training Runner
# ============================================================

class AnomalyTrainingRunner:
    """
    Train and save anomaly detection model.
    """

    def __init__(
        self,
        iqr_multiplier: float = 1.5,
        z_score_threshold: float = 3.0,
        contamination: float = 0.05,
        random_state: int = 42,
    ) -> None:
        """
        Initialize AnomalyTrainingRunner.

        Args:
            iqr_multiplier (float): IQR multiplier.
            z_score_threshold (float): Z-score threshold.
            contamination (float): Isolation Forest contamination.
            random_state (int): Random seed.
        """
        self.iqr_multiplier = iqr_multiplier
        self.z_score_threshold = z_score_threshold
        self.contamination = contamination
        self.random_state = random_state

        self.detector = FinancialAnomalyDetector(
            date_column="date",
            asset_column="asset",
            target_column="price",
            iqr_multiplier=self.iqr_multiplier,
            z_score_threshold=self.z_score_threshold,
            contamination=self.contamination,
            random_state=self.random_state,
        )

    def run(self) -> Dict[str, Any]:
        """
        Run complete anomaly model training workflow.

        Returns:
            Dict[str, Any]: Training summary.
        """
        log_section(logger, "Starting Anomaly Model Training")

        try:
            ensure_training_directories()
            create_required_directories()

            with track_step("load_anomaly_training_data", category="anomaly_training"):
                dataframe = load_financial_data(DATA_PATH)

            log_dataframe_info(logger, dataframe, "Anomaly Training Data")

            dataset_summary = get_dataset_summary(dataframe)

            logger.info("Dataset summary: %s", dataset_summary)

            with track_step("train_anomaly_detector", category="anomaly_training"):
                workflow_summary = self.detector.fit_detect_save(dataframe)

            asset_summary_dataframe = save_asset_summary(self.detector.summary)
            top_anomalies_dataframe = save_top_anomaly_records(self.detector.summary)

            save_performance_reports()
            performance_summary = get_performance_summary()

            training_summary = {
                "status": "success",
                "dataset_summary": dataset_summary,
                "model_config": {
                    "model_type": "Multi-method anomaly detection",
                    "methods": [
                        "IQR",
                        "Z-Score",
                        "Isolation Forest",
                    ],
                    "iqr_multiplier": self.iqr_multiplier,
                    "z_score_threshold": self.z_score_threshold,
                    "contamination": self.contamination,
                    "random_state": self.random_state,
                },
                "anomaly_summary": self.detector.summary,
                "workflow_summary": workflow_summary,
                "output_paths": {
                    "model": "models/anomaly_detection_model.pkl",
                    "results": "outputs/anomaly_detection_results.csv",
                    "summary_json": "reports/tables/anomaly_summary.json",
                    "summary_csv": "reports/tables/anomaly_summary.csv",
                    "report": "reports/anomaly_report.md",
                    "training_summary_json": str(ANOMALY_TRAINING_SUMMARY_JSON_PATH),
                    "training_summary_md": str(ANOMALY_TRAINING_SUMMARY_MD_PATH),
                    "asset_summary": str(ANOMALY_ASSET_SUMMARY_PATH),
                    "top_anomaly_records": str(ANOMALY_TOP_RECORDS_PATH),
                    "performance_json": "reports/tables/performance_metrics.json",
                    "performance_csv": "reports/tables/performance_metrics.csv",
                },
                "performance_summary": performance_summary,
            }

            save_training_json_summary(training_summary)
            save_markdown_training_summary(
                summary=training_summary,
                asset_summary_dataframe=asset_summary_dataframe,
                top_anomalies_dataframe=top_anomalies_dataframe,
            )

            log_section(logger, "Anomaly Model Training Completed Successfully")

            return training_summary

        except Exception as error:
            log_step_error(logger, "Anomaly model training", error)

            save_performance_reports()

            return {
                "status": "failed",
                "error_type": error.__class__.__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            }


# ============================================================
# Main Function
# ============================================================

def main() -> None:
    """
    Main function to train anomaly detection model.
    """
    runner = AnomalyTrainingRunner(
        iqr_multiplier=1.5,
        z_score_threshold=3.0,
        contamination=0.05,
        random_state=42,
    )

    summary = runner.run()

    print("\n" + "=" * 80)
    print("ANOMALY MODEL TRAINING SUMMARY")
    print("=" * 80)

    print("Status:", summary.get("status"))

    if summary.get("status") == "success":
        dataset_summary = summary.get("dataset_summary", {})
        model_config = summary.get("model_config", {})
        anomaly_summary = summary.get("anomaly_summary", {})
        output_paths = summary.get("output_paths", {})

        print("\nDataset Summary:")
        print("- Total Records:", dataset_summary.get("total_records"))
        print("- Total Assets:", dataset_summary.get("total_assets"))
        print("- Start Date:", dataset_summary.get("start_date"))
        print("- End Date:", dataset_summary.get("end_date"))

        print("\nModel Configuration:")
        print("- Model Type:", model_config.get("model_type"))
        print("- Methods:", model_config.get("methods"))
        print("- IQR Multiplier:", model_config.get("iqr_multiplier"))
        print("- Z-Score Threshold:", model_config.get("z_score_threshold"))
        print("- Contamination:", model_config.get("contamination"))

        print("\nAnomaly Summary:")
        print("- Total Records:", anomaly_summary.get("total_records"))
        print("- Total Anomalies:", anomaly_summary.get("total_anomalies"))
        print("- Anomaly Rate %:", anomaly_summary.get("anomaly_rate_percent"))
        print("- Asset Count:", anomaly_summary.get("asset_count"))

        print("\nSaved Files:")
        for name, path in output_paths.items():
            print(f"- {name}: {path}")

    else:
        print("\nAnomaly model training failed.")
        print("Error Type:", summary.get("error_type"))
        print("Error Message:", summary.get("error_message"))
        print("\nCheck logs/training.log or logs/app.log for details.")

    print("=" * 80)


# ============================================================
# Script Entry Point
# ============================================================

if __name__ == "__main__":
    main()