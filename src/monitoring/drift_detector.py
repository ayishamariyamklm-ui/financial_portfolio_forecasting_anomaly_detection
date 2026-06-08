"""
Data drift detection utilities for the Financial Portfolio Forecasting
and Anomaly Detection project.

This module provides:
- Numerical feature drift detection
- Categorical feature drift detection
- Target drift detection
- Portfolio value distribution comparison
- PSI calculation
- KS test based drift detection
- Drift report generation in JSON, CSV, and Markdown formats
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import chi2_contingency, ks_2samp
except ImportError:
    chi2_contingency = None
    ks_2samp = None

try:
    from src.monitoring.logger import get_project_logger
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)
    get_project_logger = lambda: logging.getLogger("drift_detector")


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
DRIFT_REPORT_JSON_PATH = TABLES_DIR / "drift_report.json"
DRIFT_REPORT_CSV_PATH = TABLES_DIR / "drift_report.csv"
DRIFT_REPORT_MD_PATH = REPORTS_DIR / "drift_report.md"


# ============================================================
# Logger
# ============================================================

logger = get_project_logger()


# ============================================================
# Helper Functions
# ============================================================

def ensure_drift_dirs() -> None:
    """
    Create folders required for drift reports.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def get_current_timestamp() -> str:
    """
    Get current timestamp.

    Returns:
        str: Current timestamp string.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_float(value: Any) -> Optional[float]:
    """
    Convert a value to float safely.

    Args:
        value (Any): Input value.

    Returns:
        Optional[float]: Float value or None.
    """
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def get_numeric_columns(dataframe: pd.DataFrame) -> List[str]:
    """
    Get numeric columns from a dataframe.

    Args:
        dataframe (pd.DataFrame): Input dataframe.

    Returns:
        List[str]: Numeric column names.
    """
    return dataframe.select_dtypes(include=["number"]).columns.tolist()


def get_categorical_columns(dataframe: pd.DataFrame) -> List[str]:
    """
    Get categorical columns from a dataframe.

    Args:
        dataframe (pd.DataFrame): Input dataframe.

    Returns:
        List[str]: Categorical column names.
    """
    return dataframe.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()


def validate_drift_inputs(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
) -> None:
    """
    Validate reference and current data.

    Args:
        reference_data (pd.DataFrame): Baseline dataframe.
        current_data (pd.DataFrame): New/current dataframe.

    Raises:
        ValueError: If input data is invalid.
    """
    if reference_data is None or current_data is None:
        raise ValueError("Reference data and current data cannot be None.")

    if reference_data.empty:
        raise ValueError("Reference data is empty.")

    if current_data.empty:
        raise ValueError("Current data is empty.")


# ============================================================
# PSI Calculation
# ============================================================

def calculate_psi(
    reference_series: pd.Series,
    current_series: pd.Series,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """
    Calculate Population Stability Index for numerical data.

    PSI interpretation:
    - PSI < 0.10: No significant drift
    - 0.10 <= PSI < 0.25: Moderate drift
    - PSI >= 0.25: Significant drift

    Args:
        reference_series (pd.Series): Baseline data.
        current_series (pd.Series): Current data.
        bins (int): Number of bins.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        float: PSI value.
    """
    reference = pd.to_numeric(reference_series, errors="coerce").dropna()
    current = pd.to_numeric(current_series, errors="coerce").dropna()

    if reference.empty or current.empty:
        return 0.0

    try:
        quantiles = np.linspace(0, 1, bins + 1)
        breakpoints = np.unique(np.quantile(reference, quantiles))

        if len(breakpoints) <= 2:
            min_value = min(reference.min(), current.min())
            max_value = max(reference.max(), current.max())

            if min_value == max_value:
                return 0.0

            breakpoints = np.linspace(min_value, max_value, bins + 1)

        reference_counts, _ = np.histogram(reference, bins=breakpoints)
        current_counts, _ = np.histogram(current, bins=breakpoints)

        reference_percents = reference_counts / max(len(reference), 1)
        current_percents = current_counts / max(len(current), 1)

        reference_percents = np.where(
            reference_percents == 0,
            epsilon,
            reference_percents,
        )
        current_percents = np.where(
            current_percents == 0,
            epsilon,
            current_percents,
        )

        psi_values = (
            current_percents - reference_percents
        ) * np.log(current_percents / reference_percents)

        return float(np.sum(psi_values))

    except Exception as error:
        logger.warning("PSI calculation failed: %s", error)
        return 0.0


def get_psi_drift_level(psi_value: float) -> str:
    """
    Convert PSI value into a drift severity level.

    Args:
        psi_value (float): PSI value.

    Returns:
        str: Drift level.
    """
    if psi_value < 0.10:
        return "no_drift"

    if psi_value < 0.25:
        return "moderate_drift"

    return "significant_drift"


# ============================================================
# Statistical Drift Tests
# ============================================================

def calculate_ks_test(
    reference_series: pd.Series,
    current_series: pd.Series,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Run Kolmogorov-Smirnov test for numerical distributions.

    Args:
        reference_series (pd.Series): Baseline numerical data.
        current_series (pd.Series): Current numerical data.

    Returns:
        Tuple[Optional[float], Optional[float]]: KS statistic and p-value.
    """
    if ks_2samp is None:
        return None, None

    reference = pd.to_numeric(reference_series, errors="coerce").dropna()
    current = pd.to_numeric(current_series, errors="coerce").dropna()

    if reference.empty or current.empty:
        return None, None

    try:
        result = ks_2samp(reference, current)
        return float(result.statistic), float(result.pvalue)
    except Exception as error:
        logger.warning("KS test failed: %s", error)
        return None, None


def calculate_chi_square_test(
    reference_series: pd.Series,
    current_series: pd.Series,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Run chi-square test for categorical distributions.

    Args:
        reference_series (pd.Series): Baseline categorical data.
        current_series (pd.Series): Current categorical data.

    Returns:
        Tuple[Optional[float], Optional[float]]: Chi-square statistic and p-value.
    """
    if chi2_contingency is None:
        return None, None

    reference_counts = reference_series.fillna("missing").astype(str).value_counts()
    current_counts = current_series.fillna("missing").astype(str).value_counts()

    categories = sorted(set(reference_counts.index) | set(current_counts.index))

    if len(categories) <= 1:
        return None, None

    table = np.array(
        [
            [reference_counts.get(category, 0) for category in categories],
            [current_counts.get(category, 0) for category in categories],
        ]
    )

    try:
        chi2_stat, p_value, _, _ = chi2_contingency(table)
        return float(chi2_stat), float(p_value)
    except Exception as error:
        logger.warning("Chi-square test failed: %s", error)
        return None, None


def is_pvalue_drift(
    p_value: Optional[float],
    threshold: float = 0.05,
) -> bool:
    """
    Check if p-value indicates drift.

    Args:
        p_value (Optional[float]): Statistical test p-value.
        threshold (float): Significance threshold.

    Returns:
        bool: True if drift is detected.
    """
    if p_value is None:
        return False

    return p_value < threshold


# ============================================================
# Drift Detector Class
# ============================================================

class DriftDetector:
    """
    Detect data drift between reference and current datasets.
    """

    def __init__(
        self,
        psi_threshold_moderate: float = 0.10,
        psi_threshold_significant: float = 0.25,
        p_value_threshold: float = 0.05,
        bins: int = 10,
    ) -> None:
        """
        Initialize DriftDetector.

        Args:
            psi_threshold_moderate (float): PSI threshold for moderate drift.
            psi_threshold_significant (float): PSI threshold for significant drift.
            p_value_threshold (float): Statistical p-value threshold.
            bins (int): Number of bins for PSI calculation.
        """
        self.psi_threshold_moderate = psi_threshold_moderate
        self.psi_threshold_significant = psi_threshold_significant
        self.p_value_threshold = p_value_threshold
        self.bins = bins

    def detect_numeric_feature_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        numerical_columns: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect drift for numerical features.

        Args:
            reference_data (pd.DataFrame): Baseline data.
            current_data (pd.DataFrame): Current data.
            numerical_columns (Optional[List[str]]): Numeric columns to check.

        Returns:
            List[Dict[str, Any]]: Numeric drift records.
        """
        validate_drift_inputs(reference_data, current_data)

        if numerical_columns is None:
            numerical_columns = sorted(
                set(get_numeric_columns(reference_data))
                & set(get_numeric_columns(current_data))
            )

        records = []

        for column in numerical_columns:
            if column not in reference_data.columns or column not in current_data.columns:
                continue

            psi_value = calculate_psi(
                reference_data[column],
                current_data[column],
                bins=self.bins,
            )

            ks_statistic, ks_p_value = calculate_ks_test(
                reference_data[column],
                current_data[column],
            )

            psi_level = get_psi_drift_level(psi_value)
            statistical_drift = is_pvalue_drift(
                ks_p_value,
                self.p_value_threshold,
            )

            drift_detected = (
                psi_value >= self.psi_threshold_moderate
                or statistical_drift
            )

            record = {
                "timestamp": get_current_timestamp(),
                "feature": column,
                "feature_type": "numerical",
                "reference_mean": safe_float(reference_data[column].mean()),
                "current_mean": safe_float(current_data[column].mean()),
                "reference_std": safe_float(reference_data[column].std()),
                "current_std": safe_float(current_data[column].std()),
                "reference_missing_rate": safe_float(reference_data[column].isna().mean()),
                "current_missing_rate": safe_float(current_data[column].isna().mean()),
                "psi": round(float(psi_value), 6),
                "psi_drift_level": psi_level,
                "ks_statistic": safe_float(ks_statistic),
                "ks_p_value": safe_float(ks_p_value),
                "statistical_drift": statistical_drift,
                "drift_detected": drift_detected,
            }

            records.append(record)

        return records

    def detect_categorical_feature_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        categorical_columns: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect drift for categorical features.

        Args:
            reference_data (pd.DataFrame): Baseline data.
            current_data (pd.DataFrame): Current data.
            categorical_columns (Optional[List[str]]): Categorical columns to check.

        Returns:
            List[Dict[str, Any]]: Categorical drift records.
        """
        validate_drift_inputs(reference_data, current_data)

        if categorical_columns is None:
            categorical_columns = sorted(
                set(get_categorical_columns(reference_data))
                & set(get_categorical_columns(current_data))
            )

        records = []

        for column in categorical_columns:
            if column not in reference_data.columns or column not in current_data.columns:
                continue

            chi2_statistic, chi2_p_value = calculate_chi_square_test(
                reference_data[column],
                current_data[column],
            )

            reference_unique = reference_data[column].nunique(dropna=True)
            current_unique = current_data[column].nunique(dropna=True)

            reference_top = (
                reference_data[column]
                .fillna("missing")
                .astype(str)
                .value_counts(normalize=True)
                .head(1)
            )

            current_top = (
                current_data[column]
                .fillna("missing")
                .astype(str)
                .value_counts(normalize=True)
                .head(1)
            )

            reference_top_category = (
                reference_top.index[0] if not reference_top.empty else None
            )
            current_top_category = (
                current_top.index[0] if not current_top.empty else None
            )

            reference_top_share = (
                float(reference_top.iloc[0]) if not reference_top.empty else None
            )
            current_top_share = (
                float(current_top.iloc[0]) if not current_top.empty else None
            )

            statistical_drift = is_pvalue_drift(
                chi2_p_value,
                self.p_value_threshold,
            )

            category_count_change = abs(current_unique - reference_unique)
            top_category_changed = reference_top_category != current_top_category

            drift_detected = statistical_drift or top_category_changed

            record = {
                "timestamp": get_current_timestamp(),
                "feature": column,
                "feature_type": "categorical",
                "reference_unique_count": int(reference_unique),
                "current_unique_count": int(current_unique),
                "category_count_change": int(category_count_change),
                "reference_top_category": reference_top_category,
                "current_top_category": current_top_category,
                "reference_top_category_share": safe_float(reference_top_share),
                "current_top_category_share": safe_float(current_top_share),
                "reference_missing_rate": safe_float(reference_data[column].isna().mean()),
                "current_missing_rate": safe_float(current_data[column].isna().mean()),
                "chi2_statistic": safe_float(chi2_statistic),
                "chi2_p_value": safe_float(chi2_p_value),
                "statistical_drift": statistical_drift,
                "top_category_changed": top_category_changed,
                "drift_detected": drift_detected,
            }

            records.append(record)

        return records

    def detect_target_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        target_column: str = "portfolio_value",
    ) -> Dict[str, Any]:
        """
        Detect drift in the target column.

        Args:
            reference_data (pd.DataFrame): Baseline data.
            current_data (pd.DataFrame): Current data.
            target_column (str): Target column name.

        Returns:
            Dict[str, Any]: Target drift record.
        """
        validate_drift_inputs(reference_data, current_data)

        if target_column not in reference_data.columns:
            raise ValueError(f"Target column missing in reference data: {target_column}")

        if target_column not in current_data.columns:
            raise ValueError(f"Target column missing in current data: {target_column}")

        psi_value = calculate_psi(
            reference_data[target_column],
            current_data[target_column],
            bins=self.bins,
        )

        ks_statistic, ks_p_value = calculate_ks_test(
            reference_data[target_column],
            current_data[target_column],
        )

        reference_mean = safe_float(reference_data[target_column].mean())
        current_mean = safe_float(current_data[target_column].mean())

        mean_change_percent = None
        if reference_mean not in [None, 0]:
            mean_change_percent = (
                (current_mean - reference_mean) / reference_mean
            ) * 100

        statistical_drift = is_pvalue_drift(
            ks_p_value,
            self.p_value_threshold,
        )

        drift_detected = (
            psi_value >= self.psi_threshold_moderate
            or statistical_drift
        )

        return {
            "timestamp": get_current_timestamp(),
            "feature": target_column,
            "feature_type": "target",
            "reference_mean": reference_mean,
            "current_mean": current_mean,
            "mean_change_percent": safe_float(mean_change_percent),
            "reference_median": safe_float(reference_data[target_column].median()),
            "current_median": safe_float(current_data[target_column].median()),
            "reference_std": safe_float(reference_data[target_column].std()),
            "current_std": safe_float(current_data[target_column].std()),
            "psi": round(float(psi_value), 6),
            "psi_drift_level": get_psi_drift_level(psi_value),
            "ks_statistic": safe_float(ks_statistic),
            "ks_p_value": safe_float(ks_p_value),
            "statistical_drift": statistical_drift,
            "drift_detected": drift_detected,
        }

    def detect_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        numerical_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        target_column: Optional[str] = "portfolio_value",
    ) -> Dict[str, Any]:
        """
        Run complete drift detection.

        Args:
            reference_data (pd.DataFrame): Baseline data.
            current_data (pd.DataFrame): Current data.
            numerical_columns (Optional[List[str]]): Numeric columns.
            categorical_columns (Optional[List[str]]): Categorical columns.
            target_column (Optional[str]): Target column.

        Returns:
            Dict[str, Any]: Complete drift report.
        """
        validate_drift_inputs(reference_data, current_data)

        logger.info("Starting drift detection...")

        numeric_records = self.detect_numeric_feature_drift(
            reference_data=reference_data,
            current_data=current_data,
            numerical_columns=numerical_columns,
        )

        categorical_records = self.detect_categorical_feature_drift(
            reference_data=reference_data,
            current_data=current_data,
            categorical_columns=categorical_columns,
        )

        target_record = None
        if (
            target_column
            and target_column in reference_data.columns
            and target_column in current_data.columns
        ):
            target_record = self.detect_target_drift(
                reference_data=reference_data,
                current_data=current_data,
                target_column=target_column,
            )

        all_records = numeric_records + categorical_records

        if target_record is not None:
            all_records.append(target_record)

        total_features_checked = len(all_records)
        drifted_features = [
            record["feature"]
            for record in all_records
            if record.get("drift_detected") is True
        ]

        summary = {
            "generated_at": get_current_timestamp(),
            "reference_rows": int(len(reference_data)),
            "current_rows": int(len(current_data)),
            "total_features_checked": total_features_checked,
            "drifted_feature_count": len(drifted_features),
            "drifted_features": drifted_features,
            "drift_detected": len(drifted_features) > 0,
            "psi_threshold_moderate": self.psi_threshold_moderate,
            "psi_threshold_significant": self.psi_threshold_significant,
            "p_value_threshold": self.p_value_threshold,
        }

        report = {
            "summary": summary,
            "numeric_drift": numeric_records,
            "categorical_drift": categorical_records,
            "target_drift": target_record,
            "all_records": all_records,
        }

        logger.info(
            "Drift detection completed. Features checked: %s | Drifted features: %s",
            total_features_checked,
            len(drifted_features),
        )

        return report

    def save_json_report(
        self,
        report: Dict[str, Any],
        file_path: Path = DRIFT_REPORT_JSON_PATH,
    ) -> None:
        """
        Save drift report as JSON.

        Args:
            report (Dict[str, Any]): Drift report.
            file_path (Path): Output JSON path.
        """
        ensure_drift_dirs()
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=4)

        logger.info("Drift JSON report saved at: %s", file_path)

    def save_csv_report(
        self,
        report: Dict[str, Any],
        file_path: Path = DRIFT_REPORT_CSV_PATH,
    ) -> None:
        """
        Save drift report records as CSV.

        Args:
            report (Dict[str, Any]): Drift report.
            file_path (Path): Output CSV path.
        """
        ensure_drift_dirs()
        file_path.parent.mkdir(parents=True, exist_ok=True)

        records = report.get("all_records", [])

        if not records:
            logger.warning("No drift records available to save as CSV.")
            return

        dataframe = pd.DataFrame(records)
        dataframe.to_csv(file_path, index=False)

        logger.info("Drift CSV report saved at: %s", file_path)

    def save_markdown_report(
        self,
        report: Dict[str, Any],
        file_path: Path = DRIFT_REPORT_MD_PATH,
    ) -> None:
        """
        Save drift report as Markdown.

        Args:
            report (Dict[str, Any]): Drift report.
            file_path (Path): Output Markdown path.
        """
        ensure_drift_dirs()
        file_path.parent.mkdir(parents=True, exist_ok=True)

        summary = report.get("summary", {})
        records = report.get("all_records", [])

        lines = [
            "# Data Drift Report",
            "",
            "## Project",
            "",
            "Financial Portfolio Forecasting & Anomaly Detection",
            "",
            "## Summary",
            "",
            f"- Generated At: {summary.get('generated_at')}",
            f"- Reference Rows: {summary.get('reference_rows')}",
            f"- Current Rows: {summary.get('current_rows')}",
            f"- Total Features Checked: {summary.get('total_features_checked')}",
            f"- Drifted Feature Count: {summary.get('drifted_feature_count')}",
            f"- Drift Detected: {summary.get('drift_detected')}",
            "",
            "## Drifted Features",
            "",
        ]

        drifted_features = summary.get("drifted_features", [])

        if drifted_features:
            for feature in drifted_features:
                lines.append(f"- {feature}")
        else:
            lines.append("No drifted features detected.")

        lines.extend(
            [
                "",
                "## Feature-Level Drift Details",
                "",
                "| Feature | Type | Drift Detected | PSI | PSI Level | Statistical Drift |",
                "|---|---|---:|---:|---|---:|",
            ]
        )

        for record in records:
            lines.append(
                "| {feature} | {feature_type} | {drift_detected} | {psi} | "
                "{psi_drift_level} | {statistical_drift} |".format(
                    feature=record.get("feature"),
                    feature_type=record.get("feature_type"),
                    drift_detected=record.get("drift_detected"),
                    psi=record.get("psi", ""),
                    psi_drift_level=record.get("psi_drift_level", ""),
                    statistical_drift=record.get("statistical_drift"),
                )
            )

        lines.extend(
            [
                "",
                "## Interpretation Guide",
                "",
                "- PSI below 0.10 usually means no major drift.",
                "- PSI from 0.10 to 0.25 usually means moderate drift.",
                "- PSI above 0.25 usually means significant drift.",
                "- A low p-value may indicate that the current distribution differs from the reference distribution.",
                "",
                "## Recommendation",
                "",
            ]
        )

        if summary.get("drift_detected"):
            lines.extend(
                [
                    "Drift was detected. Review the drifted features before using the model for production predictions.",
                    "If business conditions changed, retrain the forecasting and anomaly detection models using updated data.",
                ]
            )
        else:
            lines.append(
                "No major drift was detected. The current data appears reasonably close to the reference data."
            )

        file_path.write_text("\n".join(lines), encoding="utf-8")

        logger.info("Drift Markdown report saved at: %s", file_path)

    def save_all_reports(
        self,
        report: Dict[str, Any],
    ) -> None:
        """
        Save drift report in JSON, CSV, and Markdown formats.

        Args:
            report (Dict[str, Any]): Drift report.
        """
        self.save_json_report(report)
        self.save_csv_report(report)
        self.save_markdown_report(report)


# ============================================================
# Convenience Functions
# ============================================================

def detect_data_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    numerical_columns: Optional[List[str]] = None,
    categorical_columns: Optional[List[str]] = None,
    target_column: Optional[str] = "portfolio_value",
    save_reports: bool = True,
) -> Dict[str, Any]:
    """
    Detect data drift using the default DriftDetector.

    Args:
        reference_data (pd.DataFrame): Baseline data.
        current_data (pd.DataFrame): Current data.
        numerical_columns (Optional[List[str]]): Numerical columns.
        categorical_columns (Optional[List[str]]): Categorical columns.
        target_column (Optional[str]): Target column.
        save_reports (bool): Whether to save reports.

    Returns:
        Dict[str, Any]: Drift report.
    """
    detector = DriftDetector()

    report = detector.detect_drift(
        reference_data=reference_data,
        current_data=current_data,
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        target_column=target_column,
    )

    if save_reports:
        detector.save_all_reports(report)

    return report


def load_csv_data(file_path: Path) -> pd.DataFrame:
    """
    Load CSV data from file.

    Args:
        file_path (Path): CSV file path.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    return pd.read_csv(file_path)


def run_drift_detection_from_files(
    reference_file_path: Path,
    current_file_path: Path,
    target_column: str = "portfolio_value",
    save_reports: bool = True,
) -> Dict[str, Any]:
    """
    Run drift detection from two CSV files.

    Args:
        reference_file_path (Path): Reference CSV path.
        current_file_path (Path): Current CSV path.
        target_column (str): Target column.
        save_reports (bool): Whether to save reports.

    Returns:
        Dict[str, Any]: Drift report.
    """
    reference_data = load_csv_data(reference_file_path)
    current_data = load_csv_data(current_file_path)

    return detect_data_drift(
        reference_data=reference_data,
        current_data=current_data,
        target_column=target_column,
        save_reports=save_reports,
    )


# ============================================================
# Script Usage Example
# ============================================================

if __name__ == "__main__":
    reference_path = PROJECT_ROOT / "data" / "processed" / "train_data.csv"
    current_path = PROJECT_ROOT / "data" / "processed" / "test_data.csv"

    if reference_path.exists() and current_path.exists():
        drift_report = run_drift_detection_from_files(
            reference_file_path=reference_path,
            current_file_path=current_path,
            target_column="portfolio_value",
            save_reports=True,
        )

        print("Drift detection completed.")
        print(drift_report["summary"])
    else:
        print("Sample files not found.")
        print(f"Expected reference file: {reference_path}")
        print(f"Expected current file: {current_path}")