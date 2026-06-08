"""
Anomaly detection model for the Financial Portfolio Forecasting
and Anomaly Detection project.

This module handles:
- IQR-based anomaly detection
- Z-score anomaly detection
- Isolation Forest anomaly detection
- Asset-wise price anomaly detection
- Anomaly report generation
- Model artifact saving and loading

Dataset expected columns:
- date
- asset
- price
"""

import json
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.monitoring.logger import get_project_logger
from src.utils.paths import create_parent_directory


warnings.filterwarnings("ignore")

logger = get_project_logger()


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

ANOMALY_MODEL_PATH = MODELS_DIR / "anomaly_detection_model.pkl"
ANOMALY_RESULTS_PATH = OUTPUTS_DIR / "anomaly_detection_results.csv"
ANOMALY_SUMMARY_JSON_PATH = REPORTS_TABLES_DIR / "anomaly_summary.json"
ANOMALY_SUMMARY_CSV_PATH = REPORTS_TABLES_DIR / "anomaly_summary.csv"
ANOMALY_REPORT_PATH = REPORTS_DIR / "anomaly_report.md"


# ============================================================
# Helper Functions
# ============================================================

def ensure_directories() -> None:
    """
    Create required directories for anomaly detection outputs.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def clean_input_data(
    dataframe: pd.DataFrame,
    date_column: str = "date",
    asset_column: str = "asset",
    target_column: str = "price",
) -> pd.DataFrame:
    """
    Clean and standardize input dataframe.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        date_column (str): Date column name.
        asset_column (str): Asset column name.
        target_column (str): Target column name.

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

    required_columns = [date_column, asset_column, target_column]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    data[target_column] = pd.to_numeric(data[target_column], errors="coerce")

    data = data.dropna(subset=[date_column, asset_column, target_column])
    data = data[data[target_column] >= 0]

    data = data.sort_values([asset_column, date_column]).reset_index(drop=True)

    return data


def add_anomaly_features(
    dataframe: pd.DataFrame,
    date_column: str = "date",
    asset_column: str = "asset",
    target_column: str = "price",
) -> pd.DataFrame:
    """
    Create anomaly detection features.

    Features created:
    - daily_return
    - price_lag_1
    - price_lag_3
    - price_lag_7
    - price_rolling_mean_7
    - price_rolling_std_7
    - price_rolling_mean_14
    - price_rolling_std_14
    - price_rolling_mean_30
    - price_rolling_std_30
    - rolling volatility features
    - calendar features

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        date_column (str): Date column.
        asset_column (str): Asset column.
        target_column (str): Target column.

    Returns:
        pd.DataFrame: Dataframe with anomaly features.
    """
    data = dataframe.copy()

    data = data.sort_values([asset_column, date_column]).reset_index(drop=True)

    data["daily_return"] = (
        data.groupby(asset_column)[target_column]
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
    )

    data["daily_return"] = data["daily_return"].fillna(0)

    for lag in [1, 3, 7]:
        data[f"{target_column}_lag_{lag}"] = (
            data.groupby(asset_column)[target_column]
            .shift(lag)
        )

    for window in [7, 14, 30]:
        data[f"{target_column}_rolling_mean_{window}"] = (
            data.groupby(asset_column)[target_column]
            .transform(
                lambda series: series.rolling(
                    window=window,
                    min_periods=1,
                ).mean()
            )
        )

        data[f"{target_column}_rolling_std_{window}"] = (
            data.groupby(asset_column)[target_column]
            .transform(
                lambda series: series.rolling(
                    window=window,
                    min_periods=1,
                ).std()
            )
        )

        data[f"return_rolling_volatility_{window}"] = (
            data.groupby(asset_column)["daily_return"]
            .transform(
                lambda series: series.rolling(
                    window=window,
                    min_periods=1,
                ).std()
            )
        )

    data["year"] = data[date_column].dt.year
    data["month"] = data[date_column].dt.month
    data["quarter"] = data[date_column].dt.quarter
    data["day"] = data[date_column].dt.day
    data["day_of_week"] = data[date_column].dt.dayofweek
    data["week_of_year"] = data[date_column].dt.isocalendar().week.astype(int)
    data["is_month_start"] = data[date_column].dt.is_month_start.astype(int)
    data["is_month_end"] = data[date_column].dt.is_month_end.astype(int)

    data = data.replace([np.inf, -np.inf], np.nan)

    numeric_columns = data.select_dtypes(include=["number"]).columns

    for column in numeric_columns:
        if data[column].isna().sum() > 0:
            data[column] = data[column].fillna(data[column].median())

    data = data.fillna(0)

    return data


def calculate_iqr_bounds(
    series: pd.Series,
    multiplier: float = 1.5,
) -> Dict[str, float]:
    """
    Calculate IQR lower and upper bounds.

    Args:
        series (pd.Series): Numeric series.
        multiplier (float): IQR multiplier.

    Returns:
        Dict[str, float]: IQR statistics.
    """
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    return {
        "q1": q1,
        "q3": q3,
        "iqr": float(iqr),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
    }


def safe_asset_filename(asset_name: str) -> str:
    """
    Convert asset name to safe filename format.

    Args:
        asset_name (str): Asset name.

    Returns:
        str: Safe asset filename.
    """
    return (
        str(asset_name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("-", "_")
    )


# ============================================================
# Anomaly Detector Class
# ============================================================

class FinancialAnomalyDetector:
    """
    Detect financial asset price anomalies using multiple methods.
    """

    def __init__(
        self,
        date_column: str = "date",
        asset_column: str = "asset",
        target_column: str = "price",
        iqr_multiplier: float = 1.5,
        z_score_threshold: float = 3.0,
        contamination: float = 0.05,
        random_state: int = 42,
    ) -> None:
        """
        Initialize FinancialAnomalyDetector.

        Args:
            date_column (str): Date column name.
            asset_column (str): Asset column name.
            target_column (str): Target column name.
            iqr_multiplier (float): IQR multiplier.
            z_score_threshold (float): Z-score threshold.
            contamination (float): Isolation Forest contamination rate.
            random_state (int): Random state.
        """
        self.date_column = date_column
        self.asset_column = asset_column
        self.target_column = target_column
        self.iqr_multiplier = iqr_multiplier
        self.z_score_threshold = z_score_threshold
        self.contamination = contamination
        self.random_state = random_state

        self.models: Dict[str, IsolationForest] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_columns: List[str] = []
        self.results: Optional[pd.DataFrame] = None
        self.summary: Dict[str, Any] = {}

        ensure_directories()

    def detect_iqr_anomalies(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies using the IQR method asset-wise.

        Args:
            dataframe (pd.DataFrame): Cleaned dataframe.

        Returns:
            pd.DataFrame: Dataframe with IQR anomaly columns.
        """
        data = dataframe.copy()

        iqr_frames = []

        for asset_name, asset_data in data.groupby(self.asset_column):
            asset_data = asset_data.copy()

            bounds = calculate_iqr_bounds(
                asset_data[self.target_column],
                multiplier=self.iqr_multiplier,
            )

            asset_data["iqr_q1"] = bounds["q1"]
            asset_data["iqr_q3"] = bounds["q3"]
            asset_data["iqr_value"] = bounds["iqr"]
            asset_data["iqr_lower_bound"] = bounds["lower_bound"]
            asset_data["iqr_upper_bound"] = bounds["upper_bound"]

            asset_data["is_anomaly_iqr"] = (
                (asset_data[self.target_column] < bounds["lower_bound"])
                | (asset_data[self.target_column] > bounds["upper_bound"])
            )

            iqr_frames.append(asset_data)

        result = pd.concat(iqr_frames, ignore_index=True)

        logger.info(
            "IQR anomaly detection completed. Detected anomalies: %s",
            int(result["is_anomaly_iqr"].sum()),
        )

        return result

    def detect_z_score_anomalies(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies using asset-wise Z-score.

        Args:
            dataframe (pd.DataFrame): Dataframe after IQR detection.

        Returns:
            pd.DataFrame: Dataframe with Z-score anomaly columns.
        """
        data = dataframe.copy()

        def calculate_group_z_score(group: pd.DataFrame) -> pd.DataFrame:
            group = group.copy()

            mean_value = group[self.target_column].mean()
            std_value = group[self.target_column].std()

            if std_value == 0 or pd.isna(std_value):
                group["price_z_score"] = 0.0
            else:
                group["price_z_score"] = (
                    group[self.target_column] - mean_value
                ) / std_value

            group["is_anomaly_zscore"] = (
                group["price_z_score"].abs() > self.z_score_threshold
            )

            return group

        result = (
            data.groupby(self.asset_column, group_keys=False)
            .apply(calculate_group_z_score)
            .reset_index(drop=True)
        )

        logger.info(
            "Z-score anomaly detection completed. Detected anomalies: %s",
            int(result["is_anomaly_zscore"].sum()),
        )

        return result

    def get_isolation_forest_features(
        self,
        dataframe: pd.DataFrame,
    ) -> List[str]:
        """
        Select numeric features for Isolation Forest.

        Args:
            dataframe (pd.DataFrame): Feature dataframe.

        Returns:
            List[str]: Feature column names.
        """
        preferred_features = [
            self.target_column,
            "daily_return",
            f"{self.target_column}_lag_1",
            f"{self.target_column}_lag_3",
            f"{self.target_column}_lag_7",
            f"{self.target_column}_rolling_mean_7",
            f"{self.target_column}_rolling_std_7",
            f"{self.target_column}_rolling_mean_14",
            f"{self.target_column}_rolling_std_14",
            f"{self.target_column}_rolling_mean_30",
            f"{self.target_column}_rolling_std_30",
            "return_rolling_volatility_7",
            "return_rolling_volatility_14",
            "return_rolling_volatility_30",
            "month",
            "quarter",
            "day_of_week",
        ]

        feature_columns = [
            column for column in preferred_features
            if column in dataframe.columns
            and pd.api.types.is_numeric_dtype(dataframe[column])
        ]

        if not feature_columns:
            feature_columns = [
                column for column in dataframe.select_dtypes(include=["number"]).columns
                if column not in ["year"]
            ]

        return feature_columns

    def detect_isolation_forest_anomalies(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Detect anomalies using Isolation Forest asset-wise.

        Args:
            dataframe (pd.DataFrame): Dataframe with anomaly features.

        Returns:
            pd.DataFrame: Dataframe with Isolation Forest anomaly columns.
        """
        data = dataframe.copy()

        self.feature_columns = self.get_isolation_forest_features(data)

        if not self.feature_columns:
            raise ValueError("No valid numeric features found for Isolation Forest.")

        output_frames = []

        for asset_name, asset_data in data.groupby(self.asset_column):
            asset_data = asset_data.copy()

            X = asset_data[self.feature_columns].copy()

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = IsolationForest(
                n_estimators=200,
                contamination=self.contamination,
                random_state=self.random_state,
            )

            anomaly_labels = model.fit_predict(X_scaled)
            anomaly_scores = model.decision_function(X_scaled)

            asset_data["isolation_forest_label"] = anomaly_labels
            asset_data["is_anomaly_isolation_forest"] = anomaly_labels == -1
            asset_data["anomaly_score"] = anomaly_scores

            self.models[asset_name] = model
            self.scalers[asset_name] = scaler

            output_frames.append(asset_data)

        result = pd.concat(output_frames, ignore_index=True)

        logger.info(
            "Isolation Forest anomaly detection completed. Detected anomalies: %s",
            int(result["is_anomaly_isolation_forest"].sum()),
        )

        return result

    def combine_anomaly_flags(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Combine anomaly flags from all methods.

        Args:
            dataframe (pd.DataFrame): Dataframe with method-specific anomaly flags.

        Returns:
            pd.DataFrame: Final anomaly results dataframe.
        """
        data = dataframe.copy()

        anomaly_columns = [
            "is_anomaly_iqr",
            "is_anomaly_zscore",
            "is_anomaly_isolation_forest",
        ]

        for column in anomaly_columns:
            if column not in data.columns:
                data[column] = False

        data["anomaly_method_count"] = data[anomaly_columns].sum(axis=1)

        data["is_anomaly"] = data["anomaly_method_count"] > 0

        data["anomaly_method"] = data.apply(
            lambda row: ", ".join(
                [
                    method_name
                    for method_name, column_name in [
                        ("IQR", "is_anomaly_iqr"),
                        ("Z-Score", "is_anomaly_zscore"),
                        ("Isolation Forest", "is_anomaly_isolation_forest"),
                    ]
                    if bool(row[column_name])
                ]
            )
            if row["is_anomaly"]
            else "None",
            axis=1,
        )

        data["anomaly_severity"] = data["anomaly_method_count"].map(
            {
                0: "normal",
                1: "low",
                2: "medium",
                3: "high",
            }
        )

        data = data.sort_values(
            [self.asset_column, self.date_column]
        ).reset_index(drop=True)

        return data

    def fit_detect(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Run complete anomaly detection workflow.

        Args:
            dataframe (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Anomaly detection results.
        """
        logger.info("Starting complete anomaly detection workflow.")

        data = clean_input_data(
            dataframe=dataframe,
            date_column=self.date_column,
            asset_column=self.asset_column,
            target_column=self.target_column,
        )

        data = add_anomaly_features(
            dataframe=data,
            date_column=self.date_column,
            asset_column=self.asset_column,
            target_column=self.target_column,
        )

        data = self.detect_iqr_anomalies(data)
        data = self.detect_z_score_anomalies(data)
        data = self.detect_isolation_forest_anomalies(data)
        data = self.combine_anomaly_flags(data)

        self.results = data
        self.summary = self.generate_summary(data)

        logger.info("Anomaly detection workflow completed successfully.")

        return data

    def generate_summary(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate anomaly detection summary.

        Args:
            dataframe (pd.DataFrame): Anomaly result dataframe.

        Returns:
            Dict[str, Any]: Summary dictionary.
        """
        total_records = len(dataframe)
        total_anomalies = int(dataframe["is_anomaly"].sum())

        anomaly_rate = (
            total_anomalies / total_records * 100
            if total_records > 0
            else 0.0
        )

        asset_summary = (
            dataframe.groupby(self.asset_column)
            .agg(
                total_records=(self.target_column, "count"),
                anomaly_count=("is_anomaly", "sum"),
                iqr_anomaly_count=("is_anomaly_iqr", "sum"),
                zscore_anomaly_count=("is_anomaly_zscore", "sum"),
                isolation_forest_anomaly_count=(
                    "is_anomaly_isolation_forest",
                    "sum",
                ),
                average_price=(self.target_column, "mean"),
                min_price=(self.target_column, "min"),
                max_price=(self.target_column, "max"),
                average_anomaly_score=("anomaly_score", "mean"),
            )
            .reset_index()
        )

        asset_summary["anomaly_rate_percent"] = (
            asset_summary["anomaly_count"]
            / asset_summary["total_records"]
            * 100
        )

        severity_summary = (
            dataframe["anomaly_severity"]
            .value_counts()
            .to_dict()
        )

        method_summary = {
            "iqr_anomalies": int(dataframe["is_anomaly_iqr"].sum()),
            "zscore_anomalies": int(dataframe["is_anomaly_zscore"].sum()),
            "isolation_forest_anomalies": int(
                dataframe["is_anomaly_isolation_forest"].sum()
            ),
        }

        top_anomalies = (
            dataframe[dataframe["is_anomaly"]]
            .sort_values(
                by=["anomaly_method_count", "anomaly_score"],
                ascending=[False, True],
            )
            .head(10)
        )

        summary = {
            "total_records": int(total_records),
            "total_anomalies": int(total_anomalies),
            "anomaly_rate_percent": round(float(anomaly_rate), 4),
            "asset_count": int(dataframe[self.asset_column].nunique()),
            "feature_columns": self.feature_columns,
            "method_summary": method_summary,
            "severity_summary": severity_summary,
            "asset_summary": asset_summary.to_dict(orient="records"),
            "top_anomalies": top_anomalies[
                [
                    self.date_column,
                    self.asset_column,
                    self.target_column,
                    "anomaly_method",
                    "anomaly_severity",
                    "anomaly_score",
                ]
            ].to_dict(orient="records"),
        }

        return summary

    def predict_new_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies on new data using saved Isolation Forest models.

        IQR and Z-score are recalculated on the new input data.
        Isolation Forest uses fitted models and scalers from training.

        Args:
            dataframe (pd.DataFrame): New input dataframe.

        Returns:
            pd.DataFrame: New anomaly result dataframe.
        """
        if not self.models or not self.scalers:
            raise ValueError("No trained anomaly models found. Run fit_detect first.")

        data = clean_input_data(
            dataframe=dataframe,
            date_column=self.date_column,
            asset_column=self.asset_column,
            target_column=self.target_column,
        )

        data = add_anomaly_features(
            dataframe=data,
            date_column=self.date_column,
            asset_column=self.asset_column,
            target_column=self.target_column,
        )

        data = self.detect_iqr_anomalies(data)
        data = self.detect_z_score_anomalies(data)

        output_frames = []

        for asset_name, asset_data in data.groupby(self.asset_column):
            asset_data = asset_data.copy()

            if asset_name not in self.models or asset_name not in self.scalers:
                logger.warning(
                    "No trained Isolation Forest model for asset %s. "
                    "Skipping Isolation Forest prediction for this asset.",
                    asset_name,
                )
                asset_data["isolation_forest_label"] = 1
                asset_data["is_anomaly_isolation_forest"] = False
                asset_data["anomaly_score"] = 0.0
                output_frames.append(asset_data)
                continue

            for column in self.feature_columns:
                if column not in asset_data.columns:
                    asset_data[column] = 0

            X = asset_data[self.feature_columns].copy()

            scaler = self.scalers[asset_name]
            model = self.models[asset_name]

            X_scaled = scaler.transform(X)

            anomaly_labels = model.predict(X_scaled)
            anomaly_scores = model.decision_function(X_scaled)

            asset_data["isolation_forest_label"] = anomaly_labels
            asset_data["is_anomaly_isolation_forest"] = anomaly_labels == -1
            asset_data["anomaly_score"] = anomaly_scores

            output_frames.append(asset_data)

        result = pd.concat(output_frames, ignore_index=True)
        result = self.combine_anomaly_flags(result)

        return result

    def save_model(
        self,
        file_path: Path = ANOMALY_MODEL_PATH,
    ) -> None:
        """
        Save anomaly detection models, scalers, and metadata.

        Args:
            file_path (Path): Output model path.
        """
        if not self.models:
            raise ValueError("No anomaly detection models available to save.")

        create_parent_directory(file_path)

        with file_path.open("wb") as file:
            pickle.dump(
                {
                    "models": self.models,
                    "scalers": self.scalers,
                    "feature_columns": self.feature_columns,
                    "summary": self.summary,
                    "date_column": self.date_column,
                    "asset_column": self.asset_column,
                    "target_column": self.target_column,
                    "iqr_multiplier": self.iqr_multiplier,
                    "z_score_threshold": self.z_score_threshold,
                    "contamination": self.contamination,
                    "random_state": self.random_state,
                },
                file,
            )

        logger.info("Anomaly detection model saved at: %s", file_path)

    def load_model(
        self,
        file_path: Path = ANOMALY_MODEL_PATH,
    ) -> "FinancialAnomalyDetector":
        """
        Load saved anomaly detection model.

        Args:
            file_path (Path): Saved model path.

        Returns:
            FinancialAnomalyDetector: Loaded detector.
        """
        if not file_path.exists():
            raise FileNotFoundError(
                f"Anomaly detection model file not found: {file_path}"
            )

        with file_path.open("rb") as file:
            saved_data = pickle.load(file)

        self.models = saved_data.get("models", {})
        self.scalers = saved_data.get("scalers", {})
        self.feature_columns = saved_data.get("feature_columns", [])
        self.summary = saved_data.get("summary", {})

        self.date_column = saved_data.get("date_column", self.date_column)
        self.asset_column = saved_data.get("asset_column", self.asset_column)
        self.target_column = saved_data.get("target_column", self.target_column)
        self.iqr_multiplier = saved_data.get("iqr_multiplier", self.iqr_multiplier)
        self.z_score_threshold = saved_data.get(
            "z_score_threshold",
            self.z_score_threshold,
        )
        self.contamination = saved_data.get("contamination", self.contamination)
        self.random_state = saved_data.get("random_state", self.random_state)

        logger.info("Anomaly detection model loaded from: %s", file_path)

        return self

    def save_results(
        self,
        dataframe: Optional[pd.DataFrame] = None,
        file_path: Path = ANOMALY_RESULTS_PATH,
    ) -> None:
        """
        Save anomaly detection results as CSV.

        Args:
            dataframe (Optional[pd.DataFrame]): Results dataframe.
            file_path (Path): Output CSV path.
        """
        results_dataframe = dataframe if dataframe is not None else self.results

        if results_dataframe is None or results_dataframe.empty:
            raise ValueError("No anomaly detection results available to save.")

        create_parent_directory(file_path)

        results_dataframe.to_csv(file_path, index=False)

        logger.info("Anomaly detection results saved at: %s", file_path)

    def save_summary_json(
        self,
        file_path: Path = ANOMALY_SUMMARY_JSON_PATH,
    ) -> None:
        """
        Save anomaly summary as JSON.

        Args:
            file_path (Path): Output JSON path.
        """
        if not self.summary:
            raise ValueError("No anomaly summary available to save.")

        create_parent_directory(file_path)

        with file_path.open("w", encoding="utf-8") as file:
            json.dump(self.summary, file, indent=4, default=str)

        logger.info("Anomaly summary JSON saved at: %s", file_path)

    def save_summary_csv(
        self,
        file_path: Path = ANOMALY_SUMMARY_CSV_PATH,
    ) -> None:
        """
        Save asset-level anomaly summary as CSV.

        Args:
            file_path (Path): Output CSV path.
        """
        if not self.summary:
            raise ValueError("No anomaly summary available to save.")

        asset_summary = self.summary.get("asset_summary", [])

        if not asset_summary:
            raise ValueError("No asset-level anomaly summary available.")

        create_parent_directory(file_path)

        pd.DataFrame(asset_summary).to_csv(file_path, index=False)

        logger.info("Anomaly summary CSV saved at: %s", file_path)

    def save_markdown_report(
        self,
        file_path: Path = ANOMALY_REPORT_PATH,
    ) -> None:
        """
        Save anomaly detection report as Markdown.

        Args:
            file_path (Path): Output Markdown path.
        """
        if not self.summary:
            raise ValueError("No anomaly summary available to save.")

        create_parent_directory(file_path)

        method_summary = self.summary.get("method_summary", {})
        severity_summary = self.summary.get("severity_summary", {})
        asset_summary = self.summary.get("asset_summary", [])
        top_anomalies = self.summary.get("top_anomalies", [])

        lines = [
            "# Anomaly Detection Report",
            "",
            "## Project",
            "",
            "Financial Portfolio Forecasting & Anomaly Detection",
            "",
            "## Summary",
            "",
            f"- Total Records: {self.summary.get('total_records')}",
            f"- Total Anomalies: {self.summary.get('total_anomalies')}",
            f"- Anomaly Rate: {self.summary.get('anomaly_rate_percent')}%",
            f"- Asset Count: {self.summary.get('asset_count')}",
            "",
            "## Method Summary",
            "",
            f"- IQR Anomalies: {method_summary.get('iqr_anomalies', 0)}",
            f"- Z-Score Anomalies: {method_summary.get('zscore_anomalies', 0)}",
            f"- Isolation Forest Anomalies: {method_summary.get('isolation_forest_anomalies', 0)}",
            "",
            "## Severity Summary",
            "",
        ]

        for severity, count in severity_summary.items():
            lines.append(f"- {severity}: {count}")

        lines.extend(
            [
                "",
                "## Asset-Level Summary",
                "",
                "| Asset | Records | Anomalies | Anomaly Rate % | Avg Price | Min Price | Max Price |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )

        for row in asset_summary:
            lines.append(
                "| {asset} | {records} | {anomalies} | {rate:.2f} | {avg:.2f} | {minp:.2f} | {maxp:.2f} |".format(
                    asset=row.get(self.asset_column),
                    records=int(row.get("total_records", 0)),
                    anomalies=int(row.get("anomaly_count", 0)),
                    rate=float(row.get("anomaly_rate_percent", 0)),
                    avg=float(row.get("average_price", 0)),
                    minp=float(row.get("min_price", 0)),
                    maxp=float(row.get("max_price", 0)),
                )
            )

        lines.extend(
            [
                "",
                "## Top Anomalies",
                "",
                "| Date | Asset | Price | Method | Severity | Score |",
                "|---|---|---:|---|---|---:|",
            ]
        )

        if top_anomalies:
            for row in top_anomalies:
                lines.append(
                    "| {date} | {asset} | {price:.2f} | {method} | {severity} | {score:.6f} |".format(
                        date=row.get(self.date_column),
                        asset=row.get(self.asset_column),
                        price=float(row.get(self.target_column, 0)),
                        method=row.get("anomaly_method"),
                        severity=row.get("anomaly_severity"),
                        score=float(row.get("anomaly_score", 0)),
                    )
                )
        else:
            lines.append("| No anomalies detected | - | - | - | - | - |")

        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                "- IQR detects values outside the normal price spread.",
                "- Z-score detects prices far away from the asset average.",
                "- Isolation Forest detects unusual patterns using multiple engineered features.",
                "- High severity means multiple methods marked the same record as anomalous.",
                "",
                "## Business Use",
                "",
                "This anomaly detection system can help identify unusual asset price movements, possible data quality issues, and high-risk market behavior for financial monitoring.",
            ]
        )

        file_path.write_text("\n".join(lines), encoding="utf-8")

        logger.info("Anomaly Markdown report saved at: %s", file_path)

    def save_all_artifacts(self) -> None:
        """
        Save model, results, summaries, and report.
        """
        self.save_model()
        self.save_results()
        self.save_summary_json()
        self.save_summary_csv()
        self.save_markdown_report()

    def fit_detect_save(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        """
        Run complete anomaly detection workflow and save outputs.

        Args:
            dataframe (pd.DataFrame): Input dataframe.

        Returns:
            Dict[str, Any]: Workflow summary.
        """
        results = self.fit_detect(dataframe)
        self.save_all_artifacts()

        workflow_summary = {
            "status": "success",
            "model_type": "Multi-method anomaly detection",
            "methods": [
                "IQR",
                "Z-Score",
                "Isolation Forest",
            ],
            "total_records": len(results),
            "total_anomalies": int(results["is_anomaly"].sum()),
            "anomaly_rate_percent": self.summary.get("anomaly_rate_percent"),
            "asset_count": self.summary.get("asset_count"),
            "model_path": str(ANOMALY_MODEL_PATH),
            "results_path": str(ANOMALY_RESULTS_PATH),
            "summary_json_path": str(ANOMALY_SUMMARY_JSON_PATH),
            "summary_csv_path": str(ANOMALY_SUMMARY_CSV_PATH),
            "report_path": str(ANOMALY_REPORT_PATH),
        }

        logger.info("Anomaly detection workflow summary: %s", workflow_summary)

        return workflow_summary


# ============================================================
# Convenience Functions
# ============================================================

def detect_anomalies(
    dataframe: pd.DataFrame,
    iqr_multiplier: float = 1.5,
    z_score_threshold: float = 3.0,
    contamination: float = 0.05,
) -> pd.DataFrame:
    """
    Detect anomalies and return results without saving.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        iqr_multiplier (float): IQR multiplier.
        z_score_threshold (float): Z-score threshold.
        contamination (float): Isolation Forest contamination.

    Returns:
        pd.DataFrame: Anomaly results.
    """
    detector = FinancialAnomalyDetector(
        iqr_multiplier=iqr_multiplier,
        z_score_threshold=z_score_threshold,
        contamination=contamination,
    )

    return detector.fit_detect(dataframe)


def detect_and_save_anomalies(
    dataframe: pd.DataFrame,
    iqr_multiplier: float = 1.5,
    z_score_threshold: float = 3.0,
    contamination: float = 0.05,
) -> Dict[str, Any]:
    """
    Detect anomalies and save all artifacts.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        iqr_multiplier (float): IQR multiplier.
        z_score_threshold (float): Z-score threshold.
        contamination (float): Isolation Forest contamination.

    Returns:
        Dict[str, Any]: Workflow summary.
    """
    detector = FinancialAnomalyDetector(
        iqr_multiplier=iqr_multiplier,
        z_score_threshold=z_score_threshold,
        contamination=contamination,
    )

    return detector.fit_detect_save(dataframe)


def load_anomaly_detector(
    model_path: Path = ANOMALY_MODEL_PATH,
) -> FinancialAnomalyDetector:
    """
    Load saved anomaly detector.

    Args:
        model_path (Path): Saved model path.

    Returns:
        FinancialAnomalyDetector: Loaded detector.
    """
    detector = FinancialAnomalyDetector()
    detector.load_model(model_path)

    return detector


def predict_anomalies_with_saved_model(
    dataframe: pd.DataFrame,
    model_path: Path = ANOMALY_MODEL_PATH,
) -> pd.DataFrame:
    """
    Predict anomalies using a saved anomaly detector.

    Args:
        dataframe (pd.DataFrame): New dataframe.
        model_path (Path): Saved anomaly model path.

    Returns:
        pd.DataFrame: Anomaly result dataframe.
    """
    detector = load_anomaly_detector(model_path)

    return detector.predict_new_data(dataframe)


# ============================================================
# Script Usage Example
# ============================================================

if __name__ == "__main__":
    raw_data_path = PROJECT_ROOT / "data" / "raw" / "financial_portfolio_data.csv"

    if not raw_data_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at: {raw_data_path}"
        )

    data = pd.read_csv(raw_data_path)

    anomaly_detector = FinancialAnomalyDetector(
        date_column="date",
        asset_column="asset",
        target_column="price",
        iqr_multiplier=1.5,
        z_score_threshold=3.0,
        contamination=0.05,
        random_state=42,
    )

    summary = anomaly_detector.fit_detect_save(data)

    print("Anomaly detection completed successfully.")
    print(summary)