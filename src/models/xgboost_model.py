"""
XGBoost forecasting model for the Financial Portfolio Forecasting
and Anomaly Detection project.

This module handles:
- Feature preparation for asset price forecasting
- Time-based train-test split
- XGBoost model training
- Forecasting/evaluation
- Model artifact saving and loading
- Metrics and prediction output saving

Dataset expected columns:
- date
- asset
- price
"""

import json
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError(
        "xgboost is required for this model. "
        "Install it using: pip install xgboost"
    ) from exc

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

XGBOOST_MODEL_PATH = MODELS_DIR / "xgboost_model.pkl"
XGBOOST_FEATURE_COLUMNS_PATH = MODELS_DIR / "xgboost_feature_columns.json"
XGBOOST_METRICS_JSON_PATH = REPORTS_TABLES_DIR / "xgboost_metrics.json"
XGBOOST_METRICS_CSV_PATH = REPORTS_TABLES_DIR / "xgboost_metrics.csv"
XGBOOST_PREDICTIONS_PATH = OUTPUTS_DIR / "xgboost_predictions.csv"
XGBOOST_FORECAST_PATH = OUTPUTS_DIR / "xgboost_forecast_results.csv"


# ============================================================
# Helper Functions
# ============================================================

def ensure_directories() -> None:
    """
    Create required directories for models, reports, and outputs.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def calculate_rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Calculate Root Mean Squared Error.

    Args:
        y_true (pd.Series): Actual values.
        y_pred (pd.Series): Predicted values.

    Returns:
        float: RMSE value.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calculate_mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    Args:
        y_true (pd.Series): Actual values.
        y_pred (pd.Series): Predicted values.

    Returns:
        float: MAPE percentage value.
    """
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)

    non_zero_mask = y_true_array != 0

    if non_zero_mask.sum() == 0:
        return 0.0

    return float(
        np.mean(
            np.abs(
                (y_true_array[non_zero_mask] - y_pred_array[non_zero_mask])
                / y_true_array[non_zero_mask]
            )
        )
        * 100
    )


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


def add_time_features(
    dataframe: pd.DataFrame,
    date_column: str = "date",
) -> pd.DataFrame:
    """
    Add calendar-based time features.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        date_column (str): Date column name.

    Returns:
        pd.DataFrame: Dataframe with time features.
    """
    data = dataframe.copy()

    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")

    data["year"] = data[date_column].dt.year
    data["month"] = data[date_column].dt.month
    data["quarter"] = data[date_column].dt.quarter
    data["day"] = data[date_column].dt.day
    data["day_of_week"] = data[date_column].dt.dayofweek
    data["week_of_year"] = data[date_column].dt.isocalendar().week.astype(int)
    data["is_month_start"] = data[date_column].dt.is_month_start.astype(int)
    data["is_month_end"] = data[date_column].dt.is_month_end.astype(int)
    data["is_quarter_start"] = data[date_column].dt.is_quarter_start.astype(int)
    data["is_quarter_end"] = data[date_column].dt.is_quarter_end.astype(int)

    return data


def add_return_features(
    dataframe: pd.DataFrame,
    asset_column: str = "asset",
    target_column: str = "price",
) -> pd.DataFrame:
    """
    Add percentage return features.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        asset_column (str): Asset column name.
        target_column (str): Target column name.

    Returns:
        pd.DataFrame: Dataframe with return features.
    """
    data = dataframe.copy()

    data["daily_return"] = (
        data.groupby(asset_column)[target_column]
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
    )

    data["daily_return"] = data["daily_return"].fillna(0)

    return data


def add_lag_features(
    dataframe: pd.DataFrame,
    asset_column: str = "asset",
    target_column: str = "price",
    lag_periods: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Add lag features for target column.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        asset_column (str): Asset column name.
        target_column (str): Target column name.
        lag_periods (Optional[List[int]]): Lag periods.

    Returns:
        pd.DataFrame: Dataframe with lag features.
    """
    data = dataframe.copy()

    lag_periods = lag_periods or [1, 3, 7, 14]

    for lag in lag_periods:
        data[f"{target_column}_lag_{lag}"] = (
            data.groupby(asset_column)[target_column]
            .shift(lag)
        )

    return data


def add_rolling_features(
    dataframe: pd.DataFrame,
    asset_column: str = "asset",
    target_column: str = "price",
    rolling_windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Add rolling mean and rolling standard deviation features.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        asset_column (str): Asset column name.
        target_column (str): Target column name.
        rolling_windows (Optional[List[int]]): Rolling windows.

    Returns:
        pd.DataFrame: Dataframe with rolling features.
    """
    data = dataframe.copy()

    rolling_windows = rolling_windows or [7, 14, 30]

    for window in rolling_windows:
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

    return data


def encode_asset_column(
    dataframe: pd.DataFrame,
    asset_column: str = "asset",
) -> pd.DataFrame:
    """
    One-hot encode asset column.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        asset_column (str): Asset column name.

    Returns:
        pd.DataFrame: Encoded dataframe.
    """
    data = dataframe.copy()

    if asset_column in data.columns:
        data = pd.get_dummies(
            data,
            columns=[asset_column],
            drop_first=False,
            dtype=int,
        )

    return data


def prepare_features(
    dataframe: pd.DataFrame,
    date_column: str = "date",
    asset_column: str = "asset",
    target_column: str = "price",
    lag_periods: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
    encode_asset: bool = True,
) -> pd.DataFrame:
    """
    Prepare model-ready features for XGBoost.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        date_column (str): Date column name.
        asset_column (str): Asset column name.
        target_column (str): Target column name.
        lag_periods (Optional[List[int]]): Lag periods.
        rolling_windows (Optional[List[int]]): Rolling windows.
        encode_asset (bool): Whether to one-hot encode asset.

    Returns:
        pd.DataFrame: Feature-engineered dataframe.
    """
    data = clean_input_data(
        dataframe=dataframe,
        date_column=date_column,
        asset_column=asset_column,
        target_column=target_column,
    )

    data = add_time_features(data, date_column=date_column)
    data = add_return_features(
        data,
        asset_column=asset_column,
        target_column=target_column,
    )
    data = add_lag_features(
        data,
        asset_column=asset_column,
        target_column=target_column,
        lag_periods=lag_periods,
    )
    data = add_rolling_features(
        data,
        asset_column=asset_column,
        target_column=target_column,
        rolling_windows=rolling_windows,
    )

    data = data.replace([np.inf, -np.inf], np.nan)

    numeric_columns = data.select_dtypes(include=["number"]).columns

    for column in numeric_columns:
        if data[column].isna().sum() > 0:
            data[column] = data[column].fillna(data[column].median())

    data = data.fillna(0)

    if encode_asset:
        data = encode_asset_column(data, asset_column=asset_column)

    return data


def time_based_train_test_split(
    dataframe: pd.DataFrame,
    date_column: str = "date",
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into train and test sets using date order.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        date_column (str): Date column name.
        test_size (float): Test data ratio.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and test dataframes.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    data = dataframe.copy()
    data = data.sort_values(date_column).reset_index(drop=True)

    split_index = int(len(data) * (1 - test_size))

    train_data = data.iloc[:split_index].copy()
    test_data = data.iloc[split_index:].copy()

    if train_data.empty or test_data.empty:
        raise ValueError("Train or test data is empty after split.")

    return train_data, test_data


# ============================================================
# XGBoost Forecasting Class
# ============================================================

class XGBoostForecaster:
    """
    XGBoost model for financial asset price forecasting.
    """

    def __init__(
        self,
        date_column: str = "date",
        asset_column: str = "asset",
        target_column: str = "price",
        test_size: float = 0.2,
        lag_periods: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize XGBoostForecaster.

        Args:
            date_column (str): Date column name.
            asset_column (str): Asset column name.
            target_column (str): Target column name.
            test_size (float): Test data ratio.
            lag_periods (Optional[List[int]]): Lag feature periods.
            rolling_windows (Optional[List[int]]): Rolling feature windows.
            model_params (Optional[Dict[str, Any]]): XGBoost model parameters.
        """
        self.date_column = date_column
        self.asset_column = asset_column
        self.target_column = target_column
        self.test_size = test_size
        self.lag_periods = lag_periods or [1, 3, 7, 14]
        self.rolling_windows = rolling_windows or [7, 14, 30]

        self.model_params = model_params or {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "reg:squarederror",
            "random_state": 42,
        }

        self.model: Optional[XGBRegressor] = None
        self.feature_columns: List[str] = []
        self.metrics: Dict[str, float] = {}
        self.predictions: Optional[pd.DataFrame] = None

        ensure_directories()

    def get_feature_columns(self, dataframe: pd.DataFrame) -> List[str]:
        """
        Get model feature columns.

        Excludes:
        - date column
        - target column

        Args:
            dataframe (pd.DataFrame): Feature dataframe.

        Returns:
            List[str]: Feature column list.
        """
        excluded_columns = {
            self.date_column,
            self.target_column,
        }

        feature_columns = [
            column for column in dataframe.columns
            if column not in excluded_columns
            and pd.api.types.is_numeric_dtype(dataframe[column])
        ]

        return feature_columns

    def prepare_training_data(
        self,
        dataframe: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
        """
        Prepare train-test data for model training.

        Args:
            dataframe (pd.DataFrame): Input dataframe.

        Returns:
            Tuple containing:
            - X_train
            - X_test
            - y_train
            - y_test
            - train_data
            - test_data
        """
        feature_data = prepare_features(
            dataframe=dataframe,
            date_column=self.date_column,
            asset_column=self.asset_column,
            target_column=self.target_column,
            lag_periods=self.lag_periods,
            rolling_windows=self.rolling_windows,
            encode_asset=True,
        )

        train_data, test_data = time_based_train_test_split(
            dataframe=feature_data,
            date_column=self.date_column,
            test_size=self.test_size,
        )

        self.feature_columns = self.get_feature_columns(feature_data)

        X_train = train_data[self.feature_columns]
        X_test = test_data[self.feature_columns]

        y_train = train_data[self.target_column]
        y_test = test_data[self.target_column]

        return X_train, X_test, y_train, y_test, train_data, test_data

    def fit(
        self,
        dataframe: pd.DataFrame,
    ) -> "XGBoostForecaster":
        """
        Train XGBoost forecasting model.

        Args:
            dataframe (pd.DataFrame): Input dataframe.

        Returns:
            XGBoostForecaster: Fitted forecaster.
        """
        logger.info("Starting XGBoost model training.")

        X_train, X_test, y_train, y_test, train_data, test_data = (
            self.prepare_training_data(dataframe)
        )

        self.model = XGBRegressor(**self.model_params)

        self.model.fit(X_train, y_train)

        predictions = self.model.predict(X_test)
        predictions = np.clip(predictions, a_min=0, a_max=None)

        self.metrics = self.evaluate_predictions(y_test, predictions)

        self.predictions = test_data[
            [self.date_column, self.target_column]
        ].copy()

        asset_columns = [
            column for column in test_data.columns
            if column.startswith(f"{self.asset_column}_")
        ]

        if asset_columns:
            self.predictions["asset_encoded_reference"] = (
                test_data[asset_columns].idxmax(axis=1)
            )
            self.predictions["asset_encoded_reference"] = (
                self.predictions["asset_encoded_reference"]
                .str.replace(f"{self.asset_column}_", "", regex=False)
            )

        self.predictions["predicted_price"] = predictions
        self.predictions["absolute_error"] = (
            self.predictions[self.target_column]
            - self.predictions["predicted_price"]
        ).abs()

        logger.info("XGBoost model training completed.")
        logger.info("XGBoost metrics: %s", self.metrics)

        return self

    def evaluate_predictions(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate model predictions.

        Args:
            y_true (pd.Series): Actual values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        mape = calculate_mape(y_true, pd.Series(y_pred))

        try:
            r2 = float(r2_score(y_true, y_pred))
        except Exception:
            r2 = 0.0

        return {
            "mae": round(mae, 6),
            "mse": round(mse, 6),
            "rmse": round(rmse, 6),
            "mape": round(mape, 6),
            "r2_score": round(r2, 6),
            "feature_count": len(self.feature_columns),
        }

    def predict(
        self,
        dataframe: pd.DataFrame,
    ) -> np.ndarray:
        """
        Predict prices using trained XGBoost model.

        Args:
            dataframe (pd.DataFrame): Input dataframe.

        Returns:
            np.ndarray: Predicted prices.
        """
        if self.model is None:
            raise ValueError("XGBoost model is not trained yet.")

        feature_data = prepare_features(
            dataframe=dataframe,
            date_column=self.date_column,
            asset_column=self.asset_column,
            target_column=self.target_column,
            lag_periods=self.lag_periods,
            rolling_windows=self.rolling_windows,
            encode_asset=True,
        )

        for column in self.feature_columns:
            if column not in feature_data.columns:
                feature_data[column] = 0

        X = feature_data[self.feature_columns]

        predictions = self.model.predict(X)
        predictions = np.clip(predictions, a_min=0, a_max=None)

        return predictions

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get XGBoost feature importance.

        Returns:
            pd.DataFrame: Feature importance dataframe.
        """
        if self.model is None:
            raise ValueError("XGBoost model is not trained yet.")

        importance_dataframe = pd.DataFrame(
            {
                "feature": self.feature_columns,
                "importance": self.model.feature_importances_,
            }
        )

        importance_dataframe = importance_dataframe.sort_values(
            "importance",
            ascending=False,
        ).reset_index(drop=True)

        return importance_dataframe

    def save_feature_importance(
        self,
        file_path: Path = REPORTS_TABLES_DIR / "xgboost_feature_importance.csv",
    ) -> None:
        """
        Save feature importance as CSV.

        Args:
            file_path (Path): Output CSV path.
        """
        importance_dataframe = self.get_feature_importance()

        create_parent_directory(file_path)
        importance_dataframe.to_csv(file_path, index=False)

        logger.info("XGBoost feature importance saved at: %s", file_path)

    def save_model(
        self,
        file_path: Path = XGBOOST_MODEL_PATH,
    ) -> None:
        """
        Save trained XGBoost model artifact.

        Args:
            file_path (Path): Output model path.
        """
        if self.model is None:
            raise ValueError("No trained model available to save.")

        create_parent_directory(file_path)

        with file_path.open("wb") as file:
            pickle.dump(
                {
                    "model": self.model,
                    "feature_columns": self.feature_columns,
                    "metrics": self.metrics,
                    "date_column": self.date_column,
                    "asset_column": self.asset_column,
                    "target_column": self.target_column,
                    "test_size": self.test_size,
                    "lag_periods": self.lag_periods,
                    "rolling_windows": self.rolling_windows,
                    "model_params": self.model_params,
                },
                file,
            )

        logger.info("XGBoost model saved at: %s", file_path)

    def load_model(
        self,
        file_path: Path = XGBOOST_MODEL_PATH,
    ) -> "XGBoostForecaster":
        """
        Load saved XGBoost model artifact.

        Args:
            file_path (Path): Saved model path.

        Returns:
            XGBoostForecaster: Loaded forecaster.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"XGBoost model file not found: {file_path}")

        with file_path.open("rb") as file:
            saved_data = pickle.load(file)

        self.model = saved_data.get("model")
        self.feature_columns = saved_data.get("feature_columns", [])
        self.metrics = saved_data.get("metrics", {})

        self.date_column = saved_data.get("date_column", self.date_column)
        self.asset_column = saved_data.get("asset_column", self.asset_column)
        self.target_column = saved_data.get("target_column", self.target_column)
        self.test_size = saved_data.get("test_size", self.test_size)
        self.lag_periods = saved_data.get("lag_periods", self.lag_periods)
        self.rolling_windows = saved_data.get(
            "rolling_windows",
            self.rolling_windows,
        )
        self.model_params = saved_data.get("model_params", self.model_params)

        logger.info("XGBoost model loaded from: %s", file_path)

        return self

    def save_feature_columns(
        self,
        file_path: Path = XGBOOST_FEATURE_COLUMNS_PATH,
    ) -> None:
        """
        Save feature columns as JSON.

        Args:
            file_path (Path): Output JSON path.
        """
        create_parent_directory(file_path)

        with file_path.open("w", encoding="utf-8") as file:
            json.dump(self.feature_columns, file, indent=4)

        logger.info("XGBoost feature columns saved at: %s", file_path)

    def save_metrics_json(
        self,
        file_path: Path = XGBOOST_METRICS_JSON_PATH,
    ) -> None:
        """
        Save metrics as JSON.

        Args:
            file_path (Path): Output JSON path.
        """
        create_parent_directory(file_path)

        with file_path.open("w", encoding="utf-8") as file:
            json.dump(self.metrics, file, indent=4)

        logger.info("XGBoost metrics JSON saved at: %s", file_path)

    def save_metrics_csv(
        self,
        file_path: Path = XGBOOST_METRICS_CSV_PATH,
    ) -> None:
        """
        Save metrics as CSV.

        Args:
            file_path (Path): Output CSV path.
        """
        create_parent_directory(file_path)

        metrics_dataframe = pd.DataFrame([self.metrics])
        metrics_dataframe.to_csv(file_path, index=False)

        logger.info("XGBoost metrics CSV saved at: %s", file_path)

    def save_predictions(
        self,
        file_path: Path = XGBOOST_PREDICTIONS_PATH,
    ) -> None:
        """
        Save test predictions as CSV.

        Args:
            file_path (Path): Output CSV path.
        """
        if self.predictions is None or self.predictions.empty:
            raise ValueError("No predictions available to save.")

        create_parent_directory(file_path)

        self.predictions.to_csv(file_path, index=False)

        logger.info("XGBoost predictions saved at: %s", file_path)

    def save_all_artifacts(self) -> None:
        """
        Save model, metrics, features, predictions, and feature importance.
        """
        self.save_model()
        self.save_feature_columns()
        self.save_metrics_json()
        self.save_metrics_csv()
        self.save_predictions()
        self.save_feature_importance()

    def recursive_forecast_asset(
        self,
        dataframe: pd.DataFrame,
        asset_name: str,
        forecast_steps: int = 30,
    ) -> pd.DataFrame:
        """
        Generate recursive future forecast for one asset.

        The model predicts one future row at a time and feeds the prediction
        back into the historical data for the next step.

        Args:
            dataframe (pd.DataFrame): Historical dataframe.
            asset_name (str): Asset name.
            forecast_steps (int): Number of future periods.

        Returns:
            pd.DataFrame: Forecast dataframe.
        """
        if self.model is None:
            raise ValueError("XGBoost model is not trained yet.")

        history = clean_input_data(
            dataframe=dataframe,
            date_column=self.date_column,
            asset_column=self.asset_column,
            target_column=self.target_column,
        )

        asset_history = history[
            history[self.asset_column] == asset_name
        ].copy()

        if asset_history.empty:
            raise ValueError(f"No historical data found for asset: {asset_name}")

        asset_history = asset_history.sort_values(self.date_column)

        future_rows = []

        last_date = asset_history[self.date_column].max()

        for step in range(1, forecast_steps + 1):
            future_date = last_date + pd.Timedelta(days=step)

            temporary_data = pd.concat(
                [
                    asset_history,
                    pd.DataFrame(
                        {
                            self.date_column: [future_date],
                            self.asset_column: [asset_name],
                            self.target_column: [asset_history[self.target_column].iloc[-1]],
                        }
                    ),
                ],
                ignore_index=True,
            )

            feature_data = prepare_features(
                dataframe=temporary_data,
                date_column=self.date_column,
                asset_column=self.asset_column,
                target_column=self.target_column,
                lag_periods=self.lag_periods,
                rolling_windows=self.rolling_windows,
                encode_asset=True,
            )

            latest_row = feature_data.tail(1).copy()

            for column in self.feature_columns:
                if column not in latest_row.columns:
                    latest_row[column] = 0

            X_future = latest_row[self.feature_columns]

            predicted_price = float(self.model.predict(X_future)[0])
            predicted_price = max(predicted_price, 0.0)

            future_rows.append(
                {
                    self.date_column: future_date,
                    self.asset_column: asset_name,
                    "forecasted_price": predicted_price,
                }
            )

            asset_history = pd.concat(
                [
                    asset_history,
                    pd.DataFrame(
                        {
                            self.date_column: [future_date],
                            self.asset_column: [asset_name],
                            self.target_column: [predicted_price],
                        }
                    ),
                ],
                ignore_index=True,
            )

        return pd.DataFrame(future_rows)

    def forecast_all_assets(
        self,
        dataframe: pd.DataFrame,
        forecast_steps: int = 30,
    ) -> pd.DataFrame:
        """
        Forecast future prices for all assets.

        Args:
            dataframe (pd.DataFrame): Historical dataframe.
            forecast_steps (int): Number of future periods.

        Returns:
            pd.DataFrame: Combined forecast dataframe.
        """
        data = clean_input_data(
            dataframe=dataframe,
            date_column=self.date_column,
            asset_column=self.asset_column,
            target_column=self.target_column,
        )

        assets = sorted(data[self.asset_column].unique().tolist())

        forecast_frames = []

        for asset_name in assets:
            try:
                asset_forecast = self.recursive_forecast_asset(
                    dataframe=data,
                    asset_name=asset_name,
                    forecast_steps=forecast_steps,
                )
                forecast_frames.append(asset_forecast)

            except Exception as error:
                logger.error(
                    "XGBoost forecast failed for asset %s: %s",
                    asset_name,
                    error,
                    exc_info=True,
                )

        if not forecast_frames:
            raise ValueError("No XGBoost forecasts were generated.")

        forecast_dataframe = pd.concat(forecast_frames, ignore_index=True)

        return forecast_dataframe

    def save_forecast_results(
        self,
        forecast_dataframe: pd.DataFrame,
        file_path: Path = XGBOOST_FORECAST_PATH,
    ) -> None:
        """
        Save forecast results as CSV.

        Args:
            forecast_dataframe (pd.DataFrame): Forecast dataframe.
            file_path (Path): Output CSV path.
        """
        if forecast_dataframe.empty:
            raise ValueError("Forecast dataframe is empty.")

        create_parent_directory(file_path)

        forecast_dataframe.to_csv(file_path, index=False)

        logger.info("XGBoost forecast results saved at: %s", file_path)

    def train_forecast_save(
        self,
        dataframe: pd.DataFrame,
        forecast_steps: int = 30,
    ) -> Dict[str, Any]:
        """
        Run complete XGBoost workflow.

        Steps:
        1. Train model
        2. Evaluate model
        3. Save artifacts
        4. Generate future forecasts
        5. Save forecast results

        Args:
            dataframe (pd.DataFrame): Input dataframe.
            forecast_steps (int): Number of future forecast periods.

        Returns:
            Dict[str, Any]: Workflow summary.
        """
        self.fit(dataframe)
        self.save_all_artifacts()

        forecast_dataframe = self.forecast_all_assets(
            dataframe=dataframe,
            forecast_steps=forecast_steps,
        )

        self.save_forecast_results(forecast_dataframe)

        summary = {
            "status": "success",
            "model_type": "XGBoost Regressor",
            "target_column": self.target_column,
            "feature_count": len(self.feature_columns),
            "forecast_steps": forecast_steps,
            "forecast_rows": len(forecast_dataframe),
            "metrics": self.metrics,
            "model_path": str(XGBOOST_MODEL_PATH),
            "metrics_path": str(XGBOOST_METRICS_JSON_PATH),
            "predictions_path": str(XGBOOST_PREDICTIONS_PATH),
            "forecast_path": str(XGBOOST_FORECAST_PATH),
        }

        logger.info("XGBoost workflow summary: %s", summary)

        return summary


# ============================================================
# Convenience Functions
# ============================================================

def train_xgboost_model(
    dataframe: pd.DataFrame,
    forecast_steps: int = 30,
    model_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Train XGBoost model and generate forecasts.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        forecast_steps (int): Number of forecast periods.
        model_params (Optional[Dict[str, Any]]): Custom model parameters.

    Returns:
        Dict[str, Any]: Training workflow summary.
    """
    forecaster = XGBoostForecaster(model_params=model_params)

    return forecaster.train_forecast_save(
        dataframe=dataframe,
        forecast_steps=forecast_steps,
    )


def load_xgboost_forecaster(
    model_path: Path = XGBOOST_MODEL_PATH,
) -> XGBoostForecaster:
    """
    Load saved XGBoost forecaster.

    Args:
        model_path (Path): Saved model path.

    Returns:
        XGBoostForecaster: Loaded forecaster.
    """
    forecaster = XGBoostForecaster()
    forecaster.load_model(model_path)

    return forecaster


def forecast_with_saved_xgboost(
    dataframe: pd.DataFrame,
    forecast_steps: int = 30,
    model_path: Path = XGBOOST_MODEL_PATH,
    output_path: Path = XGBOOST_FORECAST_PATH,
) -> pd.DataFrame:
    """
    Generate forecasts using saved XGBoost model.

    Args:
        dataframe (pd.DataFrame): Historical dataframe.
        forecast_steps (int): Number of forecast periods.
        model_path (Path): Saved model path.
        output_path (Path): Output forecast path.

    Returns:
        pd.DataFrame: Forecast dataframe.
    """
    forecaster = load_xgboost_forecaster(model_path)

    forecast_dataframe = forecaster.forecast_all_assets(
        dataframe=dataframe,
        forecast_steps=forecast_steps,
    )

    forecaster.save_forecast_results(
        forecast_dataframe=forecast_dataframe,
        file_path=output_path,
    )

    return forecast_dataframe


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

    xgb_forecaster = XGBoostForecaster(
        date_column="date",
        asset_column="asset",
        target_column="price",
        test_size=0.2,
        lag_periods=[1, 3, 7, 14],
        rolling_windows=[7, 14, 30],
    )

    workflow_summary = xgb_forecaster.train_forecast_save(
        dataframe=data,
        forecast_steps=30,
    )

    print("XGBoost workflow completed successfully.")
    print(workflow_summary)