"""
Prophet forecasting model for the Financial Portfolio Forecasting
and Anomaly Detection project.

This module handles:
- Asset-wise Prophet model training
- Future price forecasting
- Forecast evaluation
- Model artifact saving and loading
- Metrics and forecast output saving

Dataset expected columns:
- date
- asset
- price

Prophet expects:
- ds = date column
- y = target column
"""

import json
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from prophet import Prophet
except ImportError as exc:
    raise ImportError(
        "prophet is required for Prophet forecasting. "
        "Install it using: pip install prophet"
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

PROPHET_MODEL_PATH = MODELS_DIR / "prophet_models.pkl"
PROPHET_METRICS_JSON_PATH = REPORTS_TABLES_DIR / "prophet_metrics.json"
PROPHET_METRICS_CSV_PATH = REPORTS_TABLES_DIR / "prophet_metrics.csv"
PROPHET_FORECAST_PATH = OUTPUTS_DIR / "prophet_forecast_results.csv"
PROPHET_TEST_PREDICTIONS_PATH = OUTPUTS_DIR / "prophet_test_predictions.csv"


# ============================================================
# Helper Functions
# ============================================================

def ensure_directories() -> None:
    """
    Create required folders for model artifacts, reports, and outputs.
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


def prepare_prophet_dataframe(
    dataframe: pd.DataFrame,
    asset_name: str,
    date_column: str = "date",
    asset_column: str = "asset",
    target_column: str = "price",
) -> pd.DataFrame:
    """
    Prepare one asset's data for Prophet.

    Prophet requires:
    - ds column for date
    - y column for target value

    Duplicate dates are averaged.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        asset_name (str): Asset name.
        date_column (str): Date column name.
        asset_column (str): Asset column name.
        target_column (str): Target column name.

    Returns:
        pd.DataFrame: Prophet-ready dataframe.
    """
    asset_data = dataframe[dataframe[asset_column] == asset_name].copy()

    if asset_data.empty:
        raise ValueError(f"No data found for asset: {asset_name}")

    prophet_data = (
        asset_data.groupby(date_column, as_index=False)
        .agg(y=(target_column, "mean"))
        .rename(columns={date_column: "ds"})
        .sort_values("ds")
        .reset_index(drop=True)
    )

    prophet_data["ds"] = pd.to_datetime(prophet_data["ds"], errors="coerce")
    prophet_data["y"] = pd.to_numeric(prophet_data["y"], errors="coerce")

    prophet_data = prophet_data.dropna(subset=["ds", "y"])
    prophet_data = prophet_data[prophet_data["y"] >= 0]

    return prophet_data


def time_based_train_test_split(
    dataframe: pd.DataFrame,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split Prophet dataframe into train and test sets using time order.

    Args:
        dataframe (pd.DataFrame): Prophet-ready dataframe.
        test_size (float): Test data ratio.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and test dataframes.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    data = dataframe.sort_values("ds").reset_index(drop=True)

    split_index = int(len(data) * (1 - test_size))

    train_data = data.iloc[:split_index].copy()
    test_data = data.iloc[split_index:].copy()

    if train_data.empty or test_data.empty:
        raise ValueError("Train or test data is empty after split.")

    return train_data, test_data


# ============================================================
# Prophet Forecaster Class
# ============================================================

class ProphetForecaster:
    """
    Asset-wise Prophet forecaster.
    """

    def __init__(
        self,
        date_column: str = "date",
        asset_column: str = "asset",
        target_column: str = "price",
        test_size: float = 0.2,
        frequency: str = "D",
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        seasonality_mode: str = "additive",
        changepoint_prior_scale: float = 0.05,
    ) -> None:
        """
        Initialize ProphetForecaster.

        Args:
            date_column (str): Date column name.
            asset_column (str): Asset column name.
            target_column (str): Target column name.
            test_size (float): Test data ratio.
            frequency (str): Forecast frequency.
            yearly_seasonality (bool): Enable yearly seasonality.
            weekly_seasonality (bool): Enable weekly seasonality.
            daily_seasonality (bool): Enable daily seasonality.
            seasonality_mode (str): additive or multiplicative.
            changepoint_prior_scale (float): Prophet trend flexibility parameter.
        """
        self.date_column = date_column
        self.asset_column = asset_column
        self.target_column = target_column
        self.test_size = test_size
        self.frequency = frequency
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale

        self.models: Dict[str, Any] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}
        self.test_predictions: Optional[pd.DataFrame] = None
        self.forecast_results: Optional[pd.DataFrame] = None

        ensure_directories()

    def build_model(self) -> Prophet:
        """
        Build a Prophet model using configured parameters.

        Returns:
            Prophet: Prophet model instance.
        """
        model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale,
        )

        return model

    def fit_asset_model(
        self,
        train_data: pd.DataFrame,
        asset_name: str,
    ) -> Prophet:
        """
        Train Prophet model for one asset.

        Args:
            train_data (pd.DataFrame): Prophet-ready training data.
            asset_name (str): Asset name.

        Returns:
            Prophet: Fitted Prophet model.
        """
        logger.info("Training Prophet model for asset: %s", asset_name)

        model = self.build_model()
        model.fit(train_data)

        self.models[asset_name] = model

        logger.info("Prophet model trained successfully for asset: %s", asset_name)

        return model

    def predict_test_period(
        self,
        model: Prophet,
        test_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Predict values for the test period.

        Args:
            model (Prophet): Fitted Prophet model.
            test_data (pd.DataFrame): Test dataframe with ds and y.

        Returns:
            pd.DataFrame: Forecast dataframe for test period.
        """
        future = test_data[["ds"]].copy()
        forecast = model.predict(future)

        predictions = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()

        predictions["yhat"] = predictions["yhat"].clip(lower=0)
        predictions["yhat_lower"] = predictions["yhat_lower"].clip(lower=0)
        predictions["yhat_upper"] = predictions["yhat_upper"].clip(lower=0)

        return predictions

    def evaluate_asset_model(
        self,
        test_data: pd.DataFrame,
        predictions: pd.DataFrame,
        asset_name: str,
        train_records: int,
    ) -> Dict[str, float]:
        """
        Evaluate Prophet model predictions for one asset.

        Args:
            test_data (pd.DataFrame): Actual test dataframe.
            predictions (pd.DataFrame): Prediction dataframe.
            asset_name (str): Asset name.
            train_records (int): Number of training records.

        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        merged = test_data.merge(
            predictions[["ds", "yhat"]],
            on="ds",
            how="inner",
        )

        if merged.empty:
            raise ValueError(f"No matching prediction dates for asset: {asset_name}")

        y_true = merged["y"]
        y_pred = merged["yhat"]

        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = calculate_rmse(y_true, y_pred)
        mape = calculate_mape(y_true, y_pred)

        try:
            r2 = float(r2_score(y_true, y_pred))
        except Exception:
            r2 = 0.0

        metrics = {
            "mae": round(mae, 6),
            "mse": round(mse, 6),
            "rmse": round(rmse, 6),
            "mape": round(mape, 6),
            "r2_score": round(r2, 6),
            "train_records": int(train_records),
            "test_records": int(len(test_data)),
        }

        self.metrics[asset_name] = metrics

        logger.info("Prophet metrics for %s: %s", asset_name, metrics)

        return metrics

    def fit(
        self,
        dataframe: pd.DataFrame,
        assets: Optional[List[str]] = None,
    ) -> "ProphetForecaster":
        """
        Train Prophet models for all or selected assets.

        Args:
            dataframe (pd.DataFrame): Input dataframe.
            assets (Optional[List[str]]): Selected assets.

        Returns:
            ProphetForecaster: Fitted forecaster.
        """
        logger.info("Starting Prophet training process.")

        data = clean_input_data(
            dataframe=dataframe,
            date_column=self.date_column,
            asset_column=self.asset_column,
            target_column=self.target_column,
        )

        if assets is None:
            assets = sorted(data[self.asset_column].unique().tolist())

        all_test_predictions = []

        for asset_name in assets:
            try:
                prophet_data = prepare_prophet_dataframe(
                    dataframe=data,
                    asset_name=asset_name,
                    date_column=self.date_column,
                    asset_column=self.asset_column,
                    target_column=self.target_column,
                )

                if len(prophet_data) < 10:
                    logger.warning(
                        "Skipping asset %s because it has fewer than 10 records.",
                        asset_name,
                    )
                    continue

                train_data, test_data = time_based_train_test_split(
                    dataframe=prophet_data,
                    test_size=self.test_size,
                )

                model = self.fit_asset_model(
                    train_data=train_data,
                    asset_name=asset_name,
                )

                predictions = self.predict_test_period(
                    model=model,
                    test_data=test_data,
                )

                self.evaluate_asset_model(
                    test_data=test_data,
                    predictions=predictions,
                    asset_name=asset_name,
                    train_records=len(train_data),
                )

                test_output = test_data.merge(
                    predictions,
                    on="ds",
                    how="left",
                )

                test_output[self.asset_column] = asset_name
                test_output = test_output.rename(
                    columns={
                        "ds": self.date_column,
                        "y": "actual_price",
                        "yhat": "predicted_price",
                    }
                )

                test_output["absolute_error"] = (
                    test_output["actual_price"]
                    - test_output["predicted_price"]
                ).abs()

                all_test_predictions.append(test_output)

            except Exception as error:
                logger.error(
                    "Failed to train Prophet model for asset %s: %s",
                    asset_name,
                    error,
                    exc_info=True,
                )

        if all_test_predictions:
            self.test_predictions = pd.concat(
                all_test_predictions,
                ignore_index=True,
            )

        logger.info("Prophet training process completed.")

        return self

    def forecast_asset(
        self,
        asset_name: str,
        periods: int = 30,
    ) -> pd.DataFrame:
        """
        Forecast future prices for one asset.

        Args:
            asset_name (str): Asset name.
            periods (int): Number of future periods.

        Returns:
            pd.DataFrame: Forecast dataframe.
        """
        if asset_name not in self.models:
            raise ValueError(
                f"No trained Prophet model found for asset '{asset_name}'."
            )

        model = self.models[asset_name]

        future = model.make_future_dataframe(
            periods=periods,
            freq=self.frequency,
            include_history=False,
        )

        forecast = model.predict(future)

        forecast_dataframe = forecast[
            ["ds", "yhat", "yhat_lower", "yhat_upper"]
        ].copy()

        forecast_dataframe = forecast_dataframe.rename(
            columns={
                "ds": self.date_column,
                "yhat": "forecasted_price",
                "yhat_lower": "forecast_lower_bound",
                "yhat_upper": "forecast_upper_bound",
            }
        )

        forecast_dataframe[self.asset_column] = asset_name

        forecast_dataframe["forecasted_price"] = forecast_dataframe[
            "forecasted_price"
        ].clip(lower=0)

        forecast_dataframe["forecast_lower_bound"] = forecast_dataframe[
            "forecast_lower_bound"
        ].clip(lower=0)

        forecast_dataframe["forecast_upper_bound"] = forecast_dataframe[
            "forecast_upper_bound"
        ].clip(lower=0)

        forecast_dataframe = forecast_dataframe[
            [
                self.date_column,
                self.asset_column,
                "forecasted_price",
                "forecast_lower_bound",
                "forecast_upper_bound",
            ]
        ]

        return forecast_dataframe

    def forecast_all(
        self,
        periods: int = 30,
    ) -> pd.DataFrame:
        """
        Forecast future prices for all trained assets.

        Args:
            periods (int): Number of future periods.

        Returns:
            pd.DataFrame: Combined forecast dataframe.
        """
        if not self.models:
            raise ValueError("No Prophet models are trained yet.")

        all_forecasts = []

        for asset_name in self.models.keys():
            try:
                forecast_dataframe = self.forecast_asset(
                    asset_name=asset_name,
                    periods=periods,
                )
                all_forecasts.append(forecast_dataframe)

            except Exception as error:
                logger.error(
                    "Forecast failed for asset %s: %s",
                    asset_name,
                    error,
                    exc_info=True,
                )

        if not all_forecasts:
            raise ValueError("No Prophet forecasts were generated.")

        self.forecast_results = pd.concat(
            all_forecasts,
            ignore_index=True,
        )

        return self.forecast_results

    def save_models(
        self,
        file_path: Path = PROPHET_MODEL_PATH,
    ) -> None:
        """
        Save trained Prophet models.

        Args:
            file_path (Path): Output model path.
        """
        if not self.models:
            raise ValueError("No trained Prophet models available to save.")

        create_parent_directory(file_path)

        with file_path.open("wb") as file:
            pickle.dump(
                {
                    "date_column": self.date_column,
                    "asset_column": self.asset_column,
                    "target_column": self.target_column,
                    "test_size": self.test_size,
                    "frequency": self.frequency,
                    "yearly_seasonality": self.yearly_seasonality,
                    "weekly_seasonality": self.weekly_seasonality,
                    "daily_seasonality": self.daily_seasonality,
                    "seasonality_mode": self.seasonality_mode,
                    "changepoint_prior_scale": self.changepoint_prior_scale,
                    "models": self.models,
                    "metrics": self.metrics,
                },
                file,
            )

        logger.info("Prophet models saved at: %s", file_path)

    def load_models(
        self,
        file_path: Path = PROPHET_MODEL_PATH,
    ) -> "ProphetForecaster":
        """
        Load saved Prophet models.

        Args:
            file_path (Path): Saved model path.

        Returns:
            ProphetForecaster: Loaded forecaster.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Prophet model file not found: {file_path}")

        with file_path.open("rb") as file:
            saved_data = pickle.load(file)

        self.date_column = saved_data.get("date_column", self.date_column)
        self.asset_column = saved_data.get("asset_column", self.asset_column)
        self.target_column = saved_data.get("target_column", self.target_column)
        self.test_size = saved_data.get("test_size", self.test_size)
        self.frequency = saved_data.get("frequency", self.frequency)
        self.yearly_seasonality = saved_data.get(
            "yearly_seasonality",
            self.yearly_seasonality,
        )
        self.weekly_seasonality = saved_data.get(
            "weekly_seasonality",
            self.weekly_seasonality,
        )
        self.daily_seasonality = saved_data.get(
            "daily_seasonality",
            self.daily_seasonality,
        )
        self.seasonality_mode = saved_data.get(
            "seasonality_mode",
            self.seasonality_mode,
        )
        self.changepoint_prior_scale = saved_data.get(
            "changepoint_prior_scale",
            self.changepoint_prior_scale,
        )

        self.models = saved_data.get("models", {})
        self.metrics = saved_data.get("metrics", {})

        logger.info("Prophet models loaded from: %s", file_path)

        return self

    def save_metrics_json(
        self,
        file_path: Path = PROPHET_METRICS_JSON_PATH,
    ) -> None:
        """
        Save Prophet metrics as JSON.

        Args:
            file_path (Path): Output JSON path.
        """
        if not self.metrics:
            raise ValueError("No Prophet metrics available to save.")

        create_parent_directory(file_path)

        with file_path.open("w", encoding="utf-8") as file:
            json.dump(self.metrics, file, indent=4)

        logger.info("Prophet metrics JSON saved at: %s", file_path)

    def get_metrics_dataframe(self) -> pd.DataFrame:
        """
        Convert metrics dictionary into a dataframe.

        Returns:
            pd.DataFrame: Metrics dataframe.
        """
        if not self.metrics:
            return pd.DataFrame()

        metrics_dataframe = (
            pd.DataFrame(self.metrics)
            .T
            .reset_index()
            .rename(columns={"index": self.asset_column})
        )

        return metrics_dataframe

    def save_metrics_csv(
        self,
        file_path: Path = PROPHET_METRICS_CSV_PATH,
    ) -> None:
        """
        Save Prophet metrics as CSV.

        Args:
            file_path (Path): Output CSV path.
        """
        metrics_dataframe = self.get_metrics_dataframe()

        if metrics_dataframe.empty:
            raise ValueError("No Prophet metrics available to save.")

        create_parent_directory(file_path)

        metrics_dataframe.to_csv(file_path, index=False)

        logger.info("Prophet metrics CSV saved at: %s", file_path)

    def save_test_predictions(
        self,
        file_path: Path = PROPHET_TEST_PREDICTIONS_PATH,
    ) -> None:
        """
        Save Prophet test predictions as CSV.

        Args:
            file_path (Path): Output CSV path.
        """
        if self.test_predictions is None or self.test_predictions.empty:
            raise ValueError("No Prophet test predictions available to save.")

        create_parent_directory(file_path)

        self.test_predictions.to_csv(file_path, index=False)

        logger.info("Prophet test predictions saved at: %s", file_path)

    def save_forecasts(
        self,
        dataframe: Optional[pd.DataFrame] = None,
        file_path: Path = PROPHET_FORECAST_PATH,
    ) -> None:
        """
        Save Prophet future forecasts as CSV.

        Args:
            dataframe (Optional[pd.DataFrame]): Forecast dataframe.
            file_path (Path): Output CSV path.
        """
        forecast_dataframe = dataframe if dataframe is not None else self.forecast_results

        if forecast_dataframe is None or forecast_dataframe.empty:
            raise ValueError("No Prophet forecast results available to save.")

        create_parent_directory(file_path)

        forecast_dataframe.to_csv(file_path, index=False)

        logger.info("Prophet forecast results saved at: %s", file_path)

    def save_all_artifacts(self) -> None:
        """
        Save Prophet models, metrics, and test predictions.
        """
        self.save_models()
        self.save_metrics_json()
        self.save_metrics_csv()

        if self.test_predictions is not None and not self.test_predictions.empty:
            self.save_test_predictions()

    def train_forecast_save(
        self,
        dataframe: pd.DataFrame,
        forecast_periods: int = 30,
        assets: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run complete Prophet workflow.

        Steps:
        1. Train Prophet models
        2. Evaluate test predictions
        3. Generate future forecasts
        4. Save models, metrics, predictions, and forecasts

        Args:
            dataframe (pd.DataFrame): Input dataframe.
            forecast_periods (int): Number of future periods.
            assets (Optional[List[str]]): Optional selected assets.

        Returns:
            Dict[str, Any]: Workflow summary.
        """
        self.fit(dataframe=dataframe, assets=assets)

        forecast_dataframe = self.forecast_all(periods=forecast_periods)

        self.save_all_artifacts()
        self.save_forecasts(forecast_dataframe)

        summary = {
            "status": "success",
            "model_type": "Prophet",
            "target_column": self.target_column,
            "trained_assets": list(self.models.keys()),
            "asset_count": len(self.models),
            "forecast_periods": forecast_periods,
            "forecast_rows": len(forecast_dataframe),
            "metrics": self.metrics,
            "model_path": str(PROPHET_MODEL_PATH),
            "metrics_path": str(PROPHET_METRICS_JSON_PATH),
            "forecast_path": str(PROPHET_FORECAST_PATH),
        }

        logger.info("Prophet workflow summary: %s", summary)

        return summary


# ============================================================
# Convenience Functions
# ============================================================

def train_prophet_models(
    dataframe: pd.DataFrame,
    forecast_periods: int = 30,
    assets: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Train Prophet models and generate forecasts.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        forecast_periods (int): Forecast periods.
        assets (Optional[List[str]]): Optional selected assets.

    Returns:
        Dict[str, Any]: Workflow summary.
    """
    forecaster = ProphetForecaster()

    return forecaster.train_forecast_save(
        dataframe=dataframe,
        forecast_periods=forecast_periods,
        assets=assets,
    )


def load_prophet_forecaster(
    model_path: Path = PROPHET_MODEL_PATH,
) -> ProphetForecaster:
    """
    Load saved Prophet forecaster.

    Args:
        model_path (Path): Saved model path.

    Returns:
        ProphetForecaster: Loaded forecaster.
    """
    forecaster = ProphetForecaster()
    forecaster.load_models(model_path)

    return forecaster


def forecast_with_saved_prophet(
    forecast_periods: int = 30,
    model_path: Path = PROPHET_MODEL_PATH,
    output_path: Path = PROPHET_FORECAST_PATH,
) -> pd.DataFrame:
    """
    Generate forecasts using saved Prophet models.

    Args:
        forecast_periods (int): Number of future periods.
        model_path (Path): Saved Prophet model path.
        output_path (Path): Output forecast path.

    Returns:
        pd.DataFrame: Forecast dataframe.
    """
    forecaster = load_prophet_forecaster(model_path)

    forecast_dataframe = forecaster.forecast_all(periods=forecast_periods)

    forecaster.save_forecasts(
        dataframe=forecast_dataframe,
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

    prophet_forecaster = ProphetForecaster(
        date_column="date",
        asset_column="asset",
        target_column="price",
        test_size=0.2,
        frequency="D",
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
    )

    workflow_summary = prophet_forecaster.train_forecast_save(
        dataframe=data,
        forecast_periods=30,
    )

    print("Prophet workflow completed successfully.")
    print(workflow_summary)