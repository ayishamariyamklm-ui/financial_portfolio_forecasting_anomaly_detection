"""
ARIMA forecasting model for the Financial Portfolio Forecasting
and Anomaly Detection project.

This module handles:
- Asset-wise ARIMA model training
- Price forecasting
- Model evaluation
- Forecast result saving
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    try:
        from statsmodels.tsa.arima_model import ARIMA
    except ImportError as exc:
        raise ImportError(
            "statsmodels is required for ARIMA modeling. "
            "Install it using: pip install statsmodels"
        ) from exc

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.monitoring.logger import get_project_logger
from src.utils.paths import create_parent_directory, get_path


warnings.filterwarnings("ignore")

logger = get_project_logger()


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

ARIMA_MODEL_PATH = MODELS_DIR / "arima_models.pkl"
ARIMA_METRICS_PATH = REPORTS_TABLES_DIR / "arima_metrics.json"
ARIMA_FORECAST_PATH = OUTPUTS_DIR / "arima_forecast_results.csv"


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
        float: MAPE value.
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


def clean_time_series_data(
    dataframe: pd.DataFrame,
    date_column: str = "date",
    asset_column: str = "asset",
    target_column: str = "price",
) -> pd.DataFrame:
    """
    Clean and prepare dataframe for ARIMA modeling.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        date_column (str): Date column.
        asset_column (str): Asset column.
        target_column (str): Target column.

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


def prepare_asset_series(
    dataframe: pd.DataFrame,
    asset_name: str,
    date_column: str = "date",
    asset_column: str = "asset",
    target_column: str = "price",
    frequency: str = "D",
) -> pd.Series:
    """
    Prepare a single asset price series for ARIMA.

    If duplicate dates exist for an asset, prices are averaged by date.
    Missing dates are filled using forward fill and backward fill.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        asset_name (str): Asset name.
        date_column (str): Date column.
        asset_column (str): Asset column.
        target_column (str): Target column.
        frequency (str): Time series frequency.

    Returns:
        pd.Series: Asset price time series.
    """
    asset_data = dataframe[dataframe[asset_column] == asset_name].copy()

    if asset_data.empty:
        raise ValueError(f"No data found for asset: {asset_name}")

    asset_series = (
        asset_data.groupby(date_column)[target_column]
        .mean()
        .sort_index()
    )

    asset_series.index = pd.to_datetime(asset_series.index)

    asset_series = asset_series.asfreq(frequency)

    asset_series = asset_series.ffill().bfill()

    return asset_series


def time_based_train_test_split(
    series: pd.Series,
    test_size: float = 0.2,
) -> Tuple[pd.Series, pd.Series]:
    """
    Split a time series into train and test sets.

    Args:
        series (pd.Series): Input time series.
        test_size (float): Test size ratio.

    Returns:
        Tuple[pd.Series, pd.Series]: Train and test series.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    split_index = int(len(series) * (1 - test_size))

    train_series = series.iloc[:split_index]
    test_series = series.iloc[split_index:]

    if train_series.empty or test_series.empty:
        raise ValueError("Train or test series is empty after splitting.")

    return train_series, test_series


# ============================================================
# ARIMA Forecaster Class
# ============================================================

class ARIMAForecaster:
    """
    Asset-wise ARIMA forecaster.
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (5, 1, 0),
        date_column: str = "date",
        asset_column: str = "asset",
        target_column: str = "price",
        frequency: str = "D",
        test_size: float = 0.2,
    ) -> None:
        """
        Initialize ARIMAForecaster.

        Args:
            order (Tuple[int, int, int]): ARIMA order as (p, d, q).
            date_column (str): Date column name.
            asset_column (str): Asset column name.
            target_column (str): Target column name.
            frequency (str): Time series frequency.
            test_size (float): Test data ratio.
        """
        self.order = order
        self.date_column = date_column
        self.asset_column = asset_column
        self.target_column = target_column
        self.frequency = frequency
        self.test_size = test_size

        self.models: Dict[str, Any] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}
        self.forecast_results: Optional[pd.DataFrame] = None

        ensure_directories()

    def fit_asset_model(
        self,
        series: pd.Series,
        asset_name: str,
    ) -> Any:
        """
        Fit ARIMA model for one asset.

        Args:
            series (pd.Series): Asset price series.
            asset_name (str): Asset name.

        Returns:
            Any: Fitted ARIMA result object.
        """
        logger.info("Training ARIMA model for asset: %s", asset_name)

        model = ARIMA(series, order=self.order)
        fitted_model = model.fit()

        self.models[asset_name] = fitted_model

        logger.info("ARIMA model trained successfully for asset: %s", asset_name)

        return fitted_model

    def evaluate_asset_model(
        self,
        fitted_model: Any,
        train_series: pd.Series,
        test_series: pd.Series,
        asset_name: str,
    ) -> Dict[str, float]:
        """
        Evaluate ARIMA model on test data.

        Args:
            fitted_model (Any): Fitted ARIMA model.
            train_series (pd.Series): Training series.
            test_series (pd.Series): Test series.
            asset_name (str): Asset name.

        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        forecast_steps = len(test_series)

        predictions = fitted_model.forecast(steps=forecast_steps)

        predictions = pd.Series(
            predictions.values,
            index=test_series.index,
            name="predicted_price",
        )

        mae = float(mean_absolute_error(test_series, predictions))
        mse = float(mean_squared_error(test_series, predictions))
        rmse = calculate_rmse(test_series, predictions)
        mape = calculate_mape(test_series, predictions)

        try:
            r2 = float(r2_score(test_series, predictions))
        except Exception:
            r2 = 0.0

        metrics = {
            "mae": round(mae, 6),
            "mse": round(mse, 6),
            "rmse": round(rmse, 6),
            "mape": round(mape, 6),
            "r2_score": round(r2, 6),
            "train_records": int(len(train_series)),
            "test_records": int(len(test_series)),
        }

        self.metrics[asset_name] = metrics

        logger.info("ARIMA metrics for %s: %s", asset_name, metrics)

        return metrics

    def fit(
        self,
        dataframe: pd.DataFrame,
        assets: Optional[List[str]] = None,
    ) -> "ARIMAForecaster":
        """
        Fit ARIMA models for all or selected assets.

        Args:
            dataframe (pd.DataFrame): Input dataframe.
            assets (Optional[List[str]]): Selected assets to model.

        Returns:
            ARIMAForecaster: Fitted object.
        """
        logger.info("Starting ARIMA training process.")

        data = clean_time_series_data(
            dataframe=dataframe,
            date_column=self.date_column,
            asset_column=self.asset_column,
            target_column=self.target_column,
        )

        if assets is None:
            assets = sorted(data[self.asset_column].unique().tolist())

        for asset_name in assets:
            try:
                series = prepare_asset_series(
                    dataframe=data,
                    asset_name=asset_name,
                    date_column=self.date_column,
                    asset_column=self.asset_column,
                    target_column=self.target_column,
                    frequency=self.frequency,
                )

                if len(series) < 10:
                    logger.warning(
                        "Skipping asset %s because it has fewer than 10 records.",
                        asset_name,
                    )
                    continue

                train_series, test_series = time_based_train_test_split(
                    series=series,
                    test_size=self.test_size,
                )

                fitted_model = self.fit_asset_model(
                    series=train_series,
                    asset_name=asset_name,
                )

                self.evaluate_asset_model(
                    fitted_model=fitted_model,
                    train_series=train_series,
                    test_series=test_series,
                    asset_name=asset_name,
                )

            except Exception as error:
                logger.error(
                    "Failed to train ARIMA model for asset %s: %s",
                    asset_name,
                    error,
                    exc_info=True,
                )

        logger.info("ARIMA training process completed.")

        return self

    def forecast_asset(
        self,
        asset_name: str,
        steps: int = 30,
    ) -> pd.DataFrame:
        """
        Forecast future prices for one asset.

        Args:
            asset_name (str): Asset name.
            steps (int): Number of future periods.

        Returns:
            pd.DataFrame: Forecast dataframe.
        """
        if asset_name not in self.models:
            raise ValueError(
                f"No trained ARIMA model found for asset '{asset_name}'."
            )

        fitted_model = self.models[asset_name]

        forecast_values = fitted_model.forecast(steps=steps)

        last_date = fitted_model.data.dates[-1]

        if last_date is None:
            forecast_dates = pd.date_range(
                start=pd.Timestamp.today().normalize(),
                periods=steps,
                freq=self.frequency,
            )
        else:
            forecast_dates = pd.date_range(
                start=pd.to_datetime(last_date) + pd.tseries.frequencies.to_offset(self.frequency),
                periods=steps,
                freq=self.frequency,
            )

        forecast_dataframe = pd.DataFrame(
            {
                self.date_column: forecast_dates,
                self.asset_column: asset_name,
                "forecasted_price": forecast_values.values,
            }
        )

        forecast_dataframe["forecasted_price"] = forecast_dataframe[
            "forecasted_price"
        ].clip(lower=0)

        return forecast_dataframe

    def forecast_all(
        self,
        steps: int = 30,
    ) -> pd.DataFrame:
        """
        Forecast future prices for all trained assets.

        Args:
            steps (int): Number of future periods.

        Returns:
            pd.DataFrame: Combined forecast dataframe.
        """
        if not self.models:
            raise ValueError("No ARIMA models are trained yet.")

        all_forecasts = []

        for asset_name in self.models.keys():
            try:
                forecast_dataframe = self.forecast_asset(
                    asset_name=asset_name,
                    steps=steps,
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
            raise ValueError("No forecasts were generated.")

        self.forecast_results = pd.concat(
            all_forecasts,
            ignore_index=True,
        )

        return self.forecast_results

    def save_models(
        self,
        file_path: Path = ARIMA_MODEL_PATH,
    ) -> None:
        """
        Save trained ARIMA models.

        Args:
            file_path (Path): Output model path.
        """
        create_parent_directory(file_path)

        with file_path.open("wb") as file:
            pickle.dump(
                {
                    "order": self.order,
                    "date_column": self.date_column,
                    "asset_column": self.asset_column,
                    "target_column": self.target_column,
                    "frequency": self.frequency,
                    "test_size": self.test_size,
                    "models": self.models,
                    "metrics": self.metrics,
                },
                file,
            )

        logger.info("ARIMA models saved at: %s", file_path)

    def load_models(
        self,
        file_path: Path = ARIMA_MODEL_PATH,
    ) -> "ARIMAForecaster":
        """
        Load saved ARIMA models.

        Args:
            file_path (Path): Saved model path.

        Returns:
            ARIMAForecaster: Loaded object.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"ARIMA model file not found: {file_path}")

        with file_path.open("rb") as file:
            saved_data = pickle.load(file)

        self.order = saved_data.get("order", self.order)
        self.date_column = saved_data.get("date_column", self.date_column)
        self.asset_column = saved_data.get("asset_column", self.asset_column)
        self.target_column = saved_data.get("target_column", self.target_column)
        self.frequency = saved_data.get("frequency", self.frequency)
        self.test_size = saved_data.get("test_size", self.test_size)

        self.models = saved_data.get("models", {})
        self.metrics = saved_data.get("metrics", {})

        logger.info("ARIMA models loaded from: %s", file_path)

        return self

    def save_metrics(
        self,
        file_path: Path = ARIMA_METRICS_PATH,
    ) -> None:
        """
        Save ARIMA evaluation metrics as JSON.

        Args:
            file_path (Path): Output metrics path.
        """
        create_parent_directory(file_path)

        with file_path.open("w", encoding="utf-8") as file:
            json.dump(self.metrics, file, indent=4)

        logger.info("ARIMA metrics saved at: %s", file_path)

    def save_forecasts(
        self,
        dataframe: Optional[pd.DataFrame] = None,
        file_path: Path = ARIMA_FORECAST_PATH,
    ) -> None:
        """
        Save forecast results as CSV.

        Args:
            dataframe (Optional[pd.DataFrame]): Forecast dataframe.
            file_path (Path): Output CSV path.
        """
        forecast_dataframe = dataframe if dataframe is not None else self.forecast_results

        if forecast_dataframe is None or forecast_dataframe.empty:
            raise ValueError("No forecast results available to save.")

        create_parent_directory(file_path)

        forecast_dataframe.to_csv(file_path, index=False)

        logger.info("ARIMA forecast results saved at: %s", file_path)

    def get_metrics_dataframe(self) -> pd.DataFrame:
        """
        Convert metrics dictionary into dataframe.

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
        file_path: Path = REPORTS_TABLES_DIR / "arima_metrics.csv",
    ) -> None:
        """
        Save ARIMA metrics as CSV.

        Args:
            file_path (Path): Output CSV path.
        """
        metrics_dataframe = self.get_metrics_dataframe()

        if metrics_dataframe.empty:
            raise ValueError("No ARIMA metrics available to save.")

        create_parent_directory(file_path)

        metrics_dataframe.to_csv(file_path, index=False)

        logger.info("ARIMA metrics CSV saved at: %s", file_path)

    def train_forecast_save(
        self,
        dataframe: pd.DataFrame,
        forecast_steps: int = 30,
        assets: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run complete ARIMA workflow.

        Steps:
        1. Train ARIMA models
        2. Evaluate models
        3. Forecast future prices
        4. Save models, metrics, and forecasts

        Args:
            dataframe (pd.DataFrame): Input dataframe.
            forecast_steps (int): Number of periods to forecast.
            assets (Optional[List[str]]): Selected assets.

        Returns:
            Dict[str, Any]: Workflow summary.
        """
        self.fit(dataframe=dataframe, assets=assets)

        forecast_dataframe = self.forecast_all(steps=forecast_steps)

        self.save_models()
        self.save_metrics()
        self.save_metrics_csv()
        self.save_forecasts(forecast_dataframe)

        summary = {
            "status": "success",
            "model_type": "ARIMA",
            "order": self.order,
            "trained_assets": list(self.models.keys()),
            "asset_count": len(self.models),
            "forecast_steps": forecast_steps,
            "forecast_rows": len(forecast_dataframe),
            "model_path": str(ARIMA_MODEL_PATH),
            "metrics_path": str(ARIMA_METRICS_PATH),
            "forecast_path": str(ARIMA_FORECAST_PATH),
        }

        logger.info("ARIMA workflow summary: %s", summary)

        return summary


# ============================================================
# Convenience Functions
# ============================================================

def train_arima_models(
    dataframe: pd.DataFrame,
    order: Tuple[int, int, int] = (5, 1, 0),
    forecast_steps: int = 30,
    assets: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Train ARIMA models and generate forecasts.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        order (Tuple[int, int, int]): ARIMA order.
        forecast_steps (int): Number of forecast periods.
        assets (Optional[List[str]]): Optional selected assets.

    Returns:
        Dict[str, Any]: Training summary.
    """
    forecaster = ARIMAForecaster(order=order)

    return forecaster.train_forecast_save(
        dataframe=dataframe,
        forecast_steps=forecast_steps,
        assets=assets,
    )


def load_arima_forecaster(
    model_path: Path = ARIMA_MODEL_PATH,
) -> ARIMAForecaster:
    """
    Load saved ARIMA forecaster.

    Args:
        model_path (Path): Saved model path.

    Returns:
        ARIMAForecaster: Loaded forecaster.
    """
    forecaster = ARIMAForecaster()
    forecaster.load_models(model_path)

    return forecaster


def forecast_with_saved_arima(
    forecast_steps: int = 30,
    model_path: Path = ARIMA_MODEL_PATH,
    output_path: Path = ARIMA_FORECAST_PATH,
) -> pd.DataFrame:
    """
    Generate forecasts using saved ARIMA models.

    Args:
        forecast_steps (int): Number of forecast periods.
        model_path (Path): Saved ARIMA model path.
        output_path (Path): Output forecast path.

    Returns:
        pd.DataFrame: Forecast dataframe.
    """
    forecaster = load_arima_forecaster(model_path)
    forecast_dataframe = forecaster.forecast_all(steps=forecast_steps)
    forecaster.save_forecasts(forecast_dataframe, output_path)

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

    arima_forecaster = ARIMAForecaster(
        order=(5, 1, 0),
        date_column="date",
        asset_column="asset",
        target_column="price",
        frequency="D",
        test_size=0.2,
    )

    workflow_summary = arima_forecaster.train_forecast_save(
        dataframe=data,
        forecast_steps=30,
    )

    print("ARIMA workflow completed successfully.")
    print(workflow_summary)