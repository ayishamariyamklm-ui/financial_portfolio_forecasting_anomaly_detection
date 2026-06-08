"""
LSTM forecasting model for the Financial Portfolio Forecasting
and Anomaly Detection project.

This module handles:
- Asset-wise LSTM model training
- Sequence preparation for time series forecasting
- Feature scaling
- Model evaluation
- Future price forecasting
- Model artifact saving and loading
- Metrics and forecast output saving

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
    import tensorflow as tf  # type: ignore[reportMissingModuleSource]
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore[reportMissingModuleSource]
    from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore[reportMissingModuleSource]
    from tensorflow.keras.models import Sequential, load_model # type: ignore[reportMissingModuleSource] 
except ImportError as exc:
    raise ImportError(
        "TensorFlow is required for LSTM modeling. "
        "Install it using: pip install tensorflow"
    ) from exc

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

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

LSTM_MODEL_DIR = MODELS_DIR / "lstm_models"
LSTM_SCALER_PATH = MODELS_DIR / "lstm_scalers.pkl"
LSTM_METRICS_JSON_PATH = REPORTS_TABLES_DIR / "lstm_metrics.json"
LSTM_METRICS_CSV_PATH = REPORTS_TABLES_DIR / "lstm_metrics.csv"
LSTM_PREDICTIONS_PATH = OUTPUTS_DIR / "lstm_test_predictions.csv"
LSTM_FORECAST_PATH = OUTPUTS_DIR / "lstm_forecast_results.csv"


# ============================================================
# Helper Functions
# ============================================================

def ensure_directories() -> None:
    """
    Create required directories for models, reports, and outputs.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LSTM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed (int): Random seed.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.

    Args:
        y_true (np.ndarray): Actual values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: RMSE value.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    Args:
        y_true (np.ndarray): Actual values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: MAPE value.
    """
    y_true_array = np.array(y_true).flatten()
    y_pred_array = np.array(y_pred).flatten()

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


def prepare_asset_series(
    dataframe: pd.DataFrame,
    asset_name: str,
    date_column: str = "date",
    asset_column: str = "asset",
    target_column: str = "price",
    frequency: str = "D",
) -> pd.DataFrame:
    """
    Prepare one asset's price series for LSTM.

    Duplicate dates are averaged.
    Missing dates are filled using forward fill and backward fill.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        asset_name (str): Asset name.
        date_column (str): Date column name.
        asset_column (str): Asset column name.
        target_column (str): Target column name.
        frequency (str): Time frequency.

    Returns:
        pd.DataFrame: Asset dataframe with date and price.
    """
    asset_data = dataframe[dataframe[asset_column] == asset_name].copy()

    if asset_data.empty:
        raise ValueError(f"No data found for asset: {asset_name}")

    asset_series = (
        asset_data.groupby(date_column, as_index=False)
        .agg(price=(target_column, "mean"))
        .sort_values(date_column)
    )

    asset_series[date_column] = pd.to_datetime(
        asset_series[date_column],
        errors="coerce",
    )

    asset_series = asset_series.dropna(subset=[date_column, "price"])
    asset_series = asset_series.set_index(date_column).asfreq(frequency)
    asset_series["price"] = asset_series["price"].ffill().bfill()
    asset_series = asset_series.reset_index()

    asset_series[asset_column] = asset_name

    return asset_series[[date_column, asset_column, "price"]]


def create_sequences(
    scaled_values: np.ndarray,
    sequence_length: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input-output sequences for LSTM.

    Args:
        scaled_values (np.ndarray): Scaled target values.
        sequence_length (int): Number of previous steps used for prediction.

    Returns:
        Tuple[np.ndarray, np.ndarray]: X and y arrays.
    """
    X = []
    y = []

    for index in range(sequence_length, len(scaled_values)):
        X.append(scaled_values[index - sequence_length:index])
        y.append(scaled_values[index])

    return np.array(X), np.array(y)


def inverse_transform_values(
    scaler: MinMaxScaler,
    values: np.ndarray,
) -> np.ndarray:
    """
    Convert scaled values back to original scale.

    Args:
        scaler (MinMaxScaler): Fitted scaler.
        values (np.ndarray): Scaled values.

    Returns:
        np.ndarray: Values in original scale.
    """
    values = np.array(values).reshape(-1, 1)
    return scaler.inverse_transform(values).flatten()


def safe_asset_filename(asset_name: str) -> str:
    """
    Convert asset name into safe filename.

    Args:
        asset_name (str): Asset name.

    Returns:
        str: Safe filename.
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
# LSTM Forecaster Class
# ============================================================

class LSTMForecaster:
    """
    Asset-wise LSTM forecaster for asset price prediction.
    """

    def __init__(
        self,
        date_column: str = "date",
        asset_column: str = "asset",
        target_column: str = "price",
        frequency: str = "D",
        sequence_length: int = 10,
        test_size: float = 0.2,
        epochs: int = 50,
        batch_size: int = 16,
        random_state: int = 42,
    ) -> None:
        """
        Initialize LSTMForecaster.

        Args:
            date_column (str): Date column name.
            asset_column (str): Asset column name.
            target_column (str): Target column name.
            frequency (str): Time frequency.
            sequence_length (int): Number of past steps used to predict next price.
            test_size (float): Test data ratio.
            epochs (int): Training epochs.
            batch_size (int): Training batch size.
            random_state (int): Random seed.
        """
        self.date_column = date_column
        self.asset_column = asset_column
        self.target_column = target_column
        self.frequency = frequency
        self.sequence_length = sequence_length
        self.test_size = test_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, MinMaxScaler] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}
        self.test_predictions: Optional[pd.DataFrame] = None
        self.forecast_results: Optional[pd.DataFrame] = None

        ensure_directories()
        set_random_seed(self.random_state)

    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Build LSTM neural network.

        Args:
            input_shape (Tuple[int, int]): Shape of input sequence.

        Returns:
            Sequential: Compiled LSTM model.
        """
        model = Sequential(
            [
                LSTM(
                    units=64,
                    return_sequences=True,
                    input_shape=input_shape,
                ),
                Dropout(0.2),
                LSTM(
                    units=32,
                    return_sequences=False,
                ),
                Dropout(0.2),
                Dense(units=16, activation="relu"),
                Dense(units=1),
            ]
        )

        model.compile(
            optimizer="adam",
            loss="mean_squared_error",
            metrics=["mae"],
        )

        return model

    def prepare_train_test_sequences(
        self,
        asset_dataframe: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, pd.DataFrame]:
        """
        Prepare train and test sequences for one asset.

        Args:
            asset_dataframe (pd.DataFrame): Asset price dataframe.

        Returns:
            Tuple containing:
            - X_train
            - X_test
            - y_train
            - y_test
            - scaler
            - sequence_dataframe
        """
        data = asset_dataframe.copy()
        data = data.sort_values(self.date_column).reset_index(drop=True)

        values = data[["price"]].values

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_values = scaler.fit_transform(values)

        X, y = create_sequences(
            scaled_values=scaled_values,
            sequence_length=self.sequence_length,
        )

        if len(X) == 0:
            raise ValueError(
                "Not enough records to create LSTM sequences. "
                f"Need more than {self.sequence_length} records."
            )

        sequence_dates = data[self.date_column].iloc[self.sequence_length:].reset_index(drop=True)

        sequence_dataframe = pd.DataFrame(
            {
                self.date_column: sequence_dates,
                "actual_price": data["price"].iloc[self.sequence_length:].values,
            }
        )

        split_index = int(len(X) * (1 - self.test_size))

        X_train = X[:split_index]
        X_test = X[split_index:]
        y_train = y[:split_index]
        y_test = y[split_index:]

        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError("Train or test sequence set is empty.")

        return X_train, X_test, y_train, y_test, scaler, sequence_dataframe

    def fit_asset_model(
        self,
        asset_dataframe: pd.DataFrame,
        asset_name: str,
    ) -> Dict[str, float]:
        """
        Train LSTM model for one asset.

        Args:
            asset_dataframe (pd.DataFrame): Asset dataframe.
            asset_name (str): Asset name.

        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        logger.info("Training LSTM model for asset: %s", asset_name)

        (
            X_train,
            X_test,
            y_train,
            y_test,
            scaler,
            sequence_dataframe,
        ) = self.prepare_train_test_sequences(asset_dataframe)

        model = self.build_model(
            input_shape=(X_train.shape[1], X_train.shape[2])
        )

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=0.0001,
            ),
        ]

        model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0,
        )

        scaled_predictions = model.predict(X_test, verbose=0)

        y_test_original = inverse_transform_values(scaler, y_test)
        predictions_original = inverse_transform_values(scaler, scaled_predictions)

        predictions_original = np.clip(predictions_original, a_min=0, a_max=None)

        metrics = self.evaluate_predictions(
            y_true=y_test_original,
            y_pred=predictions_original,
            train_records=len(X_train),
            test_records=len(X_test),
        )

        self.models[asset_name] = model
        self.scalers[asset_name] = scaler
        self.metrics[asset_name] = metrics

        test_start_index = len(sequence_dataframe) - len(y_test_original)

        asset_predictions = sequence_dataframe.iloc[test_start_index:].copy()
        asset_predictions[self.asset_column] = asset_name
        asset_predictions["predicted_price"] = predictions_original
        asset_predictions["absolute_error"] = (
            asset_predictions["actual_price"]
            - asset_predictions["predicted_price"]
        ).abs()

        logger.info("LSTM metrics for %s: %s", asset_name, metrics)

        return {
            "metrics": metrics,
            "predictions": asset_predictions,
        }

    def evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        train_records: int,
        test_records: int,
    ) -> Dict[str, float]:
        """
        Evaluate LSTM predictions.

        Args:
            y_true (np.ndarray): Actual prices.
            y_pred (np.ndarray): Predicted prices.
            train_records (int): Number of training records.
            test_records (int): Number of test records.

        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = calculate_rmse(y_true, y_pred)
        mape = calculate_mape(y_true, y_pred)

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
            "train_records": int(train_records),
            "test_records": int(test_records),
        }

    def fit(
        self,
        dataframe: pd.DataFrame,
        assets: Optional[List[str]] = None,
    ) -> "LSTMForecaster":
        """
        Train LSTM models for all or selected assets.

        Args:
            dataframe (pd.DataFrame): Input dataframe.
            assets (Optional[List[str]]): Optional list of selected assets.

        Returns:
            LSTMForecaster: Fitted forecaster.
        """
        logger.info("Starting LSTM training process.")

        data = clean_input_data(
            dataframe=dataframe,
            date_column=self.date_column,
            asset_column=self.asset_column,
            target_column=self.target_column,
        )

        if assets is None:
            assets = sorted(data[self.asset_column].unique().tolist())

        all_predictions = []

        for asset_name in assets:
            try:
                asset_dataframe = prepare_asset_series(
                    dataframe=data,
                    asset_name=asset_name,
                    date_column=self.date_column,
                    asset_column=self.asset_column,
                    target_column=self.target_column,
                    frequency=self.frequency,
                )

                if len(asset_dataframe) <= self.sequence_length + 5:
                    logger.warning(
                        "Skipping asset %s because it has insufficient records for LSTM.",
                        asset_name,
                    )
                    continue

                result = self.fit_asset_model(
                    asset_dataframe=asset_dataframe,
                    asset_name=asset_name,
                )

                all_predictions.append(result["predictions"])

            except Exception as error:
                logger.error(
                    "Failed to train LSTM model for asset %s: %s",
                    asset_name,
                    error,
                    exc_info=True,
                )

        if all_predictions:
            self.test_predictions = pd.concat(
                all_predictions,
                ignore_index=True,
            )

        logger.info("LSTM training process completed.")

        return self

    def recursive_forecast_asset(
        self,
        asset_name: str,
        historical_dataframe: pd.DataFrame,
        forecast_steps: int = 30,
    ) -> pd.DataFrame:
        """
        Generate recursive future forecast for one asset.

        Args:
            asset_name (str): Asset name.
            historical_dataframe (pd.DataFrame): Historical dataframe for all assets.
            forecast_steps (int): Number of future forecast periods.

        Returns:
            pd.DataFrame: Forecast dataframe.
        """
        if asset_name not in self.models:
            raise ValueError(f"No trained LSTM model found for asset: {asset_name}")

        if asset_name not in self.scalers:
            raise ValueError(f"No fitted scaler found for asset: {asset_name}")

        model = self.models[asset_name]
        scaler = self.scalers[asset_name]

        data = clean_input_data(
            dataframe=historical_dataframe,
            date_column=self.date_column,
            asset_column=self.asset_column,
            target_column=self.target_column,
        )

        asset_dataframe = prepare_asset_series(
            dataframe=data,
            asset_name=asset_name,
            date_column=self.date_column,
            asset_column=self.asset_column,
            target_column=self.target_column,
            frequency=self.frequency,
        )

        asset_dataframe = asset_dataframe.sort_values(self.date_column).reset_index(drop=True)

        recent_prices = asset_dataframe[["price"]].values[-self.sequence_length:]

        if len(recent_prices) < self.sequence_length:
            raise ValueError(
                f"Not enough recent values to forecast asset: {asset_name}"
            )

        scaled_recent_prices = scaler.transform(recent_prices)

        current_sequence = scaled_recent_prices.reshape(
            1,
            self.sequence_length,
            1,
        )

        forecast_rows = []

        last_date = asset_dataframe[self.date_column].max()

        for step in range(1, forecast_steps + 1):
            scaled_prediction = model.predict(current_sequence, verbose=0)

            predicted_price = inverse_transform_values(
                scaler,
                scaled_prediction,
            )[0]

            predicted_price = max(float(predicted_price), 0.0)

            future_date = last_date + pd.Timedelta(days=step)

            forecast_rows.append(
                {
                    self.date_column: future_date,
                    self.asset_column: asset_name,
                    "forecasted_price": predicted_price,
                }
            )

            new_scaled_value = scaler.transform([[predicted_price]])

            current_sequence = np.append(
                current_sequence[:, 1:, :],
                new_scaled_value.reshape(1, 1, 1),
                axis=1,
            )

        return pd.DataFrame(forecast_rows)

    def forecast_all(
        self,
        dataframe: pd.DataFrame,
        forecast_steps: int = 30,
    ) -> pd.DataFrame:
        """
        Forecast future prices for all trained assets.

        Args:
            dataframe (pd.DataFrame): Historical dataframe.
            forecast_steps (int): Forecast horizon.

        Returns:
            pd.DataFrame: Forecast dataframe.
        """
        if not self.models:
            raise ValueError("No LSTM models are trained yet.")

        all_forecasts = []

        for asset_name in self.models.keys():
            try:
                forecast_dataframe = self.recursive_forecast_asset(
                    asset_name=asset_name,
                    historical_dataframe=dataframe,
                    forecast_steps=forecast_steps,
                )

                all_forecasts.append(forecast_dataframe)

            except Exception as error:
                logger.error(
                    "LSTM forecast failed for asset %s: %s",
                    asset_name,
                    error,
                    exc_info=True,
                )

        if not all_forecasts:
            raise ValueError("No LSTM forecasts were generated.")

        self.forecast_results = pd.concat(
            all_forecasts,
            ignore_index=True,
        )

        return self.forecast_results

    def save_models(self) -> None:
        """
        Save all trained LSTM models as .keras files.
        """
        if not self.models:
            raise ValueError("No LSTM models available to save.")

        LSTM_MODEL_DIR.mkdir(parents=True, exist_ok=True)

        for asset_name, model in self.models.items():
            safe_name = safe_asset_filename(asset_name)
            model_path = LSTM_MODEL_DIR / f"lstm_{safe_name}.keras"
            model.save(model_path)
            logger.info("LSTM model for %s saved at: %s", asset_name, model_path)

    def save_scalers(
        self,
        file_path: Path = LSTM_SCALER_PATH,
    ) -> None:
        """
        Save fitted scalers.

        Args:
            file_path (Path): Output scaler path.
        """
        if not self.scalers:
            raise ValueError("No LSTM scalers available to save.")

        create_parent_directory(file_path)

        with file_path.open("wb") as file:
            pickle.dump(
                {
                    "scalers": self.scalers,
                    "date_column": self.date_column,
                    "asset_column": self.asset_column,
                    "target_column": self.target_column,
                    "frequency": self.frequency,
                    "sequence_length": self.sequence_length,
                    "test_size": self.test_size,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "metrics": self.metrics,
                },
                file,
            )

        logger.info("LSTM scalers saved at: %s", file_path)

    def load_models_and_scalers(
        self,
        model_dir: Path = LSTM_MODEL_DIR,
        scaler_path: Path = LSTM_SCALER_PATH,
    ) -> "LSTMForecaster":
        """
        Load saved LSTM models and scalers.

        Args:
            model_dir (Path): Directory containing .keras models.
            scaler_path (Path): Scaler pickle path.

        Returns:
            LSTMForecaster: Loaded forecaster.
        """
        if not scaler_path.exists():
            raise FileNotFoundError(f"LSTM scaler file not found: {scaler_path}")

        with scaler_path.open("rb") as file:
            saved_data = pickle.load(file)

        self.scalers = saved_data.get("scalers", {})
        self.metrics = saved_data.get("metrics", {})

        self.date_column = saved_data.get("date_column", self.date_column)
        self.asset_column = saved_data.get("asset_column", self.asset_column)
        self.target_column = saved_data.get("target_column", self.target_column)
        self.frequency = saved_data.get("frequency", self.frequency)
        self.sequence_length = saved_data.get("sequence_length", self.sequence_length)
        self.test_size = saved_data.get("test_size", self.test_size)
        self.epochs = saved_data.get("epochs", self.epochs)
        self.batch_size = saved_data.get("batch_size", self.batch_size)

        self.models = {}

        for asset_name in self.scalers.keys():
            safe_name = safe_asset_filename(asset_name)
            model_path = model_dir / f"lstm_{safe_name}.keras"

            if model_path.exists():
                self.models[asset_name] = load_model(model_path)
                logger.info("Loaded LSTM model for %s from %s", asset_name, model_path)
            else:
                logger.warning(
                    "LSTM model file missing for asset %s at %s",
                    asset_name,
                    model_path,
                )

        return self

    def save_metrics_json(
        self,
        file_path: Path = LSTM_METRICS_JSON_PATH,
    ) -> None:
        """
        Save LSTM metrics as JSON.

        Args:
            file_path (Path): Output JSON path.
        """
        if not self.metrics:
            raise ValueError("No LSTM metrics available to save.")

        create_parent_directory(file_path)

        with file_path.open("w", encoding="utf-8") as file:
            json.dump(self.metrics, file, indent=4)

        logger.info("LSTM metrics JSON saved at: %s", file_path)

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
        file_path: Path = LSTM_METRICS_CSV_PATH,
    ) -> None:
        """
        Save LSTM metrics as CSV.

        Args:
            file_path (Path): Output CSV path.
        """
        metrics_dataframe = self.get_metrics_dataframe()

        if metrics_dataframe.empty:
            raise ValueError("No LSTM metrics available to save.")

        create_parent_directory(file_path)

        metrics_dataframe.to_csv(file_path, index=False)

        logger.info("LSTM metrics CSV saved at: %s", file_path)

    def save_test_predictions(
        self,
        file_path: Path = LSTM_PREDICTIONS_PATH,
    ) -> None:
        """
        Save LSTM test predictions as CSV.

        Args:
            file_path (Path): Output CSV path.
        """
        if self.test_predictions is None or self.test_predictions.empty:
            raise ValueError("No LSTM test predictions available to save.")

        create_parent_directory(file_path)

        self.test_predictions.to_csv(file_path, index=False)

        logger.info("LSTM test predictions saved at: %s", file_path)

    def save_forecasts(
        self,
        dataframe: Optional[pd.DataFrame] = None,
        file_path: Path = LSTM_FORECAST_PATH,
    ) -> None:
        """
        Save LSTM forecast results as CSV.

        Args:
            dataframe (Optional[pd.DataFrame]): Forecast dataframe.
            file_path (Path): Output CSV path.
        """
        forecast_dataframe = dataframe if dataframe is not None else self.forecast_results

        if forecast_dataframe is None or forecast_dataframe.empty:
            raise ValueError("No LSTM forecast results available to save.")

        create_parent_directory(file_path)

        forecast_dataframe.to_csv(file_path, index=False)

        logger.info("LSTM forecast results saved at: %s", file_path)

    def save_all_artifacts(self) -> None:
        """
        Save LSTM models, scalers, metrics, and predictions.
        """
        self.save_models()
        self.save_scalers()
        self.save_metrics_json()
        self.save_metrics_csv()

        if self.test_predictions is not None and not self.test_predictions.empty:
            self.save_test_predictions()

    def train_forecast_save(
        self,
        dataframe: pd.DataFrame,
        forecast_steps: int = 30,
        assets: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run complete LSTM workflow.

        Steps:
        1. Train asset-wise LSTM models
        2. Evaluate test predictions
        3. Generate future forecasts
        4. Save models, scalers, metrics, predictions, and forecasts

        Args:
            dataframe (pd.DataFrame): Input dataframe.
            forecast_steps (int): Number of future forecast periods.
            assets (Optional[List[str]]): Optional selected assets.

        Returns:
            Dict[str, Any]: Workflow summary.
        """
        self.fit(dataframe=dataframe, assets=assets)

        forecast_dataframe = self.forecast_all(
            dataframe=dataframe,
            forecast_steps=forecast_steps,
        )

        self.save_all_artifacts()
        self.save_forecasts(forecast_dataframe)

        summary = {
            "status": "success",
            "model_type": "LSTM",
            "target_column": self.target_column,
            "trained_assets": list(self.models.keys()),
            "asset_count": len(self.models),
            "sequence_length": self.sequence_length,
            "forecast_steps": forecast_steps,
            "forecast_rows": len(forecast_dataframe),
            "metrics": self.metrics,
            "model_dir": str(LSTM_MODEL_DIR),
            "scaler_path": str(LSTM_SCALER_PATH),
            "metrics_path": str(LSTM_METRICS_JSON_PATH),
            "forecast_path": str(LSTM_FORECAST_PATH),
        }

        logger.info("LSTM workflow summary: %s", summary)

        return summary


# ============================================================
# Convenience Functions
# ============================================================

def train_lstm_models(
    dataframe: pd.DataFrame,
    forecast_steps: int = 30,
    assets: Optional[List[str]] = None,
    sequence_length: int = 10,
    epochs: int = 50,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """
    Train LSTM models and generate forecasts.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        forecast_steps (int): Forecast horizon.
        assets (Optional[List[str]]): Optional selected assets.
        sequence_length (int): Number of previous records used for forecasting.
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.

    Returns:
        Dict[str, Any]: Workflow summary.
    """
    forecaster = LSTMForecaster(
        sequence_length=sequence_length,
        epochs=epochs,
        batch_size=batch_size,
    )

    return forecaster.train_forecast_save(
        dataframe=dataframe,
        forecast_steps=forecast_steps,
        assets=assets,
    )


def load_lstm_forecaster(
    model_dir: Path = LSTM_MODEL_DIR,
    scaler_path: Path = LSTM_SCALER_PATH,
) -> LSTMForecaster:
    """
    Load saved LSTM forecaster.

    Args:
        model_dir (Path): Saved model directory.
        scaler_path (Path): Saved scaler path.

    Returns:
        LSTMForecaster: Loaded forecaster.
    """
    forecaster = LSTMForecaster()
    forecaster.load_models_and_scalers(
        model_dir=model_dir,
        scaler_path=scaler_path,
    )

    return forecaster


def forecast_with_saved_lstm(
    dataframe: pd.DataFrame,
    forecast_steps: int = 30,
    model_dir: Path = LSTM_MODEL_DIR,
    scaler_path: Path = LSTM_SCALER_PATH,
    output_path: Path = LSTM_FORECAST_PATH,
) -> pd.DataFrame:
    """
    Generate forecasts using saved LSTM models.

    Args:
        dataframe (pd.DataFrame): Historical dataframe.
        forecast_steps (int): Forecast horizon.
        model_dir (Path): Saved model directory.
        scaler_path (Path): Saved scaler path.
        output_path (Path): Output forecast path.

    Returns:
        pd.DataFrame: Forecast dataframe.
    """
    forecaster = load_lstm_forecaster(
        model_dir=model_dir,
        scaler_path=scaler_path,
    )

    forecast_dataframe = forecaster.forecast_all(
        dataframe=dataframe,
        forecast_steps=forecast_steps,
    )

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

    lstm_forecaster = LSTMForecaster(
        date_column="date",
        asset_column="asset",
        target_column="price",
        frequency="D",
        sequence_length=10,
        test_size=0.2,
        epochs=50,
        batch_size=16,
        random_state=42,
    )

    workflow_summary = lstm_forecaster.train_forecast_save(
        dataframe=data,
        forecast_steps=30,
    )

    print("LSTM workflow completed successfully.")
    print(workflow_summary)