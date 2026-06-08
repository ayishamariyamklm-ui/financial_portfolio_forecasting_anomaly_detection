"""
Data loading utilities for the Financial Portfolio Forecasting
and Anomaly Detection project.

This module handles:
- Loading raw financial portfolio data
- Loading processed train/test/anomaly/forecast data
- Saving datasets safely
- Basic file validation
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from src.monitoring.logger import get_project_logger
from src.utils.exception_handler import (
    DataError,
    EmptyDataFrameError,
    FileNotFoundProjectError,
    validate_not_empty,
)
from src.utils.paths import (
    create_parent_directory,
    get_anomaly_results_path,
    get_cleaned_data_path,
    get_forecast_results_path,
    get_raw_data_path,
    get_test_data_path,
    get_train_data_path,
    resolve_path,
)


logger = get_project_logger()


class DataLoader:
    """
    Load and save financial portfolio datasets.
    """

    def __init__(self) -> None:
        """
        Initialize DataLoader.
        """
        logger.info("DataLoader initialized.")

    def load_csv(
        self,
        file_path: Path,
        parse_dates: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Load a CSV file as a pandas DataFrame.

        Args:
            file_path (Path): CSV file path.
            parse_dates (Optional[list]): Columns to parse as dates.

        Returns:
            pd.DataFrame: Loaded dataframe.

        Raises:
            FileNotFoundProjectError: If file does not exist.
            EmptyDataFrameError: If loaded dataframe is empty.
            DataError: If loading fails.
        """
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                raise FileNotFoundProjectError(file_path)

            dataframe = pd.read_csv(file_path, parse_dates=parse_dates)

            validate_not_empty(dataframe, name=str(file_path))

            logger.info(
                "CSV loaded successfully from %s with shape %s",
                file_path,
                dataframe.shape,
            )

            return dataframe

        except FileNotFoundProjectError:
            raise

        except EmptyDataFrameError:
            raise

        except Exception as error:
            logger.error(
                "Failed to load CSV file from %s: %s",
                file_path,
                error,
                exc_info=True,
            )
            raise DataError(
                message="Failed to load CSV file.",
                details={"file_path": str(file_path), "error": str(error)},
            ) from error

    def save_csv(
        self,
        dataframe: pd.DataFrame,
        file_path: Path,
        index: bool = False,
    ) -> None:
        """
        Save a dataframe to CSV.

        Args:
            dataframe (pd.DataFrame): Dataframe to save.
            file_path (Path): Output CSV path.
            index (bool): Whether to save dataframe index.

        Raises:
            EmptyDataFrameError: If dataframe is empty.
            DataError: If saving fails.
        """
        try:
            validate_not_empty(dataframe, name="dataframe")

            file_path = Path(file_path)
            create_parent_directory(file_path)

            dataframe.to_csv(file_path, index=index)

            logger.info(
                "CSV saved successfully at %s with shape %s",
                file_path,
                dataframe.shape,
            )

        except EmptyDataFrameError:
            raise

        except Exception as error:
            logger.error(
                "Failed to save CSV file at %s: %s",
                file_path,
                error,
                exc_info=True,
            )
            raise DataError(
                message="Failed to save CSV file.",
                details={"file_path": str(file_path), "error": str(error)},
            ) from error

    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw financial portfolio data.

        Returns:
            pd.DataFrame: Raw financial data.
        """
        return self.load_csv(get_raw_data_path(), parse_dates=["Date"])

    def load_cleaned_data(self) -> pd.DataFrame:
        """
        Load cleaned financial portfolio data.

        Returns:
            pd.DataFrame: Cleaned financial data.
        """
        return self.load_csv(get_cleaned_data_path(), parse_dates=["Date"])

    def load_train_data(self) -> pd.DataFrame:
        """
        Load training data.

        Returns:
            pd.DataFrame: Training dataframe.
        """
        return self.load_csv(get_train_data_path(), parse_dates=["Date"])

    def load_test_data(self) -> pd.DataFrame:
        """
        Load test data.

        Returns:
            pd.DataFrame: Test dataframe.
        """
        return self.load_csv(get_test_data_path(), parse_dates=["Date"])

    def load_anomaly_results(self) -> pd.DataFrame:
        """
        Load anomaly detection results.

        Returns:
            pd.DataFrame: Anomaly results dataframe.
        """
        return self.load_csv(get_anomaly_results_path(), parse_dates=["Date"])

    def load_forecast_results(self) -> pd.DataFrame:
        """
        Load forecast results.

        Returns:
            pd.DataFrame: Forecast results dataframe.
        """
        return self.load_csv(get_forecast_results_path(), parse_dates=["Date"])

    def save_cleaned_data(self, dataframe: pd.DataFrame) -> None:
        """
        Save cleaned financial data.

        Args:
            dataframe (pd.DataFrame): Cleaned dataframe.
        """
        self.save_csv(dataframe, get_cleaned_data_path())

    def save_train_data(self, dataframe: pd.DataFrame) -> None:
        """
        Save train data.

        Args:
            dataframe (pd.DataFrame): Training dataframe.
        """
        self.save_csv(dataframe, get_train_data_path())

    def save_test_data(self, dataframe: pd.DataFrame) -> None:
        """
        Save test data.

        Args:
            dataframe (pd.DataFrame): Test dataframe.
        """
        self.save_csv(dataframe, get_test_data_path())

    def save_anomaly_results(self, dataframe: pd.DataFrame) -> None:
        """
        Save anomaly detection results.

        Args:
            dataframe (pd.DataFrame): Anomaly results dataframe.
        """
        self.save_csv(dataframe, get_anomaly_results_path())

    def save_forecast_results(self, dataframe: pd.DataFrame) -> None:
        """
        Save forecast results.

        Args:
            dataframe (pd.DataFrame): Forecast results dataframe.
        """
        self.save_csv(dataframe, get_forecast_results_path())

    def load_custom_data(
        self,
        file_path: str,
        parse_dates: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Load a custom CSV file using a relative or absolute path.

        Args:
            file_path (str): Custom file path.
            parse_dates (Optional[list]): Date columns to parse.

        Returns:
            pd.DataFrame: Loaded dataframe.
        """
        resolved_file_path = resolve_path(file_path)
        return self.load_csv(resolved_file_path, parse_dates=parse_dates)


def load_raw_data() -> pd.DataFrame:
    """
    Convenience function to load raw data.

    Returns:
        pd.DataFrame: Raw data.
    """
    return DataLoader().load_raw_data()


def load_cleaned_data() -> pd.DataFrame:
    """
    Convenience function to load cleaned data.

    Returns:
        pd.DataFrame: Cleaned data.
    """
    return DataLoader().load_cleaned_data()


def save_cleaned_data(dataframe: pd.DataFrame) -> None:
    """
    Convenience function to save cleaned data.

    Args:
        dataframe (pd.DataFrame): Cleaned dataframe.
    """
    DataLoader().save_cleaned_data(dataframe)


if __name__ == "__main__":
    loader = DataLoader()

    try:
        data = loader.load_raw_data()
        print("Raw data loaded successfully.")
        print(data.head())
        print(data.shape)

    except Exception as exc:
        logger.error("Data loading test failed: %s", exc, exc_info=True)