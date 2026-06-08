"""
Train-test split utilities for the Financial Portfolio Forecasting
and Anomaly Detection project.

This module handles:
- Time-based train/test split
- Optional validation split
- Saving train, validation, and test datasets
- Preventing time series data leakage
"""

from typing import Optional, Tuple

import pandas as pd

from src.monitoring.logger import get_project_logger, log_dataframe_info
from src.utils.config import (
    get_date_column,
    get_test_size,
    get_validation_size,
    should_shuffle_split,
)
from src.utils.exception_handler import DataError, DataValidationError, validate_not_empty


logger = get_project_logger()


class TimeSeriesSplitter:
    """
    Time-based splitter for financial portfolio forecasting data.
    """

    def __init__(
        self,
        date_column: Optional[str] = None,
        test_size: Optional[float] = None,
        validation_size: Optional[float] = None,
        shuffle: Optional[bool] = None,
    ) -> None:
        """
        Initialize TimeSeriesSplitter.

        Args:
            date_column (Optional[str]): Date column name.
            test_size (Optional[float]): Test data ratio.
            validation_size (Optional[float]): Validation data ratio.
            shuffle (Optional[bool]): Whether to shuffle data.
        """
        self.date_column = date_column or get_date_column()
        self.test_size = test_size if test_size is not None else get_test_size()
        self.validation_size = (
            validation_size if validation_size is not None else get_validation_size()
        )
        self.shuffle = shuffle if shuffle is not None else should_shuffle_split()

    def validate_split_parameters(self) -> None:
        """
        Validate split parameters.

        Raises:
            DataValidationError: If split parameters are invalid.
        """
        if not 0 < self.test_size < 1:
            raise DataValidationError(
                message="test_size must be between 0 and 1.",
                details={"test_size": self.test_size},
            )

        if not 0 <= self.validation_size < 1:
            raise DataValidationError(
                message="validation_size must be between 0 and 1.",
                details={"validation_size": self.validation_size},
            )

        if self.test_size + self.validation_size >= 1:
            raise DataValidationError(
                message="test_size + validation_size must be less than 1.",
                details={
                    "test_size": self.test_size,
                    "validation_size": self.validation_size,
                },
            )

    def sort_by_date(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Sort dataframe by date.

        Args:
            dataframe (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Sorted dataframe.
        """
        dataframe = dataframe.copy()

        if self.date_column not in dataframe.columns:
            raise DataValidationError(
                message=f"Date column '{self.date_column}' is missing.",
                details={"available_columns": list(dataframe.columns)},
            )

        dataframe[self.date_column] = pd.to_datetime(
            dataframe[self.date_column],
            errors="coerce",
        )

        dataframe = dataframe.dropna(subset=[self.date_column])
        dataframe = dataframe.sort_values(self.date_column).reset_index(drop=True)

        return dataframe

    def split_train_test(
        self,
        dataframe: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets.

        Args:
            dataframe (pd.DataFrame): Input dataframe.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Train and test dataframes.
        """
        try:
            logger.info("Starting train-test split.")

            validate_not_empty(dataframe, name="input dataframe")
            self.validate_split_parameters()

            dataframe = dataframe.copy()

            if self.shuffle:
                dataframe = dataframe.sample(
                    frac=1,
                    random_state=42,
                ).reset_index(drop=True)
                logger.warning(
                    "Shuffle is enabled. For time series forecasting, "
                    "shuffle=False is recommended to prevent data leakage."
                )
            else:
                dataframe = self.sort_by_date(dataframe)

            split_index = int(len(dataframe) * (1 - self.test_size))

            train_data = dataframe.iloc[:split_index].copy()
            test_data = dataframe.iloc[split_index:].copy()

            validate_not_empty(train_data, name="train data")
            validate_not_empty(test_data, name="test data")

            logger.info(
                "Train-test split completed. Train shape: %s | Test shape: %s",
                train_data.shape,
                test_data.shape,
            )

            return train_data, test_data

        except Exception as error:
            logger.error("Train-test split failed: %s", error, exc_info=True)
            raise DataError(
                message="Train-test split failed.",
                details={"error": str(error)},
            ) from error

    def split_train_validation_test(
        self,
        dataframe: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.

        Args:
            dataframe (pd.DataFrame): Input dataframe.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            Train, validation, and test dataframes.
        """
        try:
            logger.info("Starting train-validation-test split.")

            validate_not_empty(dataframe, name="input dataframe")
            self.validate_split_parameters()

            dataframe = dataframe.copy()

            if self.shuffle:
                dataframe = dataframe.sample(
                    frac=1,
                    random_state=42,
                ).reset_index(drop=True)
                logger.warning(
                    "Shuffle is enabled. For time series forecasting, "
                    "shuffle=False is recommended."
                )
            else:
                dataframe = self.sort_by_date(dataframe)

            total_rows = len(dataframe)

            train_end = int(total_rows * (1 - self.validation_size - self.test_size))
            validation_end = int(total_rows * (1 - self.test_size))

            train_data = dataframe.iloc[:train_end].copy()
            validation_data = dataframe.iloc[train_end:validation_end].copy()
            test_data = dataframe.iloc[validation_end:].copy()

            validate_not_empty(train_data, name="train data")
            validate_not_empty(validation_data, name="validation data")
            validate_not_empty(test_data, name="test data")

            logger.info(
                "Train-validation-test split completed. "
                "Train: %s | Validation: %s | Test: %s",
                train_data.shape,
                validation_data.shape,
                test_data.shape,
            )

            return train_data, validation_data, test_data

        except Exception as error:
            logger.error(
                "Train-validation-test split failed: %s",
                error,
                exc_info=True,
            )
            raise DataError(
                message="Train-validation-test split failed.",
                details={"error": str(error)},
            ) from error

    @staticmethod
    def get_split_summary(
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None,
        date_column: str = "date",
    ) -> dict:
        """
        Create split summary dictionary.

        Args:
            train_data (pd.DataFrame): Train dataframe.
            test_data (pd.DataFrame): Test dataframe.
            validation_data (Optional[pd.DataFrame]): Validation dataframe.
            date_column (str): Date column name.

        Returns:
            dict: Split summary.
        """
        summary = {
            "train_rows": len(train_data),
            "test_rows": len(test_data),
            "total_rows": len(train_data) + len(test_data),
        }

        if date_column in train_data.columns:
            summary["train_start_date"] = str(train_data[date_column].min())
            summary["train_end_date"] = str(train_data[date_column].max())

        if date_column in test_data.columns:
            summary["test_start_date"] = str(test_data[date_column].min())
            summary["test_end_date"] = str(test_data[date_column].max())

        if validation_data is not None:
            summary["validation_rows"] = len(validation_data)
            summary["total_rows"] += len(validation_data)

            if date_column in validation_data.columns:
                summary["validation_start_date"] = str(
                    validation_data[date_column].min()
                )
                summary["validation_end_date"] = str(
                    validation_data[date_column].max()
                )

        return summary

    def save_splits(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Save train, validation, and test datasets.

        Args:
            train_data (pd.DataFrame): Train dataframe.
            test_data (pd.DataFrame): Test dataframe.
            validation_data (Optional[pd.DataFrame]): Validation dataframe.
        """
        from src.data_processing.data_loader import DataLoader
        from src.utils.paths import get_path

        loader = DataLoader()

        loader.save_train_data(train_data)
        loader.save_test_data(test_data)

        if validation_data is not None:
            validation_path = get_path("data", "validation")
            loader.save_csv(validation_data, validation_path)

        logger.info("Split datasets saved successfully.")


def split_train_test(
    dataframe: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function for train-test split.

    Args:
        dataframe (pd.DataFrame): Input dataframe.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and test dataframes.
    """
    splitter = TimeSeriesSplitter()
    return splitter.split_train_test(dataframe)


def split_train_validation_test(
    dataframe: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function for train-validation-test split.

    Args:
        dataframe (pd.DataFrame): Input dataframe.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        Train, validation, and test dataframes.
    """
    splitter = TimeSeriesSplitter()
    return splitter.split_train_validation_test(dataframe)


if __name__ == "__main__":
    from src.data_processing.data_loader import DataLoader

    loader = DataLoader()
    splitter = TimeSeriesSplitter()

    cleaned_data = loader.load_cleaned_data()

    train, validation, test = splitter.split_train_validation_test(cleaned_data)

    splitter.save_splits(
        train_data=train,
        validation_data=validation,
        test_data=test,
    )

    summary = splitter.get_split_summary(
        train_data=train,
        validation_data=validation,
        test_data=test,
        date_column=splitter.date_column,
    )

    print("Data split completed successfully.")
    print(summary)