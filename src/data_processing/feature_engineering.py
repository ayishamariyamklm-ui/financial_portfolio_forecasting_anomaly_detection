"""
Feature engineering utilities for the Financial Portfolio Forecasting
and Anomaly Detection project.

This module creates:
- Daily returns
- Lag features
- Rolling mean/std features
- Time-based features
- Portfolio-level summary features
- Clean model-ready feature dataset
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from src.monitoring.logger import get_project_logger, log_dataframe_info
from src.utils.config import (
    get_date_column,
    get_lag_periods,
    get_rolling_windows,
    get_target_column,
)
from src.utils.exception_handler import (
    DataError,
    DataValidationError,
    validate_not_empty,
)


logger = get_project_logger()


class FeatureEngineer:
    """
    Create time series and financial portfolio features.
    """

    def __init__(
        self,
        date_column: Optional[str] = None,
        target_column: Optional[str] = None,
        group_column: str = "portfolio_id",
    ) -> None:
        """
        Initialize FeatureEngineer.

        Args:
            date_column (Optional[str]): Date column name.
            target_column (Optional[str]): Target column name.
            group_column (str): Portfolio grouping column.
        """
        self.date_column = date_column or get_date_column()
        self.target_column = target_column or get_target_column()
        self.group_column = group_column

    def sort_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Sort data by portfolio and date.

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

        sort_columns = [self.date_column]

        if self.group_column in dataframe.columns:
            sort_columns = [self.group_column, self.date_column]

        dataframe = dataframe.sort_values(sort_columns).reset_index(drop=True)

        return dataframe

    def create_daily_returns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Create daily return feature from portfolio value.

        Args:
            dataframe (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with daily_return.
        """
        dataframe = dataframe.copy()

        if self.target_column not in dataframe.columns:
            raise DataValidationError(
                message=f"Target column '{self.target_column}' is missing.",
                details={"available_columns": list(dataframe.columns)},
            )

        if self.group_column in dataframe.columns:
            dataframe["daily_return"] = dataframe.groupby(self.group_column)[
                self.target_column
            ].pct_change()
        else:
            dataframe["daily_return"] = dataframe[self.target_column].pct_change()

        dataframe["daily_return"] = dataframe["daily_return"].replace(
            [np.inf, -np.inf],
            np.nan,
        )

        dataframe["daily_return"] = dataframe["daily_return"].fillna(0)

        return dataframe

    def create_lag_features(
        self,
        dataframe: pd.DataFrame,
        columns: Optional[List[str]] = None,
        lag_periods: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Create lag features.

        Args:
            dataframe (pd.DataFrame): Input dataframe.
            columns (Optional[List[str]]): Columns to lag.
            lag_periods (Optional[List[int]]): Lag periods.

        Returns:
            pd.DataFrame: Dataframe with lag features.
        """
        dataframe = dataframe.copy()

        columns = columns or [self.target_column, "daily_return"]
        lag_periods = lag_periods or get_lag_periods()

        for column in columns:
            if column not in dataframe.columns:
                continue

            for lag in lag_periods:
                feature_name = f"{column}_lag_{lag}"

                if self.group_column in dataframe.columns:
                    dataframe[feature_name] = dataframe.groupby(self.group_column)[
                        column
                    ].shift(lag)
                else:
                    dataframe[feature_name] = dataframe[column].shift(lag)

        return dataframe

    def create_rolling_features(
        self,
        dataframe: pd.DataFrame,
        columns: Optional[List[str]] = None,
        rolling_windows: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Create rolling mean and standard deviation features.

        Args:
            dataframe (pd.DataFrame): Input dataframe.
            columns (Optional[List[str]]): Columns for rolling features.
            rolling_windows (Optional[List[int]]): Rolling windows.

        Returns:
            pd.DataFrame: Dataframe with rolling features.
        """
        dataframe = dataframe.copy()

        columns = columns or [self.target_column, "daily_return"]
        rolling_windows = rolling_windows or get_rolling_windows()

        for column in columns:
            if column not in dataframe.columns:
                continue

            for window in rolling_windows:
                mean_feature = f"{column}_rolling_mean_{window}"
                std_feature = f"{column}_rolling_std_{window}"

                if self.group_column in dataframe.columns:
                    dataframe[mean_feature] = dataframe.groupby(self.group_column)[
                        column
                    ].transform(
                        lambda series: series.rolling(
                            window=window,
                            min_periods=1,
                        ).mean()
                    )

                    dataframe[std_feature] = dataframe.groupby(self.group_column)[
                        column
                    ].transform(
                        lambda series: series.rolling(
                            window=window,
                            min_periods=1,
                        ).std()
                    )
                else:
                    dataframe[mean_feature] = dataframe[column].rolling(
                        window=window,
                        min_periods=1,
                    ).mean()

                    dataframe[std_feature] = dataframe[column].rolling(
                        window=window,
                        min_periods=1,
                    ).std()

        return dataframe

    def create_time_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Create calendar-based time features.

        Args:
            dataframe (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with time features.
        """
        dataframe = dataframe.copy()

        dataframe[self.date_column] = pd.to_datetime(
            dataframe[self.date_column],
            errors="coerce",
        )

        dataframe["day"] = dataframe[self.date_column].dt.day
        dataframe["month"] = dataframe[self.date_column].dt.month
        dataframe["quarter"] = dataframe[self.date_column].dt.quarter
        dataframe["year"] = dataframe[self.date_column].dt.year
        dataframe["day_of_week"] = dataframe[self.date_column].dt.dayofweek
        dataframe["week_of_year"] = dataframe[self.date_column].dt.isocalendar().week.astype(int)
        dataframe["is_month_start"] = dataframe[self.date_column].dt.is_month_start.astype(int)
        dataframe["is_month_end"] = dataframe[self.date_column].dt.is_month_end.astype(int)
        dataframe["is_quarter_start"] = dataframe[self.date_column].dt.is_quarter_start.astype(int)
        dataframe["is_quarter_end"] = dataframe[self.date_column].dt.is_quarter_end.astype(int)

        return dataframe

    @staticmethod
    def create_asset_features(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Create asset-level features.

        Args:
            dataframe (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with asset features.
        """
        dataframe = dataframe.copy()

        if "asset_value" in dataframe.columns and "portfolio_value" in dataframe.columns:
            dataframe["asset_weight"] = dataframe["asset_value"] / dataframe[
                "portfolio_value"
            ].replace(0, np.nan)

            dataframe["asset_weight"] = dataframe["asset_weight"].replace(
                [np.inf, -np.inf],
                np.nan,
            ).fillna(0)

        if "price" in dataframe.columns and "quantity" in dataframe.columns:
            dataframe["price_quantity_interaction"] = (
                dataframe["price"] * dataframe["quantity"]
            )

        return dataframe

    def create_portfolio_aggregate_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create portfolio-level aggregate features per day.

        Args:
            dataframe (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with aggregate features.
        """
        dataframe = dataframe.copy()

        required_columns = {self.date_column, self.group_column, "asset_value"}

        if not required_columns.issubset(set(dataframe.columns)):
            return dataframe

        aggregate_data = dataframe.groupby(
            [self.group_column, self.date_column],
            as_index=False,
        ).agg(
            total_asset_value=("asset_value", "sum"),
            average_asset_value=("asset_value", "mean"),
            max_asset_value=("asset_value", "max"),
            min_asset_value=("asset_value", "min"),
            asset_count=("asset_value", "count"),
        )

        dataframe = dataframe.merge(
            aggregate_data,
            on=[self.group_column, self.date_column],
            how="left",
        )

        return dataframe

    @staticmethod
    def encode_categorical_columns(
        dataframe: pd.DataFrame,
        categorical_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        One-hot encode categorical columns.

        Args:
            dataframe (pd.DataFrame): Input dataframe.
            categorical_columns (Optional[List[str]]): Categorical columns.

        Returns:
            pd.DataFrame: Encoded dataframe.
        """
        dataframe = dataframe.copy()

        if categorical_columns is None:
            categorical_columns = dataframe.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.tolist()

        existing_columns = [
            column for column in categorical_columns if column in dataframe.columns
        ]

        if existing_columns:
            dataframe = pd.get_dummies(
                dataframe,
                columns=existing_columns,
                drop_first=True,
                dtype=int,
            )

        return dataframe

    @staticmethod
    def clean_engineered_features(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Clean engineered feature values.

        Args:
            dataframe (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Cleaned dataframe.
        """
        dataframe = dataframe.copy()

        dataframe = dataframe.replace([np.inf, -np.inf], np.nan)

        numeric_columns = dataframe.select_dtypes(include=["number"]).columns

        for column in numeric_columns:
            if dataframe[column].isna().sum() > 0:
                dataframe[column] = dataframe[column].fillna(dataframe[column].median())

        dataframe = dataframe.fillna(0)

        return dataframe

    def engineer_features(
        self,
        dataframe: pd.DataFrame,
        encode_categoricals: bool = False,
    ) -> pd.DataFrame:
        """
        Run full feature engineering pipeline.

        Args:
            dataframe (pd.DataFrame): Cleaned dataframe.
            encode_categoricals (bool): Whether to one-hot encode categorical columns.

        Returns:
            pd.DataFrame: Feature-engineered dataframe.
        """
        try:
            logger.info("Starting feature engineering pipeline.")

            validate_not_empty(dataframe, name="cleaned dataframe")

            dataframe = self.sort_data(dataframe)
            dataframe = self.create_daily_returns(dataframe)
            dataframe = self.create_lag_features(dataframe)
            dataframe = self.create_rolling_features(dataframe)
            dataframe = self.create_time_features(dataframe)
            dataframe = self.create_asset_features(dataframe)
            dataframe = self.create_portfolio_aggregate_features(dataframe)

            if encode_categoricals:
                dataframe = self.encode_categorical_columns(dataframe)

            dataframe = self.clean_engineered_features(dataframe)

            validate_not_empty(dataframe, name="feature engineered dataframe")

            logger.info("Feature engineering completed successfully.")
            log_dataframe_info(logger, dataframe, "Feature Engineered Data")

            return dataframe

        except Exception as error:
            logger.error("Feature engineering failed: %s", error, exc_info=True)
            raise DataError(
                message="Feature engineering failed.",
                details={"error": str(error)},
            ) from error


def create_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to create features.

    Args:
        dataframe (pd.DataFrame): Cleaned dataframe.

    Returns:
        pd.DataFrame: Feature-engineered dataframe.
    """
    engineer = FeatureEngineer()
    return engineer.engineer_features(dataframe)


if __name__ == "__main__":
    from src.data_processing.data_loader import DataLoader

    loader = DataLoader()
    engineer = FeatureEngineer()

    cleaned_data = loader.load_cleaned_data()
    feature_data = engineer.engineer_features(cleaned_data)

    loader.save_csv(feature_data, loader.load_cleaned_data().pipe(lambda _: __import__("pathlib").Path("data/processed/feature_engineered_data.csv")))

    print("Feature engineering completed.")