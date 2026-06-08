"""
Data cleaning utilities for the Financial Portfolio Forecasting
and Anomaly Detection project.

This module handles:
- Column standardization
- Simple CSV schema adaptation
- Duplicate removal
- Missing value handling
- Date conversion
- Numeric conversion
- Financial feature validation
- Basic outlier capping

Supported raw CSV format:
    Date, Asset, Price

The module converts it into:
    date, asset, price, asset_name, asset_type, quantity,
    asset_value, portfolio_id, portfolio_value
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from src.monitoring.logger import get_project_logger, log_dataframe_info
from src.utils.config import (
    get_categorical_missing_strategy,
    get_date_column,
    get_numeric_missing_strategy,
    get_required_columns,
    should_handle_missing_values,
    should_remove_duplicates,
)
from src.utils.exception_handler import (
    DataError,
    DataValidationError,
    MissingColumnError,
    validate_not_empty,
    validate_required_columns,
)


logger = get_project_logger()


class DataCleaner:
    """
    Clean raw financial portfolio data.
    """

    def __init__(
        self,
        date_column: Optional[str] = None,
        required_columns: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize DataCleaner.

        Args:
            date_column (Optional[str]): Date column name.
            required_columns (Optional[List[str]]): Required dataset columns.
        """
        self.date_column = date_column or get_date_column()
        self.required_columns = required_columns or get_required_columns()

    @staticmethod
    def standardize_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize dataframe column names.

        Example:
            Portfolio Value -> portfolio_value
            Date -> date
            Asset -> asset
            Price -> price

        Args:
            dataframe (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with standardized columns.
        """
        dataframe = dataframe.copy()

        dataframe.columns = (
            dataframe.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
            .str.replace("-", "_", regex=False)
            .str.replace("/", "_", regex=False)
        )

        return dataframe

    @staticmethod
    def remove_duplicates(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows.

        Args:
            dataframe (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe without duplicates.
        """
        before_rows = len(dataframe)
        dataframe = dataframe.drop_duplicates().reset_index(drop=True)
        after_rows = len(dataframe)

        logger.info("Removed %s duplicate rows.", before_rows - after_rows)

        return dataframe

    def convert_date_column(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Convert date column to datetime.

        Args:
            dataframe (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with parsed date column.

        Raises:
            DataValidationError: If date column is missing.
        """
        dataframe = dataframe.copy()

        dataframe = self.standardize_column_names(dataframe)

        normalized_date_column = (
            self.date_column.strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
        )

        if normalized_date_column not in dataframe.columns:
            raise DataValidationError(
                message=f"Date column '{normalized_date_column}' is missing.",
                details={"available_columns": list(dataframe.columns)},
            )

        self.date_column = normalized_date_column

        dataframe[self.date_column] = pd.to_datetime(
            dataframe[self.date_column],
            errors="coerce",
        )

        missing_dates = dataframe[self.date_column].isna().sum()

        if missing_dates > 0:
            logger.warning("Found %s invalid dates. Dropping those rows.", missing_dates)
            dataframe = dataframe.dropna(subset=[self.date_column])

        dataframe = dataframe.sort_values(self.date_column).reset_index(drop=True)

        return dataframe

    @staticmethod
    def convert_numeric_columns(
        dataframe: pd.DataFrame,
        numeric_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Convert selected columns to numeric.

        Args:
            dataframe (pd.DataFrame): Input dataframe.
            numeric_columns (Optional[List[str]]): Numeric columns.

        Returns:
            pd.DataFrame: Dataframe with numeric columns converted.
        """
        dataframe = dataframe.copy()

        default_numeric_columns = [
            "price",
            "quantity",
            "asset_value",
            "portfolio_value",
            "daily_return",
        ]

        columns_to_convert = numeric_columns or default_numeric_columns

        for column in columns_to_convert:
            if column in dataframe.columns:
                dataframe[column] = pd.to_numeric(
                    dataframe[column],
                    errors="coerce",
                )

        return dataframe

    @staticmethod
    def detect_asset_type(asset_name: str) -> str:
        """
        Detect asset type from asset name.

        Args:
            asset_name (str): Asset name.

        Returns:
            str: Detected asset type.
        """
        asset_name = str(asset_name).lower()

        if "stock" in asset_name or "share" in asset_name or "equity" in asset_name:
            return "stock"

        if "bond" in asset_name:
            return "bond"

        if "crypto" in asset_name or "bitcoin" in asset_name or "ethereum" in asset_name:
            return "crypto"

        if "fund" in asset_name or "etf" in asset_name or "mutual" in asset_name:
            return "fund"

        if "gold" in asset_name or "silver" in asset_name or "commodity" in asset_name:
            return "commodity"

        return "stock"

    @staticmethod
    def adapt_simple_portfolio_csv(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Adapt simple CSV format into full portfolio format.

        Input CSV columns:
            date, asset, price

        Required project columns created:
            asset_name, asset_type, quantity, asset_value,
            portfolio_id, portfolio_value

        Args:
            dataframe (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe adapted to portfolio schema.
        """
        dataframe = dataframe.copy()

        simple_columns = {"date", "asset", "price"}

        if simple_columns.issubset(set(dataframe.columns)):
            logger.info(
                "Simple CSV format detected. Converting date, asset, price "
                "into full portfolio schema."
            )

            dataframe["asset_name"] = dataframe["asset"].astype(str)
            dataframe["asset_type"] = dataframe["asset_name"].apply(
                DataCleaner.detect_asset_type
            )

            dataframe["quantity"] = 1

            dataframe["asset_value"] = pd.to_numeric(
                dataframe["price"],
                errors="coerce",
            )

            dataframe["portfolio_id"] = "PORTFOLIO_001"

            dataframe["portfolio_value"] = dataframe.groupby(
                ["portfolio_id", "date"]
            )["asset_value"].transform("sum")

        return dataframe

    @staticmethod
    def handle_missing_values(
        dataframe: pd.DataFrame,
        numeric_strategy: Optional[str] = None,
        categorical_strategy: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Handle missing values in numerical and categorical columns.

        Args:
            dataframe (pd.DataFrame): Input dataframe.
            numeric_strategy (Optional[str]): mean, median, or zero.
            categorical_strategy (Optional[str]): mode or unknown.

        Returns:
            pd.DataFrame: Dataframe with missing values handled.
        """
        dataframe = dataframe.copy()

        numeric_strategy = numeric_strategy or get_numeric_missing_strategy()
        categorical_strategy = (
            categorical_strategy or get_categorical_missing_strategy()
        )

        numeric_columns = dataframe.select_dtypes(include=["number"]).columns
        categorical_columns = dataframe.select_dtypes(
            include=["object", "category", "bool"]
        ).columns

        for column in numeric_columns:
            if dataframe[column].isna().sum() == 0:
                continue

            if numeric_strategy == "mean":
                fill_value = dataframe[column].mean()
            elif numeric_strategy == "zero":
                fill_value = 0
            else:
                fill_value = dataframe[column].median()

            dataframe[column] = dataframe[column].fillna(fill_value)

        for column in categorical_columns:
            if dataframe[column].isna().sum() == 0:
                continue

            if categorical_strategy == "unknown":
                fill_value = "Unknown"
            else:
                mode_values = dataframe[column].mode()
                fill_value = mode_values.iloc[0] if not mode_values.empty else "Unknown"

            dataframe[column] = dataframe[column].fillna(fill_value)

        return dataframe

    @staticmethod
    def create_quantity_if_missing(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Create quantity column if missing.

        Args:
            dataframe (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with quantity column.
        """
        dataframe = dataframe.copy()

        if "quantity" not in dataframe.columns:
            dataframe["quantity"] = 1
            logger.info("Created quantity column with default value 1.")

        return dataframe

    @staticmethod
    def create_asset_name_if_missing(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Create asset_name column if missing.

        Args:
            dataframe (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with asset_name column.
        """
        dataframe = dataframe.copy()

        if "asset_name" not in dataframe.columns and "asset" in dataframe.columns:
            dataframe["asset_name"] = dataframe["asset"].astype(str)
            logger.info("Created asset_name column from asset column.")

        return dataframe

    @staticmethod
    def create_asset_type_if_missing(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Create asset_type column if missing.

        Args:
            dataframe (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with asset_type column.
        """
        dataframe = dataframe.copy()

        if "asset_type" not in dataframe.columns:
            if "asset_name" in dataframe.columns:
                dataframe["asset_type"] = dataframe["asset_name"].apply(
                    DataCleaner.detect_asset_type
                )
            else:
                dataframe["asset_type"] = "stock"

            logger.info("Created asset_type column.")

        return dataframe

    @staticmethod
    def create_portfolio_id_if_missing(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Create portfolio_id column if missing.

        Args:
            dataframe (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with portfolio_id column.
        """
        dataframe = dataframe.copy()

        if "portfolio_id" not in dataframe.columns:
            dataframe["portfolio_id"] = "PORTFOLIO_001"
            logger.info("Created portfolio_id column with default value PORTFOLIO_001.")

        return dataframe

    @staticmethod
    def create_asset_value_if_missing(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Create asset_value column if missing.

        Priority:
        1. price * quantity
        2. price

        Args:
            dataframe (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with asset_value column.
        """
        dataframe = dataframe.copy()

        if "asset_value" not in dataframe.columns:
            if "price" in dataframe.columns and "quantity" in dataframe.columns:
                dataframe["asset_value"] = dataframe["price"] * dataframe["quantity"]
                logger.info("Created asset_value column using price * quantity.")
            elif "price" in dataframe.columns:
                dataframe["asset_value"] = dataframe["price"]
                logger.info("Created asset_value column using price.")

        return dataframe

    @staticmethod
    def create_portfolio_value_if_missing(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Create portfolio_value column if missing.

        If portfolio_id and date exist, portfolio value is calculated
        as total asset value per portfolio per day.

        Args:
            dataframe (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with portfolio_value column.
        """
        dataframe = dataframe.copy()

        required = {"portfolio_id", "date", "asset_value"}

        if "portfolio_value" not in dataframe.columns and required.issubset(dataframe.columns):
            dataframe["portfolio_value"] = dataframe.groupby(
                ["portfolio_id", "date"]
            )["asset_value"].transform("sum")

            logger.info("Created portfolio_value column from daily asset values.")

        return dataframe

    @staticmethod
    def cap_outliers_iqr(
        dataframe: pd.DataFrame,
        columns: Optional[List[str]] = None,
        multiplier: float = 1.5,
    ) -> pd.DataFrame:
        """
        Cap numerical outliers using IQR method.

        Args:
            dataframe (pd.DataFrame): Input dataframe.
            columns (Optional[List[str]]): Columns to cap.
            multiplier (float): IQR multiplier.

        Returns:
            pd.DataFrame: Dataframe with capped outliers.
        """
        dataframe = dataframe.copy()

        if columns is None:
            columns = dataframe.select_dtypes(include=["number"]).columns.tolist()

        for column in columns:
            if column not in dataframe.columns:
                continue

            q1 = dataframe[column].quantile(0.25)
            q3 = dataframe[column].quantile(0.75)
            iqr = q3 - q1

            if iqr == 0 or pd.isna(iqr):
                continue

            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr

            dataframe[column] = np.where(
                dataframe[column] < lower_bound,
                lower_bound,
                dataframe[column],
            )

            dataframe[column] = np.where(
                dataframe[column] > upper_bound,
                upper_bound,
                dataframe[column],
            )

        return dataframe

    @staticmethod
    def remove_invalid_financial_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Remove invalid financial rows.

        Removes rows where price, quantity, asset_value, or portfolio_value
        are negative when those columns exist.

        Args:
            dataframe (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Valid dataframe.
        """
        dataframe = dataframe.copy()

        financial_columns = [
            "price",
            "quantity",
            "asset_value",
            "portfolio_value",
        ]

        before_rows = len(dataframe)

        for column in financial_columns:
            if column in dataframe.columns:
                dataframe = dataframe[dataframe[column] >= 0]

        after_rows = len(dataframe)

        logger.info("Removed %s invalid financial rows.", before_rows - after_rows)

        return dataframe.reset_index(drop=True)

    def validate_columns(self, dataframe: pd.DataFrame) -> None:
        """
        Validate required columns.

        Args:
            dataframe (pd.DataFrame): Input dataframe.

        Raises:
            MissingColumnError: If required columns are missing.
        """
        if self.required_columns:
            validate_required_columns(
                available_columns=dataframe.columns,
                required_columns=self.required_columns,
            )

    @staticmethod
    def sort_cleaned_data(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Sort cleaned data by date and asset name.

        Args:
            dataframe (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Sorted dataframe.
        """
        dataframe = dataframe.copy()

        sort_columns = []

        if "date" in dataframe.columns:
            sort_columns.append("date")

        if "asset_name" in dataframe.columns:
            sort_columns.append("asset_name")

        if sort_columns:
            dataframe = dataframe.sort_values(sort_columns).reset_index(drop=True)

        return dataframe

    def clean_data(
        self,
        dataframe: pd.DataFrame,
        cap_outliers: bool = False,
    ) -> pd.DataFrame:
        """
        Run full data cleaning pipeline.

        Args:
            dataframe (pd.DataFrame): Raw dataframe.
            cap_outliers (bool): Whether to cap outliers.

        Returns:
            pd.DataFrame: Cleaned dataframe.
        """
        try:
            logger.info("Starting data cleaning pipeline.")
            validate_not_empty(dataframe, name="raw dataframe")

            dataframe = self.standardize_column_names(dataframe)
            dataframe = self.convert_date_column(dataframe)
            dataframe = self.convert_numeric_columns(dataframe)

            dataframe = self.adapt_simple_portfolio_csv(dataframe)

            dataframe = self.create_quantity_if_missing(dataframe)
            dataframe = self.create_asset_name_if_missing(dataframe)
            dataframe = self.create_asset_type_if_missing(dataframe)
            dataframe = self.create_portfolio_id_if_missing(dataframe)
            dataframe = self.create_asset_value_if_missing(dataframe)
            dataframe = self.create_portfolio_value_if_missing(dataframe)

            self.validate_columns(dataframe)

            if should_remove_duplicates():
                dataframe = self.remove_duplicates(dataframe)

            dataframe = self.remove_invalid_financial_rows(dataframe)

            if should_handle_missing_values():
                dataframe = self.handle_missing_values(dataframe)

            if cap_outliers:
                dataframe = self.cap_outliers_iqr(dataframe)

            dataframe = self.sort_cleaned_data(dataframe)

            validate_not_empty(dataframe, name="cleaned dataframe")

            logger.info("Data cleaning completed successfully.")
            log_dataframe_info(logger, dataframe, "Cleaned Data")

            return dataframe

        except MissingColumnError:
            raise

        except Exception as error:
            logger.error("Data cleaning failed: %s", error, exc_info=True)
            raise DataError(
                message="Data cleaning failed.",
                details={"error": str(error)},
            ) from error


def clean_financial_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to clean financial portfolio data.

    Args:
        dataframe (pd.DataFrame): Raw dataframe.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    cleaner = DataCleaner()
    return cleaner.clean_data(dataframe)


if __name__ == "__main__":
    from src.data_processing.data_loader import DataLoader

    loader = DataLoader()
    cleaner = DataCleaner()

    raw_data = loader.load_raw_data()
    cleaned_data = cleaner.clean_data(raw_data)
    loader.save_cleaned_data(cleaned_data)

    print("Data cleaning completed and saved.")