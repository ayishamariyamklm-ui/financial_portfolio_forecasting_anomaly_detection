"""
Run Data Pipeline Script

Financial Portfolio Forecasting & Anomaly Detection

This script runs the complete data preparation pipeline:

1. Load raw financial portfolio data
2. Clean raw data
3. Create time-series and portfolio features
4. Split data into train, validation, and test sets
5. Save processed datasets

Run from project root:

    python scripts/run_data_pipeline.py
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd


# ============================================================
# Add Project Root to Python Path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ============================================================
# Project Imports
# ============================================================

from src.data_processing.data_cleaning import DataCleaner
from src.data_processing.data_loader import DataLoader
from src.data_processing.feature_engineering import FeatureEngineer
from src.data_processing.train_test_split import TimeSeriesSplitter
from src.monitoring.logger import (
    get_project_logger,
    log_dataframe_info,
    log_section,
    log_step_end,
    log_step_error,
    log_step_start,
)
from src.monitoring.performance_tracker import (
    get_performance_summary,
    save_performance_reports,
    track_step,
)
from src.utils.exception_handler import FinancialPortfolioException
from src.utils.paths import (
    create_required_directories,
    get_path,
)


# ============================================================
# Logger
# ============================================================

logger = get_project_logger()


# ============================================================
# Data Pipeline Class
# ============================================================

class DataPipelineRunner:
    """
    Run the complete data pipeline for the financial portfolio project.
    """

    def __init__(
        self,
        save_validation: bool = True,
        cap_outliers: bool = False,
        encode_categoricals: bool = False,
    ) -> None:
        """
        Initialize DataPipelineRunner.

        Args:
            save_validation (bool): Whether to create and save validation data.
            cap_outliers (bool): Whether to cap outliers using IQR during cleaning.
            encode_categoricals (bool): Whether to one-hot encode categorical columns.
        """
        self.save_validation = save_validation
        self.cap_outliers = cap_outliers
        self.encode_categoricals = encode_categoricals

        self.loader = DataLoader()
        self.cleaner = DataCleaner()
        self.engineer = FeatureEngineer()
        self.splitter = TimeSeriesSplitter()

    def setup_directories(self) -> None:
        """
        Create all required project directories.
        """
        log_step_start(logger, "Create required directories")

        with track_step("create_required_directories", category="data_pipeline"):
            create_required_directories()

        log_step_end(logger, "Create required directories")

    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw financial portfolio dataset.

        Returns:
            pd.DataFrame: Raw dataframe.
        """
        log_step_start(logger, "Load raw data")

        with track_step("load_raw_data", category="data_pipeline"):
            raw_data = self.loader.load_raw_data()

        log_dataframe_info(logger, raw_data, "Raw Data")
        log_step_end(logger, "Load raw data")

        return raw_data

    def clean_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw financial portfolio data.

        Args:
            raw_data (pd.DataFrame): Raw dataframe.

        Returns:
            pd.DataFrame: Cleaned dataframe.
        """
        log_step_start(logger, "Clean data")

        with track_step("clean_data", category="data_pipeline"):
            cleaned_data = self.cleaner.clean_data(
                dataframe=raw_data,
                cap_outliers=self.cap_outliers,
            )

            self.loader.save_cleaned_data(cleaned_data)

        log_dataframe_info(logger, cleaned_data, "Cleaned Data")
        log_step_end(logger, "Clean data")

        return cleaned_data

    def engineer_features(self, cleaned_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create feature-engineered dataset.

        Args:
            cleaned_data (pd.DataFrame): Cleaned dataframe.

        Returns:
            pd.DataFrame: Feature-engineered dataframe.
        """
        log_step_start(logger, "Feature engineering")

        with track_step("feature_engineering", category="data_pipeline"):
            feature_data = self.engineer.engineer_features(
                dataframe=cleaned_data,
                encode_categoricals=self.encode_categoricals,
            )

            feature_data_path = get_path("data", "processed_dir") / "feature_engineered_data.csv"
            self.loader.save_csv(feature_data, feature_data_path)

        log_dataframe_info(logger, feature_data, "Feature Engineered Data")
        log_step_end(logger, "Feature engineering")

        return feature_data

    def split_data(
        self,
        feature_data: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
        """
        Split feature-engineered data into train, validation, and test sets.

        Args:
            feature_data (pd.DataFrame): Feature-engineered dataframe.

        Returns:
            Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
            Train, validation, and test dataframes.
        """
        log_step_start(logger, "Split data")

        with track_step("split_data", category="data_pipeline"):
            if self.save_validation:
                train_data, validation_data, test_data = (
                    self.splitter.split_train_validation_test(feature_data)
                )

                self.splitter.save_splits(
                    train_data=train_data,
                    validation_data=validation_data,
                    test_data=test_data,
                )

            else:
                train_data, test_data = self.splitter.split_train_test(feature_data)
                validation_data = None

                self.splitter.save_splits(
                    train_data=train_data,
                    test_data=test_data,
                )

        log_dataframe_info(logger, train_data, "Train Data")

        if validation_data is not None:
            log_dataframe_info(logger, validation_data, "Validation Data")

        log_dataframe_info(logger, test_data, "Test Data")

        log_step_end(logger, "Split data")

        return train_data, validation_data, test_data

    def generate_split_summary(
        self,
        train_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame],
        test_data: pd.DataFrame,
    ) -> Dict:
        """
        Generate and save split summary.

        Args:
            train_data (pd.DataFrame): Train dataframe.
            validation_data (Optional[pd.DataFrame]): Validation dataframe.
            test_data (pd.DataFrame): Test dataframe.

        Returns:
            Dict: Split summary dictionary.
        """
        log_step_start(logger, "Generate split summary")

        with track_step("generate_split_summary", category="data_pipeline"):
            summary = self.splitter.get_split_summary(
                train_data=train_data,
                validation_data=validation_data,
                test_data=test_data,
                date_column=self.splitter.date_column,
            )

            summary_path = get_path("reports", "tables_dir") / "data_split_summary.csv"
            summary_dataframe = pd.DataFrame([summary])
            self.loader.save_csv(summary_dataframe, summary_path)

        logger.info("Data split summary: %s", summary)
        log_step_end(logger, "Generate split summary")

        return summary

    def run(self) -> Dict:
        """
        Run the full data pipeline.

        Returns:
            Dict: Pipeline summary.
        """
        log_section(logger, "Starting Financial Portfolio Data Pipeline")

        try:
            self.setup_directories()

            raw_data = self.load_raw_data()

            cleaned_data = self.clean_data(raw_data)

            feature_data = self.engineer_features(cleaned_data)

            train_data, validation_data, test_data = self.split_data(feature_data)

            split_summary = self.generate_split_summary(
                train_data=train_data,
                validation_data=validation_data,
                test_data=test_data,
            )

            save_performance_reports()

            performance_summary = get_performance_summary()

            pipeline_summary = {
                "status": "success",
                "raw_rows": len(raw_data),
                "cleaned_rows": len(cleaned_data),
                "feature_rows": len(feature_data),
                "feature_columns": len(feature_data.columns),
                "train_rows": len(train_data),
                "validation_rows": len(validation_data) if validation_data is not None else 0,
                "test_rows": len(test_data),
                "split_summary": split_summary,
                "performance_summary": performance_summary,
            }

            log_section(logger, "Data Pipeline Completed Successfully")
            logger.info("Pipeline summary: %s", pipeline_summary)

            return pipeline_summary

        except FinancialPortfolioException as error:
            log_step_error(logger, "Data pipeline", error)
            save_performance_reports()

            return {
                "status": "failed",
                "error_type": error.__class__.__name__,
                "error_message": str(error),
            }

        except Exception as error:
            log_step_error(logger, "Data pipeline", error)
            save_performance_reports()

            return {
                "status": "failed",
                "error_type": error.__class__.__name__,
                "error_message": str(error),
            }


# ============================================================
# Main Function
# ============================================================

def main() -> None:
    """
    Main function to run the data pipeline.
    """
    runner = DataPipelineRunner(
        save_validation=True,
        cap_outliers=False,
        encode_categoricals=False,
    )

    summary = runner.run()

    print("\n" + "=" * 80)
    print("FINANCIAL PORTFOLIO DATA PIPELINE SUMMARY")
    print("=" * 80)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("=" * 80)

    if summary.get("status") == "success":
        print("\nData pipeline completed successfully.")
        print("Generated files:")
        print("- data/processed/cleaned_financial_data.csv")
        print("- data/processed/feature_engineered_data.csv")
        print("- data/processed/train_data.csv")
        print("- data/processed/validation_data.csv")
        print("- data/processed/test_data.csv")
        print("- reports/tables/data_split_summary.csv")
        print("- reports/tables/performance_metrics.json")
        print("- reports/tables/performance_metrics.csv")
    else:
        print("\nData pipeline failed. Check logs/app.log for details.")


# ============================================================
# Script Entry Point
# ============================================================

if __name__ == "__main__":
    main()