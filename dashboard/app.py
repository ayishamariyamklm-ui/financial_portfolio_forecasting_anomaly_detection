"""
Streamlit dashboard for the Financial Portfolio Forecasting
and Anomaly Detection project.

This dashboard shows:

1. Project overview
2. Dataset summary
3. Asset price trends
4. Forecasting model comparison
5. Forecast results
6. Anomaly detection results
7. Asset-wise anomaly summary
8. API/model artifact health
9. Downloadable project outputs

Run from project root:

    streamlit run dashboard/app.py

Dataset expected columns:
- date
- asset
- price
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# Add Project Root to Python Path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ============================================================
# Project Paths
# ============================================================

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "financial_portfolio_data.csv"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"
LOGS_DIR = PROJECT_ROOT / "logs"

FORECAST_COMPARISON_PATHS = [
    TABLES_DIR / "model_comparison.csv",
    TABLES_DIR / "forecasting_model_comparison.csv",
    TABLES_DIR / "03_forecasting_model_comparison.csv",
    TABLES_DIR / "05_forecast_comparison_final.csv",
]

FORECAST_OUTPUT_PATHS = [
    OUTPUTS_DIR / "inference_forecasts.csv",
    OUTPUTS_DIR / "api_forecast_results.csv",
    OUTPUTS_DIR / "xgboost_forecast_results.csv",
    OUTPUTS_DIR / "prophet_forecast_results.csv",
    OUTPUTS_DIR / "lstm_forecast_results.csv",
    OUTPUTS_DIR / "arima_forecast_results.csv",
    OUTPUTS_DIR / "03_random_forest_future_forecast.csv",
]

PREDICTION_OUTPUT_PATHS = [
    OUTPUTS_DIR / "03_forecasting_predictions.csv",
    OUTPUTS_DIR / "xgboost_predictions.csv",
    OUTPUTS_DIR / "prophet_test_predictions.csv",
    OUTPUTS_DIR / "lstm_test_predictions.csv",
]

ANOMALY_OUTPUT_PATHS = [
    OUTPUTS_DIR / "api_anomaly_results.csv",
    OUTPUTS_DIR / "inference_anomaly_results.csv",
    OUTPUTS_DIR / "anomaly_detection_results.csv",
    OUTPUTS_DIR / "04_anomaly_detection_results.csv",
    OUTPUTS_DIR / "04_full_anomaly_detection_results.csv",
]

ASSET_ANOMALY_SUMMARY_PATHS = [
    TABLES_DIR / "05_asset_anomaly_summary_final.csv",
    TABLES_DIR / "04_asset_anomaly_summary.csv",
    TABLES_DIR / "anomaly_asset_summary.csv",
    TABLES_DIR / "anomaly_summary.csv",
]

TOP_ANOMALY_RECORDS_PATHS = [
    TABLES_DIR / "05_top_anomaly_records_final.csv",
    TABLES_DIR / "04_top_anomaly_records.csv",
    TABLES_DIR / "top_anomaly_records.csv",
]

SUMMARY_JSON_PATHS = [
    TABLES_DIR / "05_final_project_evaluation.json",
    TABLES_DIR / "evaluation_summary.json",
    TABLES_DIR / "api_summary.json",
    TABLES_DIR / "all_models_training_summary.json",
    TABLES_DIR / "inference_summary.json",
]

BEST_MODEL_PATHS = [
    TABLES_DIR / "best_forecasting_model.json",
    TABLES_DIR / "03_best_forecasting_model.json",
]

MODEL_ARTIFACT_PATHS = {
    "XGBoost Model": MODELS_DIR / "xgboost_model.pkl",
    "ARIMA Models": MODELS_DIR / "arima_models.pkl",
    "Prophet Models": MODELS_DIR / "prophet_models.pkl",
    "LSTM Models Directory": MODELS_DIR / "lstm_models",
    "LSTM Scalers": MODELS_DIR / "lstm_scalers.pkl",
    "Anomaly Detector": MODELS_DIR / "anomaly_detection_model.pkl",
}


# ============================================================
# Streamlit Page Config
# ============================================================

st.set_page_config(
    page_title="Financial Portfolio Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Utility Functions
# ============================================================

def ensure_dashboard_directories() -> None:
    """
    Create required dashboard directories.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_data(show_spinner=False)
def load_csv(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load CSV safely.

    Args:
        file_path (str): CSV path.

    Returns:
        Optional[pd.DataFrame]: Loaded dataframe or None.
    """
    path = Path(file_path)

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load JSON safely.

    Args:
        file_path (str): JSON path.

    Returns:
        Dict[str, Any]: Loaded JSON or empty dictionary.
    """
    path = Path(file_path)

    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        return data if isinstance(data, dict) else {}

    except Exception:
        return {}


def load_first_available_csv(paths: list[Path]) -> tuple[Optional[pd.DataFrame], Optional[Path]]:
    """
    Load first available CSV from list.

    Args:
        paths (list[Path]): Candidate CSV paths.

    Returns:
        tuple: Dataframe and source path.
    """
    for path in paths:
        dataframe = load_csv(str(path))

        if dataframe is not None and not dataframe.empty:
            return dataframe, path

    return None, None


def load_first_available_json(paths: list[Path]) -> tuple[Dict[str, Any], Optional[Path]]:
    """
    Load first available JSON from list.

    Args:
        paths (list[Path]): Candidate JSON paths.

    Returns:
        tuple: JSON data and source path.
    """
    for path in paths:
        data = load_json(str(path))

        if data:
            return data, path

    return {}, None


def standardize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize dataframe columns.

    Args:
        dataframe (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Standardized dataframe.
    """
    data = dataframe.copy()

    data.columns = (
        data.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    return data


def clean_price_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw financial price data.

    Args:
        dataframe (pd.DataFrame): Raw dataframe.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    data = standardize_columns(dataframe)

    required_columns = ["date", "asset", "price"]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        return pd.DataFrame()

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["price"] = pd.to_numeric(data["price"], errors="coerce")

    data = data.dropna(subset=["date", "asset", "price"])
    data = data[data["price"] >= 0]
    data = data.sort_values(["asset", "date"]).reset_index(drop=True)

    return data


def get_file_status(file_path: Path) -> Dict[str, Any]:
    """
    Get file status.

    Args:
        file_path (Path): File path.

    Returns:
        Dict[str, Any]: File status.
    """
    if not file_path.exists():
        return {
            "exists": False,
            "path": str(file_path),
            "size_kb": None,
            "modified_at": None,
        }

    return {
        "exists": True,
        "path": str(file_path),
        "size_kb": round(file_path.stat().st_size / 1024, 2),
        "modified_at": datetime.fromtimestamp(
            file_path.stat().st_mtime
        ).strftime("%Y-%m-%d %H:%M:%S"),
    }


def dataframe_download_button(
    dataframe: pd.DataFrame,
    file_name: str,
    label: str,
) -> None:
    """
    Create CSV download button.

    Args:
        dataframe (pd.DataFrame): Dataframe to download.
        file_name (str): Download file name.
        label (str): Button label.
    """
    csv_data = dataframe.to_csv(index=False).encode("utf-8")

    st.download_button(
        label=label,
        data=csv_data,
        file_name=file_name,
        mime="text/csv",
    )


def json_download_button(
    data: Dict[str, Any],
    file_name: str,
    label: str,
) -> None:
    """
    Create JSON download button.

    Args:
        data (Dict[str, Any]): JSON data.
        file_name (str): Download file name.
        label (str): Button label.
    """
    json_data = json.dumps(data, indent=4, default=str).encode("utf-8")

    st.download_button(
        label=label,
        data=json_data,
        file_name=file_name,
        mime="application/json",
    )


def show_missing_file_warning(title: str, suggested_command: str) -> None:
    """
    Show warning for missing files.

    Args:
        title (str): Warning title.
        suggested_command (str): Suggested command.
    """
    st.warning(title)
    st.code(suggested_command, language="bash")


# ============================================================
# Load Project Data
# ============================================================

ensure_dashboard_directories()

raw_dataframe = load_csv(str(DATA_PATH))
price_data = clean_price_data(raw_dataframe) if raw_dataframe is not None else pd.DataFrame()

forecast_comparison, forecast_comparison_source = load_first_available_csv(
    FORECAST_COMPARISON_PATHS
)

forecast_results, forecast_results_source = load_first_available_csv(
    FORECAST_OUTPUT_PATHS
)

prediction_results, prediction_results_source = load_first_available_csv(
    PREDICTION_OUTPUT_PATHS
)

anomaly_results, anomaly_results_source = load_first_available_csv(
    ANOMALY_OUTPUT_PATHS
)

asset_anomaly_summary, asset_anomaly_summary_source = load_first_available_csv(
    ASSET_ANOMALY_SUMMARY_PATHS
)

top_anomaly_records, top_anomaly_records_source = load_first_available_csv(
    TOP_ANOMALY_RECORDS_PATHS
)

project_summary_json, project_summary_source = load_first_available_json(
    SUMMARY_JSON_PATHS
)

best_model_json, best_model_source = load_first_available_json(
    BEST_MODEL_PATHS
)


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Overview",
        "Dataset",
        "Forecasting",
        "Anomaly Detection",
        "Model Health",
        "Downloads",
    ],
)

st.sidebar.markdown("---")

st.sidebar.subheader("Project Files")

st.sidebar.write("Dataset:")
st.sidebar.caption(str(DATA_PATH))

dataset_available = DATA_PATH.exists()

if dataset_available:
    st.sidebar.success("Dataset available")
else:
    st.sidebar.error("Dataset missing")

st.sidebar.markdown("---")

st.sidebar.subheader("Run Commands")

st.sidebar.code(
    "python scripts/run_all_models.py\n"
    "python scripts/evaluate_models.py\n"
    "streamlit run dashboard/app.py",
    language="bash",
)


# ============================================================
# Header
# ============================================================

st.title("📈 Financial Portfolio Forecasting & Anomaly Detection")
st.caption("Month 5 Practical Project | Forecasting, anomaly detection, evaluation, and reporting dashboard")

st.markdown("---")


# ============================================================
# Page: Overview
# ============================================================

if page == "Overview":
    st.header("📌 Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    total_records = len(price_data) if not price_data.empty else 0
    total_assets = price_data["asset"].nunique() if not price_data.empty else 0

    total_forecast_rows = len(forecast_results) if forecast_results is not None else 0

    if anomaly_results is not None and "is_anomaly" in standardize_columns(anomaly_results).columns:
        anomaly_temp = standardize_columns(anomaly_results)
        total_anomalies = int(anomaly_temp["is_anomaly"].astype(bool).sum())
    else:
        total_anomalies = 0

    col1.metric("Total Records", total_records)
    col2.metric("Total Assets", total_assets)
    col3.metric("Forecast Rows", total_forecast_rows)
    col4.metric("Anomalies", total_anomalies)

    st.subheader("Business Problem")

    st.write(
        """
        This project builds an end-to-end financial portfolio monitoring system.
        It analyzes historical asset prices, predicts future price movement, and detects unusual
        price behavior that may indicate risk, market shifts, or data quality problems.
        """
    )

    st.subheader("Project Workflow")

    workflow_data = pd.DataFrame(
        [
            {"step": "1", "stage": "Data Understanding", "description": "Load and inspect asset price data."},
            {"step": "2", "stage": "Time Series EDA", "description": "Analyze trends, returns, volatility, and asset behavior."},
            {"step": "3", "stage": "Forecasting", "description": "Train and compare ARIMA, XGBoost, Prophet, and LSTM models."},
            {"step": "4", "stage": "Anomaly Detection", "description": "Detect unusual records using IQR, Z-score, and Isolation Forest."},
            {"step": "5", "stage": "Evaluation", "description": "Compare metrics, select best model, and summarize results."},
            {"step": "6", "stage": "Deployment", "description": "Serve predictions through FastAPI and visualize results in Streamlit."},
        ]
    )

    st.dataframe(workflow_data, use_container_width=True, hide_index=True)

    st.subheader("Latest Project Summary")

    if project_summary_json:
        st.json(project_summary_json)
    else:
        st.info("No final project summary JSON found yet.")
        st.code("python scripts/evaluate_models.py", language="bash")


# ============================================================
# Page: Dataset
# ============================================================

elif page == "Dataset":
    st.header("🗂️ Dataset Overview")

    if price_data.empty:
        show_missing_file_warning(
            "Dataset is missing or invalid. Please place your CSV file at data/raw/financial_portfolio_data.csv.",
            "python scripts/run_data_pipeline.py",
        )
    else:
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Records", len(price_data))
        col2.metric("Assets", price_data["asset"].nunique())
        col3.metric("Start Date", str(price_data["date"].min().date()))
        col4.metric("End Date", str(price_data["date"].max().date()))

        st.subheader("Dataset Preview")
        st.dataframe(price_data.head(100), use_container_width=True)

        st.subheader("Asset Summary")

        asset_summary = (
            price_data.groupby("asset")
            .agg(
                record_count=("price", "count"),
                start_date=("date", "min"),
                end_date=("date", "max"),
                average_price=("price", "mean"),
                min_price=("price", "min"),
                max_price=("price", "max"),
                price_std=("price", "std"),
            )
            .reset_index()
        )

        st.dataframe(asset_summary, use_container_width=True, hide_index=True)

        st.subheader("Asset Price Trend")

        selected_assets = st.multiselect(
            "Select assets",
            options=sorted(price_data["asset"].unique().tolist()),
            default=sorted(price_data["asset"].unique().tolist())[:5],
        )

        filtered_price_data = price_data[
            price_data["asset"].isin(selected_assets)
        ] if selected_assets else price_data

        fig = px.line(
            filtered_price_data,
            x="date",
            y="price",
            color="asset",
            title="Asset Price Trend Over Time",
            markers=True,
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Price Distribution")

        fig = px.histogram(
            price_data,
            x="price",
            color="asset",
            nbins=40,
            title="Price Distribution by Asset",
            marginal="box",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Daily Returns")

        return_data = price_data.copy()
        return_data["daily_return"] = (
            return_data.groupby("asset")["price"]
            .pct_change()
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

        fig = px.line(
            return_data,
            x="date",
            y="daily_return",
            color="asset",
            title="Daily Return Trend",
        )

        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Page: Forecasting
# ============================================================

elif page == "Forecasting":
    st.header("🔮 Forecasting Results")

    st.subheader("Best Forecasting Model")

    if best_model_json:
        st.json(best_model_json)
        if best_model_source:
            st.caption(f"Source: {best_model_source}")
    elif forecast_comparison is not None and not forecast_comparison.empty:
        comparison_clean = standardize_columns(forecast_comparison)

        if "rmse" in comparison_clean.columns:
            comparison_clean["rmse"] = pd.to_numeric(
                comparison_clean["rmse"],
                errors="coerce",
            )
            best_row = comparison_clean.dropna(subset=["rmse"]).sort_values("rmse").head(1)

            if not best_row.empty:
                st.success(
                    f"Best model based on RMSE: {best_row.iloc[0].get('model_name')}"
                )
                st.dataframe(best_row, use_container_width=True, hide_index=True)
    else:
        st.info("No best model file or comparison table found yet.")

    st.subheader("Model Comparison")

    if forecast_comparison is not None and not forecast_comparison.empty:
        comparison_data = standardize_columns(forecast_comparison)

        st.caption(f"Source: {forecast_comparison_source}")
        st.dataframe(comparison_data, use_container_width=True, hide_index=True)

        metric_options = [
            column for column in ["mae", "mse", "rmse", "mape", "smape", "r2_score"]
            if column in comparison_data.columns
        ]

        if metric_options and "model_name" in comparison_data.columns:
            selected_metric = st.selectbox(
                "Select metric for comparison",
                options=metric_options,
                index=metric_options.index("rmse") if "rmse" in metric_options else 0,
            )

            comparison_data[selected_metric] = pd.to_numeric(
                comparison_data[selected_metric],
                errors="coerce",
            )

            plot_data = comparison_data.dropna(subset=[selected_metric])

            fig = px.bar(
                plot_data,
                x="model_name",
                y=selected_metric,
                title=f"Model Comparison by {selected_metric.upper()}",
                text=selected_metric,
            )

            st.plotly_chart(fig, use_container_width=True)

    else:
        show_missing_file_warning(
            "No forecasting model comparison file found.",
            "python scripts/run_all_models.py\npython scripts/evaluate_models.py",
        )

    st.subheader("Forecast Results")

    if forecast_results is not None and not forecast_results.empty:
        forecast_data = standardize_columns(forecast_results)

        if "date" in forecast_data.columns:
            forecast_data["date"] = pd.to_datetime(forecast_data["date"], errors="coerce")

        st.caption(f"Source: {forecast_results_source}")
        st.dataframe(forecast_data.head(200), use_container_width=True)

        forecast_column = None

        for candidate_column in ["forecasted_price", "predicted_price", "yhat"]:
            if candidate_column in forecast_data.columns:
                forecast_column = candidate_column
                break

        if forecast_column and "date" in forecast_data.columns:
            if "asset" in forecast_data.columns:
                fig = px.line(
                    forecast_data,
                    x="date",
                    y=forecast_column,
                    color="asset",
                    title="Forecasted Asset Prices",
                    markers=True,
                )
            else:
                fig = px.line(
                    forecast_data,
                    x="date",
                    y=forecast_column,
                    title="Forecasted Prices",
                    markers=True,
                )

            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No forecast result file found.")
        st.code("python src/inference/predict.py", language="bash")

    st.subheader("Actual vs Predicted")

    if prediction_results is not None and not prediction_results.empty:
        prediction_data = standardize_columns(prediction_results)

        if "date" in prediction_data.columns:
            prediction_data["date"] = pd.to_datetime(
                prediction_data["date"],
                errors="coerce",
            )

        st.caption(f"Source: {prediction_results_source}")
        st.dataframe(prediction_data.head(100), use_container_width=True)

        if "actual_price" in prediction_data.columns:
            prediction_columns = [
                column for column in prediction_data.columns
                if column.endswith("_prediction")
                or column in ["baseline_previous_price", "baseline_rolling_mean"]
            ]

            if prediction_columns:
                selected_prediction = st.selectbox(
                    "Select prediction column",
                    options=prediction_columns,
                )

                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=prediction_data["date"],
                        y=prediction_data["actual_price"],
                        mode="lines",
                        name="Actual Price",
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=prediction_data["date"],
                        y=prediction_data[selected_prediction],
                        mode="lines",
                        name=selected_prediction,
                        line={"dash": "dash"},
                    )
                )

                fig.update_layout(
                    title="Actual vs Predicted Price",
                    xaxis_title="Date",
                    yaxis_title="Price",
                )

                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No prediction result file found.")
        st.code("Run notebooks/03_forecasting_experiments.ipynb", language="text")


# ============================================================
# Page: Anomaly Detection
# ============================================================

elif page == "Anomaly Detection":
    st.header("🚨 Anomaly Detection Results")

    if anomaly_results is None or anomaly_results.empty:
        show_missing_file_warning(
            "No anomaly result file found.",
            "python src/training/train_anomaly_model.py\npython scripts/evaluate_models.py",
        )
    else:
        anomaly_data = standardize_columns(anomaly_results)

        if "date" in anomaly_data.columns:
            anomaly_data["date"] = pd.to_datetime(
                anomaly_data["date"],
                errors="coerce",
            )

        if "is_anomaly" in anomaly_data.columns:
            anomaly_data["is_anomaly"] = anomaly_data["is_anomaly"].astype(bool)

        total_records = len(anomaly_data)
        total_anomalies = int(anomaly_data["is_anomaly"].sum()) if "is_anomaly" in anomaly_data.columns else 0
        anomaly_rate = (total_anomalies / total_records * 100) if total_records > 0 else 0

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Records Checked", total_records)
        col2.metric("Total Anomalies", total_anomalies)
        col3.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
        col4.metric(
            "Assets",
            anomaly_data["asset"].nunique() if "asset" in anomaly_data.columns else 0,
        )

        st.caption(f"Source: {anomaly_results_source}")

        st.subheader("Anomaly Results Table")

        only_anomalies = st.checkbox("Show only anomaly records", value=False)

        table_data = anomaly_data.copy()

        if only_anomalies and "is_anomaly" in table_data.columns:
            table_data = table_data[table_data["is_anomaly"]]

        st.dataframe(table_data.head(500), use_container_width=True)

        st.subheader("Final Anomaly Chart")

        if {"date", "asset", "price", "is_anomaly"}.issubset(anomaly_data.columns):
            selected_assets = st.multiselect(
                "Select assets for anomaly chart",
                options=sorted(anomaly_data["asset"].unique().tolist()),
                default=sorted(anomaly_data["asset"].unique().tolist())[:5],
                key="anomaly_assets",
            )

            chart_data = anomaly_data[
                anomaly_data["asset"].isin(selected_assets)
            ] if selected_assets else anomaly_data

            fig = px.line(
                chart_data,
                x="date",
                y="price",
                color="asset",
                title="Asset Prices with Detected Anomalies",
            )

            anomaly_points = chart_data[chart_data["is_anomaly"]]

            if not anomaly_points.empty:
                fig.add_trace(
                    go.Scatter(
                        x=anomaly_points["date"],
                        y=anomaly_points["price"],
                        mode="markers",
                        marker=dict(size=10, symbol="x"),
                        name="Anomaly",
                    )
                )

            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Method-wise Anomaly Count")

        method_columns = [
            "is_anomaly_iqr",
            "is_anomaly_zscore",
            "is_anomaly_isolation_forest",
        ]

        method_rows = []

        for column in method_columns:
            if column in anomaly_data.columns:
                method_rows.append(
                    {
                        "method": column.replace("is_anomaly_", "").replace("_", " ").title(),
                        "anomaly_count": int(anomaly_data[column].astype(bool).sum()),
                    }
                )

        if method_rows:
            method_summary = pd.DataFrame(method_rows)

            st.dataframe(method_summary, use_container_width=True, hide_index=True)

            fig = px.bar(
                method_summary,
                x="method",
                y="anomaly_count",
                title="Anomaly Count by Detection Method",
                text="anomaly_count",
            )

            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Asset-wise Anomaly Summary")

        if asset_anomaly_summary is not None and not asset_anomaly_summary.empty:
            asset_summary_data = standardize_columns(asset_anomaly_summary)
        elif {"asset", "is_anomaly"}.issubset(anomaly_data.columns):
            asset_summary_data = (
                anomaly_data.groupby("asset")
                .agg(
                    total_records=("is_anomaly", "count"),
                    anomaly_count=("is_anomaly", "sum"),
                    average_price=("price", "mean") if "price" in anomaly_data.columns else ("is_anomaly", "count"),
                )
                .reset_index()
            )

            asset_summary_data["anomaly_rate_percent"] = (
                asset_summary_data["anomaly_count"]
                / asset_summary_data["total_records"]
                * 100
            )
        else:
            asset_summary_data = pd.DataFrame()

        if not asset_summary_data.empty:
            st.dataframe(asset_summary_data, use_container_width=True, hide_index=True)

            if "anomaly_count" in asset_summary_data.columns:
                fig = px.bar(
                    asset_summary_data,
                    x="asset",
                    y="anomaly_count",
                    title="Anomaly Count by Asset",
                    text="anomaly_count",
                )

                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Top Anomaly Records")

        if top_anomaly_records is not None and not top_anomaly_records.empty:
            st.caption(f"Source: {top_anomaly_records_source}")
            st.dataframe(top_anomaly_records.head(50), use_container_width=True)
        elif "is_anomaly" in anomaly_data.columns:
            top_data = anomaly_data[anomaly_data["is_anomaly"]].copy()

            sort_columns = []

            if "anomaly_method_count" in top_data.columns:
                sort_columns.append("anomaly_method_count")

            if "anomaly_score" in top_data.columns:
                sort_columns.append("anomaly_score")

            if sort_columns:
                ascending_values = [
                    False if column == "anomaly_method_count" else True
                    for column in sort_columns
                ]

                top_data = top_data.sort_values(
                    by=sort_columns,
                    ascending=ascending_values,
                )

            st.dataframe(top_data.head(50), use_container_width=True)


# ============================================================
# Page: Model Health
# ============================================================

elif page == "Model Health":
    st.header("🧪 Model and Project Health")

    st.subheader("Dataset Status")

    dataset_status = get_file_status(DATA_PATH)

    st.json(dataset_status)

    if price_data.empty:
        st.error("Dataset is missing or invalid.")
    else:
        st.success("Dataset is available and valid.")

    st.subheader("Model Artifact Status")

    model_status_rows = []

    for model_name, artifact_path in MODEL_ARTIFACT_PATHS.items():
        status = get_file_status(artifact_path)

        model_status_rows.append(
            {
                "artifact": model_name,
                "exists": status["exists"],
                "path": status["path"],
                "size_kb": status["size_kb"],
                "modified_at": status["modified_at"],
            }
        )

    model_status_df = pd.DataFrame(model_status_rows)

    st.dataframe(model_status_df, use_container_width=True, hide_index=True)

    available_count = int(model_status_df["exists"].sum())

    col1, col2, col3 = st.columns(3)

    col1.metric("Artifacts Checked", len(model_status_df))
    col2.metric("Available", available_count)
    col3.metric("Missing", len(model_status_df) - available_count)

    st.subheader("Generated Output Status")

    output_file_candidates = {
        "Forecast Comparison": FORECAST_COMPARISON_PATHS,
        "Forecast Results": FORECAST_OUTPUT_PATHS,
        "Prediction Results": PREDICTION_OUTPUT_PATHS,
        "Anomaly Results": ANOMALY_OUTPUT_PATHS,
        "Asset Anomaly Summary": ASSET_ANOMALY_SUMMARY_PATHS,
        "Top Anomaly Records": TOP_ANOMALY_RECORDS_PATHS,
        "Project Summary": SUMMARY_JSON_PATHS,
        "Best Model": BEST_MODEL_PATHS,
    }

    output_rows = []

    for output_name, paths in output_file_candidates.items():
        available_path = None

        for path in paths:
            if path.exists():
                available_path = path
                break

        status = get_file_status(available_path) if available_path else {
            "exists": False,
            "path": None,
            "size_kb": None,
            "modified_at": None,
        }

        output_rows.append(
            {
                "output": output_name,
                "exists": status["exists"],
                "path": status["path"],
                "size_kb": status["size_kb"],
                "modified_at": status["modified_at"],
            }
        )

    output_status_df = pd.DataFrame(output_rows)

    st.dataframe(output_status_df, use_container_width=True, hide_index=True)

    st.subheader("Recommended Run Order")

    st.code(
        "python scripts/run_data_pipeline.py\n"
        "python scripts/run_all_models.py\n"
        "python scripts/evaluate_models.py\n"
        "python src/inference/predict.py\n"
        "streamlit run dashboard/app.py",
        language="bash",
    )


# ============================================================
# Page: Downloads
# ============================================================

elif page == "Downloads":
    st.header("⬇️ Download Project Outputs")

    st.subheader("Available Dataframes")

    download_items = {
        "Raw Dataset": price_data,
        "Forecast Comparison": standardize_columns(forecast_comparison) if forecast_comparison is not None else pd.DataFrame(),
        "Forecast Results": standardize_columns(forecast_results) if forecast_results is not None else pd.DataFrame(),
        "Prediction Results": standardize_columns(prediction_results) if prediction_results is not None else pd.DataFrame(),
        "Anomaly Results": standardize_columns(anomaly_results) if anomaly_results is not None else pd.DataFrame(),
        "Asset Anomaly Summary": standardize_columns(asset_anomaly_summary) if asset_anomaly_summary is not None else pd.DataFrame(),
        "Top Anomaly Records": standardize_columns(top_anomaly_records) if top_anomaly_records is not None else pd.DataFrame(),
    }

    for name, dataframe in download_items.items():
        with st.expander(name):
            if dataframe is not None and not dataframe.empty:
                st.dataframe(dataframe.head(100), use_container_width=True)

                safe_name = (
                    name.lower()
                    .replace(" ", "_")
                    .replace("/", "_")
                )

                dataframe_download_button(
                    dataframe=dataframe,
                    file_name=f"{safe_name}.csv",
                    label=f"Download {name} CSV",
                )
            else:
                st.info(f"{name} is not available yet.")

    st.subheader("Available JSON Summaries")

    json_items = {
        "Project Summary": project_summary_json,
        "Best Model": best_model_json,
    }

    for name, data in json_items.items():
        with st.expander(name):
            if data:
                st.json(data)

                safe_name = (
                    name.lower()
                    .replace(" ", "_")
                    .replace("/", "_")
                )

                json_download_button(
                    data=data,
                    file_name=f"{safe_name}.json",
                    label=f"Download {name} JSON",
                )
            else:
                st.info(f"{name} JSON is not available yet.")

    st.subheader("Important File Locations")

    important_locations = pd.DataFrame(
        [
            {"name": "Raw Dataset", "path": str(DATA_PATH)},
            {"name": "Outputs Folder", "path": str(OUTPUTS_DIR)},
            {"name": "Reports Folder", "path": str(REPORTS_DIR)},
            {"name": "Tables Folder", "path": str(TABLES_DIR)},
            {"name": "Figures Folder", "path": str(FIGURES_DIR)},
            {"name": "Models Folder", "path": str(MODELS_DIR)},
        ]
    )

    st.dataframe(important_locations, use_container_width=True, hide_index=True)


# ============================================================
# Footer
# ============================================================

st.markdown("---")
st.caption(
    "Financial Portfolio Forecasting & Anomaly Detection | "
    "Month 5 Practical Project | "
    f"Dashboard loaded at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)