"""
Portfolio Overview page for Streamlit dashboard.

Run dashboard from project root:

    streamlit run dashboard/app.py
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "financial_portfolio_data.csv"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

SUMMARY_JSON_PATHS = [
    TABLES_DIR / "05_final_project_evaluation.json",
    TABLES_DIR / "evaluation_summary.json",
    TABLES_DIR / "api_summary.json",
    TABLES_DIR / "all_models_training_summary.json",
    TABLES_DIR / "inference_summary.json",
]


st.set_page_config(
    page_title="Portfolio Overview",
    page_icon="📊",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_csv(file_path: str) -> Optional[pd.DataFrame]:
    path = Path(file_path)

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_json(file_path: str) -> Dict[str, Any]:
    path = Path(file_path)

    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def standardize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    data = dataframe.copy()

    data.columns = (
        data.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    return data


def clean_price_data(dataframe: Optional[pd.DataFrame]) -> pd.DataFrame:
    if dataframe is None or dataframe.empty:
        return pd.DataFrame()

    data = standardize_columns(dataframe)

    required_columns = ["date", "asset", "price"]

    if not set(required_columns).issubset(data.columns):
        return pd.DataFrame()

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["price"] = pd.to_numeric(data["price"], errors="coerce")

    data = data.dropna(subset=["date", "asset", "price"])
    data = data[data["price"] >= 0]
    data = data.sort_values(["asset", "date"]).reset_index(drop=True)

    return data


def load_first_available_json(paths: list[Path]) -> tuple[Dict[str, Any], Optional[Path]]:
    for path in paths:
        data = load_json(str(path))

        if data:
            return data, path

    return {}, None


def get_file_status(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "path": str(path),
            "size_kb": None,
            "modified_at": None,
        }

    return {
        "exists": True,
        "path": str(path),
        "size_kb": round(path.stat().st_size / 1024, 2),
        "modified_at": datetime.fromtimestamp(path.stat().st_mtime).strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    }


raw_data = load_csv(str(DATA_PATH))
price_data = clean_price_data(raw_data)
summary_json, summary_source = load_first_available_json(SUMMARY_JSON_PATHS)


st.title("📊 Portfolio Overview")
st.caption("Financial Portfolio Forecasting & Anomaly Detection | Month 5 Practical Project")

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

if not price_data.empty:
    total_records = len(price_data)
    total_assets = price_data["asset"].nunique()
    start_date = str(price_data["date"].min().date())
    end_date = str(price_data["date"].max().date())
else:
    total_records = 0
    total_assets = 0
    start_date = "N/A"
    end_date = "N/A"

col1.metric("Total Records", total_records)
col2.metric("Total Assets", total_assets)
col3.metric("Start Date", start_date)
col4.metric("End Date", end_date)

st.subheader("🎯 Business Problem")

st.write(
    """
    This project builds a financial portfolio monitoring system that predicts future
    asset prices and detects unusual asset price movements. It can support risk
    monitoring, data quality checks, asset trend analysis, and investment decision support.
    """
)

st.subheader("🧱 Project Architecture")

architecture_steps = pd.DataFrame(
    [
        {
            "Step": 1,
            "Stage": "Data Collection",
            "Description": "Load raw financial asset price data from CSV.",
            "Output": "data/raw/financial_portfolio_data.csv",
        },
        {
            "Step": 2,
            "Stage": "Data Processing",
            "Description": "Clean dates, prices, missing values, and asset labels.",
            "Output": "data/processed/",
        },
        {
            "Step": 3,
            "Stage": "Time Series EDA",
            "Description": "Analyze trends, returns, volatility, and asset-level behavior.",
            "Output": "reports/figures/",
        },
        {
            "Step": 4,
            "Stage": "Forecasting",
            "Description": "Train ARIMA, XGBoost, Prophet, and LSTM forecasting models.",
            "Output": "models/ and outputs/",
        },
        {
            "Step": 5,
            "Stage": "Anomaly Detection",
            "Description": "Detect unusual prices using IQR, Z-score, and Isolation Forest.",
            "Output": "outputs/anomaly_detection_results.csv",
        },
        {
            "Step": 6,
            "Stage": "Evaluation",
            "Description": "Compare model metrics and generate portfolio-ready reports.",
            "Output": "reports/tables/ and reports/*.md",
        },
        {
            "Step": 7,
            "Stage": "Deployment",
            "Description": "Serve predictions through FastAPI and Streamlit dashboard.",
            "Output": "api/ and dashboard/",
        },
    ]
)

st.dataframe(architecture_steps, use_container_width=True, hide_index=True)

if not price_data.empty:
    st.subheader("📈 Portfolio Price Trend")

    selected_assets = st.multiselect(
        "Select assets",
        options=sorted(price_data["asset"].unique().tolist()),
        default=sorted(price_data["asset"].unique().tolist())[:5],
    )

    filtered_data = (
        price_data[price_data["asset"].isin(selected_assets)]
        if selected_assets
        else price_data
    )

    fig = px.line(
        filtered_data,
        x="date",
        y="price",
        color="asset",
        markers=True,
        title="Asset Price Trend Over Time",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📌 Asset Summary")

    asset_summary = (
        price_data.groupby("asset")
        .agg(
            record_count=("price", "count"),
            average_price=("price", "mean"),
            min_price=("price", "min"),
            max_price=("price", "max"),
            price_std=("price", "std"),
            start_date=("date", "min"),
            end_date=("date", "max"),
        )
        .reset_index()
    )

    st.dataframe(asset_summary, use_container_width=True, hide_index=True)

    st.subheader("📉 Return & Volatility Summary")

    return_data = price_data.copy()
    return_data["daily_return"] = (
        return_data.groupby("asset")["price"]
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    volatility_summary = (
        return_data.groupby("asset")
        .agg(
            average_daily_return=("daily_return", "mean"),
            volatility=("daily_return", "std"),
            max_daily_return=("daily_return", "max"),
            min_daily_return=("daily_return", "min"),
        )
        .reset_index()
        .sort_values("volatility", ascending=False)
    )

    st.dataframe(volatility_summary, use_container_width=True, hide_index=True)

    fig = px.bar(
        volatility_summary,
        x="asset",
        y="volatility",
        title="Asset Volatility Ranking",
        text="volatility",
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Dataset not found or invalid.")
    st.code("Place your CSV at data/raw/financial_portfolio_data.csv", language="text")

st.subheader("📄 Latest Project Summary")

if summary_json:
    st.caption(f"Source: {summary_source}")
    st.json(summary_json)
else:
    st.info("No final summary JSON found yet.")
    st.code(
        "python scripts/run_all_models.py\npython scripts/evaluate_models.py",
        language="bash",
    )

st.subheader("🗂️ Core File Status")

status_rows = [
    {"File": "Raw Dataset", **get_file_status(DATA_PATH)},
    {"File": "Models Folder", **get_file_status(MODELS_DIR)},
    {"File": "Outputs Folder", **get_file_status(OUTPUTS_DIR)},
    {"File": "Reports Folder", **get_file_status(REPORTS_DIR)},
]

st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)