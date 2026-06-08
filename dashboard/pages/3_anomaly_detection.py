"""
Anomaly Detection page for Streamlit dashboard.

Run dashboard from project root:

    streamlit run dashboard/app.py
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"

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


st.set_page_config(
    page_title="Anomaly Detection",
    page_icon="🚨",
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


def load_first_available_csv(paths: list[Path]) -> tuple[Optional[pd.DataFrame], Optional[Path]]:
    for path in paths:
        dataframe = load_csv(str(path))

        if dataframe is not None and not dataframe.empty:
            return dataframe, path

    return None, None


def dataframe_download_button(dataframe: pd.DataFrame, file_name: str, label: str) -> None:
    st.download_button(
        label=label,
        data=dataframe.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
    )


anomaly_results, anomaly_source = load_first_available_csv(ANOMALY_OUTPUT_PATHS)
asset_anomaly_summary, asset_summary_source = load_first_available_csv(
    ASSET_ANOMALY_SUMMARY_PATHS
)
top_anomaly_records, top_anomaly_source = load_first_available_csv(
    TOP_ANOMALY_RECORDS_PATHS
)


st.title("🚨 Anomaly Detection")
st.caption("IQR, Z-score, Isolation Forest, and combined anomaly results")

st.markdown("---")

if anomaly_results is None or anomaly_results.empty:
    st.warning("No anomaly result file found.")
    st.code(
        "python src/training/train_anomaly_model.py\npython scripts/evaluate_models.py",
        language="bash",
    )
    st.stop()

anomaly_data = standardize_columns(anomaly_results)

if "date" in anomaly_data.columns:
    anomaly_data["date"] = pd.to_datetime(anomaly_data["date"], errors="coerce")

if "price" in anomaly_data.columns:
    anomaly_data["price"] = pd.to_numeric(anomaly_data["price"], errors="coerce")

if "is_anomaly" in anomaly_data.columns:
    anomaly_data["is_anomaly"] = anomaly_data["is_anomaly"].astype(bool)

st.caption(f"Source: {anomaly_source}")

total_records = len(anomaly_data)
total_anomalies = (
    int(anomaly_data["is_anomaly"].sum())
    if "is_anomaly" in anomaly_data.columns
    else 0
)
normal_records = total_records - total_anomalies
anomaly_rate = (total_anomalies / total_records * 100) if total_records > 0 else 0
asset_count = anomaly_data["asset"].nunique() if "asset" in anomaly_data.columns else 0

col1, col2, col3, col4 = st.columns(4)

col1.metric("Records Checked", total_records)
col2.metric("Normal Records", normal_records)
col3.metric("Anomalies", total_anomalies)
col4.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")

st.subheader("📋 Anomaly Results Table")

only_anomalies = st.checkbox("Show only anomaly records", value=False)

table_data = anomaly_data.copy()

if only_anomalies and "is_anomaly" in table_data.columns:
    table_data = table_data[table_data["is_anomaly"]]

st.dataframe(table_data.head(1000), use_container_width=True)

dataframe_download_button(
    table_data,
    "filtered_anomaly_results.csv",
    "Download Current Table CSV",
)

st.subheader("📈 Asset Price Chart with Anomalies")

required_chart_columns = {"date", "asset", "price", "is_anomaly"}

if required_chart_columns.issubset(anomaly_data.columns):
    selected_assets = st.multiselect(
        "Select assets",
        options=sorted(anomaly_data["asset"].dropna().unique().tolist()),
        default=sorted(anomaly_data["asset"].dropna().unique().tolist())[:5],
    )

    chart_data = (
        anomaly_data[anomaly_data["asset"].isin(selected_assets)]
        if selected_assets
        else anomaly_data
    )

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
else:
    st.info("Required chart columns are missing.")

st.subheader("🧪 Method-wise Anomaly Detection")

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
                "Method": column.replace("is_anomaly_", "").replace("_", " ").title(),
                "Anomaly Count": int(anomaly_data[column].astype(bool).sum()),
            }
        )

if method_rows:
    method_summary = pd.DataFrame(method_rows)

    st.dataframe(method_summary, use_container_width=True, hide_index=True)

    fig = px.bar(
        method_summary,
        x="Method",
        y="Anomaly Count",
        text="Anomaly Count",
        title="Anomaly Count by Detection Method",
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Method-wise anomaly columns were not found.")

st.subheader("🔥 Anomaly Severity Distribution")

if "anomaly_severity" in anomaly_data.columns:
    severity_summary = (
        anomaly_data["anomaly_severity"]
        .value_counts()
        .reset_index()
    )

    severity_summary.columns = ["Severity", "Record Count"]

    st.dataframe(severity_summary, use_container_width=True, hide_index=True)

    fig = px.bar(
        severity_summary,
        x="Severity",
        y="Record Count",
        text="Record Count",
        title="Anomaly Severity Distribution",
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Anomaly severity column is not available.")

st.subheader("📌 Asset-wise Anomaly Summary")

if asset_anomaly_summary is not None and not asset_anomaly_summary.empty:
    asset_summary_data = standardize_columns(asset_anomaly_summary)
    st.caption(f"Source: {asset_summary_source}")
else:
    if {"asset", "is_anomaly"}.issubset(anomaly_data.columns):
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
            text="anomaly_count",
            title="Anomaly Count by Asset",
        )

        st.plotly_chart(fig, use_container_width=True)

    dataframe_download_button(
        asset_summary_data,
        "asset_anomaly_summary.csv",
        "Download Asset Anomaly Summary CSV",
    )
else:
    st.info("Asset-wise anomaly summary is not available.")

st.subheader("🔎 Top Anomaly Records")

if top_anomaly_records is not None and not top_anomaly_records.empty:
    top_data = standardize_columns(top_anomaly_records)
    st.caption(f"Source: {top_anomaly_source}")
else:
    if "is_anomaly" in anomaly_data.columns:
        top_data = anomaly_data[anomaly_data["is_anomaly"]].copy()

        sort_columns = []
        ascending_values = []

        if "anomaly_method_count" in top_data.columns:
            sort_columns.append("anomaly_method_count")
            ascending_values.append(False)

        if "anomaly_score" in top_data.columns:
            sort_columns.append("anomaly_score")
            ascending_values.append(True)

        if sort_columns:
            top_data = top_data.sort_values(
                by=sort_columns,
                ascending=ascending_values,
            )

        top_data = top_data.head(50)
    else:
        top_data = pd.DataFrame()

if not top_data.empty:
    st.dataframe(top_data, use_container_width=True)

    dataframe_download_button(
        top_data,
        "top_anomaly_records.csv",
        "Download Top Anomaly Records CSV",
    )
else:
    st.info("Top anomaly records are not available.")

st.subheader("🧾 Anomaly Detection Interpretation")

interpretation = pd.DataFrame(
    [
        {
            "Method": "IQR",
            "Meaning": "Detects price values outside the interquartile range.",
            "Strength": "Simple and easy to explain.",
        },
        {
            "Method": "Z-score",
            "Meaning": "Detects prices far from the asset average.",
            "Strength": "Good for normally distributed data.",
        },
        {
            "Method": "Isolation Forest",
            "Meaning": "Uses multiple features to detect unusual patterns.",
            "Strength": "Good for complex anomaly detection.",
        },
        {
            "Method": "Combined Flag",
            "Meaning": "Marks records detected by at least one method.",
            "Strength": "More robust for risk monitoring.",
        },
    ]
)

st.dataframe(interpretation, use_container_width=True, hide_index=True)