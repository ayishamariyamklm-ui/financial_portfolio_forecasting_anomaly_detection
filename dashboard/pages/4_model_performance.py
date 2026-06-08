"""
Model Performance page for Streamlit dashboard.

Run dashboard from project root:

    streamlit run dashboard/app.py
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "financial_portfolio_data.csv"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"

FORECAST_COMPARISON_PATHS = [
    TABLES_DIR / "model_comparison.csv",
    TABLES_DIR / "forecasting_model_comparison.csv",
    TABLES_DIR / "03_forecasting_model_comparison.csv",
    TABLES_DIR / "05_forecast_comparison_final.csv",
]

PREDICTION_EVALUATION_PATHS = [
    TABLES_DIR / "05_prediction_evaluation.csv",
    TABLES_DIR / "forecast_evaluation_combined.csv",
]

EVALUATION_SUMMARY_PATHS = [
    TABLES_DIR / "05_final_project_evaluation.json",
    TABLES_DIR / "evaluation_summary.json",
    TABLES_DIR / "all_models_training_summary.json",
]

MODEL_ARTIFACT_PATHS = {
    "XGBoost Model": MODELS_DIR / "xgboost_model.pkl",
    "ARIMA Models": MODELS_DIR / "arima_models.pkl",
    "Prophet Models": MODELS_DIR / "prophet_models.pkl",
    "LSTM Models Directory": MODELS_DIR / "lstm_models",
    "LSTM Scalers": MODELS_DIR / "lstm_scalers.pkl",
    "Anomaly Detector": MODELS_DIR / "anomaly_detection_model.pkl",
}


st.set_page_config(
    page_title="Model Performance",
    page_icon="📉",
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


def load_first_available_csv(paths: list[Path]) -> tuple[Optional[pd.DataFrame], Optional[Path]]:
    for path in paths:
        dataframe = load_csv(str(path))

        if dataframe is not None and not dataframe.empty:
            return dataframe, path

    return None, None


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

    if path.is_dir():
        file_count = len([item for item in path.iterdir() if item.is_file()])
        return {
            "exists": True,
            "path": str(path),
            "size_kb": None,
            "modified_at": datetime.fromtimestamp(path.stat().st_mtime).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "file_count": file_count,
        }

    return {
        "exists": True,
        "path": str(path),
        "size_kb": round(path.stat().st_size / 1024, 2),
        "modified_at": datetime.fromtimestamp(path.stat().st_mtime).strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "file_count": None,
    }


def dataframe_download_button(dataframe: pd.DataFrame, file_name: str, label: str) -> None:
    st.download_button(
        label=label,
        data=dataframe.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
    )


forecast_comparison, forecast_comparison_source = load_first_available_csv(
    FORECAST_COMPARISON_PATHS
)

prediction_evaluation, prediction_evaluation_source = load_first_available_csv(
    PREDICTION_EVALUATION_PATHS
)

evaluation_summary, evaluation_summary_source = load_first_available_json(
    EVALUATION_SUMMARY_PATHS
)


st.title("📉 Model Performance")
st.caption("Forecasting metrics, model ranking, artifact status, and final evaluation")

st.markdown("---")

st.subheader("📊 Forecasting Model Metrics")

if forecast_comparison is not None and not forecast_comparison.empty:
    comparison_data = standardize_columns(forecast_comparison)

    metric_columns = ["mae", "mse", "rmse", "mape", "smape", "r2_score"]

    for column in metric_columns:
        if column in comparison_data.columns:
            comparison_data[column] = pd.to_numeric(comparison_data[column], errors="coerce")

    if "rmse" in comparison_data.columns:
        comparison_data = comparison_data.sort_values("rmse", ascending=True)

    st.caption(f"Source: {forecast_comparison_source}")
    st.dataframe(comparison_data, use_container_width=True, hide_index=True)

    col1, col2, col3 = st.columns(3)

    if "rmse" in comparison_data.columns and not comparison_data.dropna(subset=["rmse"]).empty:
        best_row = comparison_data.dropna(subset=["rmse"]).sort_values("rmse").iloc[0]

        col1.metric("Best Model", str(best_row.get("model_name")))
        col2.metric("Best RMSE", round(float(best_row.get("rmse")), 4))

        if "mae" in comparison_data.columns and pd.notna(best_row.get("mae")):
            col3.metric("Best MAE", round(float(best_row.get("mae")), 4))

    available_metrics = [
        column for column in metric_columns
        if column in comparison_data.columns
    ]

    if available_metrics and "model_name" in comparison_data.columns:
        selected_metric = st.selectbox(
            "Select metric for chart",
            options=available_metrics,
            index=available_metrics.index("rmse") if "rmse" in available_metrics else 0,
        )

        plot_data = comparison_data.dropna(subset=[selected_metric])

        fig = px.bar(
            plot_data,
            x="model_name",
            y=selected_metric,
            text=selected_metric,
            title=f"Model Performance by {selected_metric.upper()}",
        )

        st.plotly_chart(fig, use_container_width=True)

    dataframe_download_button(
        comparison_data,
        "model_performance_comparison.csv",
        "Download Model Comparison CSV",
    )

else:
    st.warning("No forecasting model comparison file found.")
    st.code(
        "python scripts/run_all_models.py\npython scripts/evaluate_models.py",
        language="bash",
    )

st.subheader("🧪 Prediction Evaluation")

if prediction_evaluation is not None and not prediction_evaluation.empty:
    prediction_eval_data = standardize_columns(prediction_evaluation)

    for column in ["mae", "mse", "rmse", "mape", "smape", "r2_score"]:
        if column in prediction_eval_data.columns:
            prediction_eval_data[column] = pd.to_numeric(
                prediction_eval_data[column],
                errors="coerce",
            )

    if "rmse" in prediction_eval_data.columns:
        prediction_eval_data = prediction_eval_data.sort_values("rmse")

    st.caption(f"Source: {prediction_evaluation_source}")
    st.dataframe(prediction_eval_data, use_container_width=True, hide_index=True)

    if "rmse" in prediction_eval_data.columns and "model_name" in prediction_eval_data.columns:
        fig = px.bar(
            prediction_eval_data.dropna(subset=["rmse"]),
            x="model_name",
            y="rmse",
            text="rmse",
            title="Prediction Evaluation by RMSE",
        )

        st.plotly_chart(fig, use_container_width=True)

    dataframe_download_button(
        prediction_eval_data,
        "prediction_evaluation.csv",
        "Download Prediction Evaluation CSV",
    )
else:
    st.info("No prediction evaluation file found.")
    st.code("python scripts/evaluate_models.py", language="bash")

st.subheader("📦 Model Artifact Health")

artifact_rows = []

for artifact_name, artifact_path in MODEL_ARTIFACT_PATHS.items():
    status = get_file_status(artifact_path)

    artifact_rows.append(
        {
            "artifact": artifact_name,
            "exists": status.get("exists"),
            "path": status.get("path"),
            "size_kb": status.get("size_kb"),
            "file_count": status.get("file_count"),
            "modified_at": status.get("modified_at"),
        }
    )

artifact_status_df = pd.DataFrame(artifact_rows)

st.dataframe(artifact_status_df, use_container_width=True, hide_index=True)

available_count = int(artifact_status_df["exists"].sum())

col1, col2, col3 = st.columns(3)

col1.metric("Artifacts Checked", len(artifact_status_df))
col2.metric("Available", available_count)
col3.metric("Missing", len(artifact_status_df) - available_count)

fig = px.pie(
    artifact_status_df,
    names="exists",
    title="Model Artifact Availability",
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("📄 Final Evaluation Summary")

if evaluation_summary:
    st.caption(f"Source: {evaluation_summary_source}")
    st.json(evaluation_summary)
else:
    st.info("No evaluation summary JSON found.")
    st.code("python scripts/evaluate_models.py", language="bash")

st.subheader("🧾 Performance Interpretation Guide")

guide = pd.DataFrame(
    [
        {
            "Metric": "MAE",
            "Purpose": "Measures average absolute error.",
            "Best Direction": "Lower is better",
            "Business Meaning": "Average money/value difference between actual and predicted price.",
        },
        {
            "Metric": "RMSE",
            "Purpose": "Measures error while penalizing large mistakes.",
            "Best Direction": "Lower is better",
            "Business Meaning": "Useful when large forecasting mistakes are risky.",
        },
        {
            "Metric": "MAPE",
            "Purpose": "Measures average percentage error.",
            "Best Direction": "Lower is better",
            "Business Meaning": "Easy to explain to non-technical stakeholders.",
        },
        {
            "Metric": "R2 Score",
            "Purpose": "Measures how much variation is explained.",
            "Best Direction": "Higher is better",
            "Business Meaning": "Shows how well the model understands price behavior.",
        },
        {
            "Metric": "Anomaly Rate",
            "Purpose": "Measures percentage of unusual records.",
            "Best Direction": "Context dependent",
            "Business Meaning": "Higher rates may indicate risk, volatility, or data quality problems.",
        },
    ]
)

st.dataframe(guide, use_container_width=True, hide_index=True)

st.subheader("✅ Recommended Run Order")

st.code(
    "python scripts/run_data_pipeline.py\n"
    "python scripts/run_all_models.py\n"
    "python scripts/evaluate_models.py\n"
    "python src/inference/predict.py\n"
    "streamlit run dashboard/app.py",
    language="bash",
)