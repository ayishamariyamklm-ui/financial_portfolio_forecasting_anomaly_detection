"""
Price Forecasting page for Streamlit dashboard.

Run dashboard from project root:

    streamlit run dashboard/app.py
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "financial_portfolio_data.csv"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"

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

BEST_MODEL_PATHS = [
    TABLES_DIR / "best_forecasting_model.json",
    TABLES_DIR / "03_best_forecasting_model.json",
]


st.set_page_config(
    page_title="Price Forecasting",
    page_icon="🔮",
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

    if not {"date", "asset", "price"}.issubset(data.columns):
        return pd.DataFrame()

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["price"] = pd.to_numeric(data["price"], errors="coerce")

    data = data.dropna(subset=["date", "asset", "price"])
    data = data[data["price"] >= 0]
    data = data.sort_values(["asset", "date"]).reset_index(drop=True)

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


def dataframe_download_button(dataframe: pd.DataFrame, file_name: str, label: str) -> None:
    st.download_button(
        label=label,
        data=dataframe.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
    )


raw_data = load_csv(str(DATA_PATH))
price_data = clean_price_data(raw_data)

forecast_comparison, forecast_comparison_source = load_first_available_csv(
    FORECAST_COMPARISON_PATHS
)

forecast_results, forecast_results_source = load_first_available_csv(
    FORECAST_OUTPUT_PATHS
)

prediction_results, prediction_results_source = load_first_available_csv(
    PREDICTION_OUTPUT_PATHS
)

best_model_json, best_model_source = load_first_available_json(BEST_MODEL_PATHS)


st.title("🔮 Price Forecasting")
st.caption("Forecasting model comparison, predictions, and future price forecasts")

st.markdown("---")

st.subheader("🏆 Best Forecasting Model")

if best_model_json:
    st.caption(f"Source: {best_model_source}")
    st.json(best_model_json)
elif forecast_comparison is not None and not forecast_comparison.empty:
    comparison_data = standardize_columns(forecast_comparison)

    if "rmse" in comparison_data.columns:
        comparison_data["rmse"] = pd.to_numeric(comparison_data["rmse"], errors="coerce")
        valid_data = comparison_data.dropna(subset=["rmse"])

        if not valid_data.empty:
            best_row = valid_data.sort_values("rmse").iloc[0]

            col1, col2, col3 = st.columns(3)

            col1.metric("Best Model", str(best_row.get("model_name")))
            col2.metric("Best RMSE", round(float(best_row.get("rmse")), 4))

            if "mae" in best_row:
                col3.metric("MAE", round(float(best_row.get("mae")), 4))
else:
    st.info("No best model information available.")
    st.code("python scripts/run_all_models.py", language="bash")

st.subheader("📊 Forecasting Model Comparison")

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

    available_metrics = [column for column in metric_columns if column in comparison_data.columns]

    if available_metrics and "model_name" in comparison_data.columns:
        selected_metric = st.selectbox(
            "Select metric",
            options=available_metrics,
            index=available_metrics.index("rmse") if "rmse" in available_metrics else 0,
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

    dataframe_download_button(
        comparison_data,
        "forecasting_model_comparison.csv",
        "Download Forecast Comparison CSV",
    )

else:
    st.warning("No forecasting comparison table found.")
    st.code(
        "python scripts/run_all_models.py\npython scripts/evaluate_models.py",
        language="bash",
    )

st.subheader("📈 Forecast Results")

if forecast_results is not None and not forecast_results.empty:
    forecast_data = standardize_columns(forecast_results)

    if "date" in forecast_data.columns:
        forecast_data["date"] = pd.to_datetime(forecast_data["date"], errors="coerce")

    forecast_column = None

    for candidate in ["forecasted_price", "predicted_price", "yhat"]:
        if candidate in forecast_data.columns:
            forecast_column = candidate
            break

    st.caption(f"Source: {forecast_results_source}")
    st.dataframe(forecast_data.head(300), use_container_width=True)

    if forecast_column and "date" in forecast_data.columns:
        if "asset" in forecast_data.columns:
            selected_assets = st.multiselect(
                "Select forecast assets",
                options=sorted(forecast_data["asset"].dropna().unique().tolist()),
                default=sorted(forecast_data["asset"].dropna().unique().tolist())[:5],
            )

            filtered_forecast = (
                forecast_data[forecast_data["asset"].isin(selected_assets)]
                if selected_assets
                else forecast_data
            )

            fig = px.line(
                filtered_forecast,
                x="date",
                y=forecast_column,
                color="asset",
                markers=True,
                title="Forecasted Asset Prices",
            )
        else:
            fig = px.line(
                forecast_data,
                x="date",
                y=forecast_column,
                markers=True,
                title="Forecasted Prices",
            )

        st.plotly_chart(fig, use_container_width=True)

    dataframe_download_button(
        forecast_data,
        "forecast_results.csv",
        "Download Forecast Results CSV",
    )

else:
    st.info("No forecast result file found.")
    st.code("python src/inference/predict.py", language="bash")

st.subheader("📌 Actual vs Predicted Prices")

if prediction_results is not None and not prediction_results.empty:
    prediction_data = standardize_columns(prediction_results)

    if "date" in prediction_data.columns:
        prediction_data["date"] = pd.to_datetime(prediction_data["date"], errors="coerce")

    st.caption(f"Source: {prediction_results_source}")
    st.dataframe(prediction_data.head(200), use_container_width=True)

    if "actual_price" in prediction_data.columns:
        prediction_columns = [
            column for column in prediction_data.columns
            if column.endswith("_prediction")
            or column in ["baseline_previous_price", "baseline_rolling_mean"]
        ]

        if prediction_columns:
            selected_prediction_column = st.selectbox(
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
                    y=prediction_data[selected_prediction_column],
                    mode="lines",
                    name=selected_prediction_column,
                    line={"dash": "dash"},
                )
            )

            fig.update_layout(
                title="Actual vs Predicted Price",
                xaxis_title="Date",
                yaxis_title="Price",
            )

            st.plotly_chart(fig, use_container_width=True)

    dataframe_download_button(
        prediction_data,
        "prediction_results.csv",
        "Download Prediction Results CSV",
    )

else:
    st.info("No prediction file found.")
    st.code("Run notebooks/03_forecasting_experiments.ipynb", language="text")

st.subheader("🧾 Forecasting Interpretation Guide")

interpretation = pd.DataFrame(
    [
        {"Metric": "MAE", "Meaning": "Average absolute prediction error", "Better": "Lower"},
        {"Metric": "MSE", "Meaning": "Average squared prediction error", "Better": "Lower"},
        {"Metric": "RMSE", "Meaning": "Penalizes large prediction errors", "Better": "Lower"},
        {"Metric": "MAPE", "Meaning": "Average percentage error", "Better": "Lower"},
        {"Metric": "R2 Score", "Meaning": "Explains price variation", "Better": "Higher"},
    ]
)

st.dataframe(interpretation, use_container_width=True, hide_index=True)