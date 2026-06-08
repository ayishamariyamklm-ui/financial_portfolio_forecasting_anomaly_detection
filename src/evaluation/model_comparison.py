"""
Model comparison utilities for the Financial Portfolio Forecasting
and Anomaly Detection project.

This module compares forecasting models using saved metric files.

Supported forecasting models:
- ARIMA
- XGBoost
- Prophet
- LSTM

Main comparison metric:
- RMSE, lower is better

Expected metric files:
- reports/tables/arima_metrics.json
- reports/tables/xgboost_metrics.json
- reports/tables/prophet_metrics.json
- reports/tables/lstm_metrics.json
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

ARIMA_METRICS_PATH = TABLES_DIR / "arima_metrics.json"
XGBOOST_METRICS_PATH = TABLES_DIR / "xgboost_metrics.json"
PROPHET_METRICS_PATH = TABLES_DIR / "prophet_metrics.json"
LSTM_METRICS_PATH = TABLES_DIR / "lstm_metrics.json"

MODEL_COMPARISON_CSV_PATH = TABLES_DIR / "model_comparison.csv"
MODEL_COMPARISON_JSON_PATH = TABLES_DIR / "model_comparison.json"
BEST_MODEL_JSON_PATH = TABLES_DIR / "best_forecasting_model.json"
MODEL_COMPARISON_REPORT_PATH = REPORTS_DIR / "model_comparison_report.md"


# ============================================================
# Directory Helper
# ============================================================

def ensure_comparison_directories() -> None:
    """
    Create required directories for model comparison outputs.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# JSON Loading Helpers
# ============================================================

def load_json_file(file_path: Path) -> Dict[str, Any]:
    """
    Load a JSON file safely.

    Args:
        file_path (Path): JSON file path.

    Returns:
        Dict[str, Any]: Loaded JSON content.
    """
    if not file_path.exists():
        return {}

    try:
        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        return data if isinstance(data, dict) else {}

    except Exception:
        return {}


def save_json_file(data: Dict[str, Any], file_path: Path) -> None:
    """
    Save dictionary as JSON.

    Args:
        data (Dict[str, Any]): Dictionary to save.
        file_path (Path): Output path.
    """
    ensure_comparison_directories()
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, default=str)


# ============================================================
# Metric Normalization Helpers
# ============================================================

def safe_float(value: Any) -> Optional[float]:
    """
    Safely convert value to float.

    Args:
        value (Any): Input value.

    Returns:
        Optional[float]: Float value or None.
    """
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def average_nested_metrics(metrics: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Average asset-wise metric dictionaries into one model-level metric dictionary.

    This is useful for ARIMA, Prophet, and LSTM where metrics are saved
    asset-wise, for example:

    {
        "Bond A": {"mae": 10, "rmse": 20},
        "Stock A": {"mae": 15, "rmse": 25}
    }

    Args:
        metrics (Dict[str, Any]): Asset-wise metrics dictionary.

    Returns:
        Dict[str, Optional[float]]: Averaged metrics.
    """
    metric_names = ["mae", "mse", "rmse", "mape", "smape", "r2_score"]
    averaged_metrics: Dict[str, Optional[float]] = {}

    if not metrics:
        return {metric: None for metric in metric_names}

    for metric_name in metric_names:
        values = []

        for value in metrics.values():
            if isinstance(value, dict):
                metric_value = safe_float(value.get(metric_name))

                if metric_value is not None:
                    values.append(metric_value)

        averaged_metrics[metric_name] = (
            round(sum(values) / len(values), 6)
            if values
            else None
        )

    return averaged_metrics


def normalize_metrics(metrics: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Normalize metrics dictionary into a standard format.

    Handles both:
    1. Direct metrics:
       {"mae": 10, "rmse": 20}

    2. Nested asset-wise metrics:
       {"Bond A": {"mae": 10}, "Stock A": {"mae": 15}}

    Args:
        metrics (Dict[str, Any]): Raw metrics dictionary.

    Returns:
        Dict[str, Optional[float]]: Standardized metric dictionary.
    """
    direct_metric_names = ["mae", "mse", "rmse", "mape", "smape", "r2_score"]

    has_direct_metrics = any(metric_name in metrics for metric_name in direct_metric_names)

    if has_direct_metrics:
        return {
            metric_name: safe_float(metrics.get(metric_name))
            for metric_name in direct_metric_names
        }

    return average_nested_metrics(metrics)


def count_trained_assets(metrics: Dict[str, Any]) -> int:
    """
    Count trained assets from metric dictionary.

    Args:
        metrics (Dict[str, Any]): Metrics dictionary.

    Returns:
        int: Trained asset count.
    """
    direct_metric_names = {"mae", "mse", "rmse", "mape", "smape", "r2_score"}

    if any(metric_name in metrics for metric_name in direct_metric_names):
        return 1

    return len(
        [
            value
            for value in metrics.values()
            if isinstance(value, dict)
        ]
    )


def get_asset_names(metrics: Dict[str, Any]) -> List[str]:
    """
    Extract asset names from asset-wise metrics.

    Args:
        metrics (Dict[str, Any]): Metrics dictionary.

    Returns:
        List[str]: Asset names.
    """
    direct_metric_names = {"mae", "mse", "rmse", "mape", "smape", "r2_score"}

    if any(metric_name in metrics for metric_name in direct_metric_names):
        return []

    return [
        str(key)
        for key, value in metrics.items()
        if isinstance(value, dict)
    ]


# ============================================================
# Model Comparison Core
# ============================================================

def build_model_row(
    model_name: str,
    metrics: Dict[str, Any],
    metrics_path: Path,
) -> Dict[str, Any]:
    """
    Build a comparison row for one model.

    Args:
        model_name (str): Model name.
        metrics (Dict[str, Any]): Metrics dictionary.
        metrics_path (Path): Source metrics path.

    Returns:
        Dict[str, Any]: Comparison row.
    """
    normalized_metrics = normalize_metrics(metrics)

    model_status = "available" if metrics else "missing"

    asset_names = get_asset_names(metrics)

    return {
        "model_name": model_name,
        "status": model_status,
        "mae": normalized_metrics.get("mae"),
        "mse": normalized_metrics.get("mse"),
        "rmse": normalized_metrics.get("rmse"),
        "mape": normalized_metrics.get("mape"),
        "smape": normalized_metrics.get("smape"),
        "r2_score": normalized_metrics.get("r2_score"),
        "trained_asset_count": count_trained_assets(metrics) if metrics else 0,
        "trained_assets": ", ".join(asset_names),
        "metrics_file": str(metrics_path),
    }


def load_all_model_metrics() -> Dict[str, Dict[str, Any]]:
    """
    Load all model metrics from reports/tables.

    Returns:
        Dict[str, Dict[str, Any]]: Model name to metrics dictionary.
    """
    return {
        "ARIMA": load_json_file(ARIMA_METRICS_PATH),
        "XGBoost": load_json_file(XGBOOST_METRICS_PATH),
        "Prophet": load_json_file(PROPHET_METRICS_PATH),
        "LSTM": load_json_file(LSTM_METRICS_PATH),
    }


def create_model_comparison_dataframe(
    model_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
    selection_metric: str = "rmse",
    lower_is_better: bool = True,
) -> pd.DataFrame:
    """
    Create model comparison dataframe.

    Args:
        model_metrics (Optional[Dict[str, Dict[str, Any]]]): Optional metric dictionary.
        selection_metric (str): Metric used for ranking.
        lower_is_better (bool): Whether lower metric value is better.

    Returns:
        pd.DataFrame: Model comparison dataframe.
    """
    ensure_comparison_directories()

    model_metrics = model_metrics or load_all_model_metrics()

    metric_paths = {
        "ARIMA": ARIMA_METRICS_PATH,
        "XGBoost": XGBOOST_METRICS_PATH,
        "Prophet": PROPHET_METRICS_PATH,
        "LSTM": LSTM_METRICS_PATH,
    }

    rows = []

    for model_name, metrics in model_metrics.items():
        row = build_model_row(
            model_name=model_name,
            metrics=metrics,
            metrics_path=metric_paths.get(model_name, TABLES_DIR),
        )
        rows.append(row)

    comparison_dataframe = pd.DataFrame(rows)

    if selection_metric in comparison_dataframe.columns:
        comparison_dataframe["_rank_metric"] = pd.to_numeric(
            comparison_dataframe[selection_metric],
            errors="coerce",
        )

        available_mask = (
            (comparison_dataframe["status"] == "available")
            & comparison_dataframe["_rank_metric"].notna()
        )

        available_models = comparison_dataframe[available_mask].copy()
        unavailable_models = comparison_dataframe[~available_mask].copy()

        if not available_models.empty:
            available_models = available_models.sort_values(
                "_rank_metric",
                ascending=lower_is_better,
            ).reset_index(drop=True)

            available_models["rank"] = range(1, len(available_models) + 1)

        if not unavailable_models.empty:
            unavailable_models["rank"] = None

        comparison_dataframe = pd.concat(
            [available_models, unavailable_models],
            ignore_index=True,
        )

        comparison_dataframe = comparison_dataframe.drop(columns=["_rank_metric"])

    return comparison_dataframe


def select_best_model(
    comparison_dataframe: pd.DataFrame,
    selection_metric: str = "rmse",
    lower_is_better: bool = True,
) -> Dict[str, Any]:
    """
    Select the best model from comparison dataframe.

    Args:
        comparison_dataframe (pd.DataFrame): Model comparison dataframe.
        selection_metric (str): Selection metric.
        lower_is_better (bool): Whether lower value is better.

    Returns:
        Dict[str, Any]: Best model information.
    """
    if comparison_dataframe.empty:
        return {
            "best_model": None,
            "selection_metric": selection_metric,
            "metric_value": None,
            "reason": "Comparison dataframe is empty.",
        }

    if selection_metric not in comparison_dataframe.columns:
        return {
            "best_model": None,
            "selection_metric": selection_metric,
            "metric_value": None,
            "reason": f"Selection metric '{selection_metric}' not found.",
        }

    available_models = comparison_dataframe[
        comparison_dataframe["status"] == "available"
    ].copy()

    available_models[selection_metric] = pd.to_numeric(
        available_models[selection_metric],
        errors="coerce",
    )

    available_models = available_models.dropna(subset=[selection_metric])

    if available_models.empty:
        return {
            "best_model": None,
            "selection_metric": selection_metric,
            "metric_value": None,
            "reason": "No available models with valid metric values.",
        }

    available_models = available_models.sort_values(
        selection_metric,
        ascending=lower_is_better,
    )

    best_row = available_models.iloc[0]

    return {
        "best_model": best_row.get("model_name"),
        "selection_metric": selection_metric,
        "metric_value": float(best_row.get(selection_metric)),
        "mae": safe_float(best_row.get("mae")),
        "mse": safe_float(best_row.get("mse")),
        "rmse": safe_float(best_row.get("rmse")),
        "mape": safe_float(best_row.get("mape")),
        "smape": safe_float(best_row.get("smape")),
        "r2_score": safe_float(best_row.get("r2_score")),
        "trained_asset_count": int(best_row.get("trained_asset_count", 0)),
        "trained_assets": best_row.get("trained_assets"),
        "reason": f"Selected because it has the best {selection_metric}.",
    }


# ============================================================
# Save Outputs
# ============================================================

def save_model_comparison_outputs(
    comparison_dataframe: pd.DataFrame,
    best_model: Dict[str, Any],
    csv_path: Path = MODEL_COMPARISON_CSV_PATH,
    json_path: Path = MODEL_COMPARISON_JSON_PATH,
    best_model_path: Path = BEST_MODEL_JSON_PATH,
) -> None:
    """
    Save model comparison dataframe and best model information.

    Args:
        comparison_dataframe (pd.DataFrame): Comparison dataframe.
        best_model (Dict[str, Any]): Best model dictionary.
        csv_path (Path): CSV output path.
        json_path (Path): JSON output path.
        best_model_path (Path): Best model JSON output path.
    """
    ensure_comparison_directories()

    comparison_dataframe.to_csv(csv_path, index=False)

    comparison_records = comparison_dataframe.to_dict(orient="records")

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(comparison_records, file, indent=4, default=str)

    save_json_file(best_model, best_model_path)


def save_model_comparison_report(
    comparison_dataframe: pd.DataFrame,
    best_model: Dict[str, Any],
    report_path: Path = MODEL_COMPARISON_REPORT_PATH,
) -> None:
    """
    Save model comparison report as Markdown.

    Args:
        comparison_dataframe (pd.DataFrame): Comparison dataframe.
        best_model (Dict[str, Any]): Best model dictionary.
        report_path (Path): Markdown report path.
    """
    ensure_comparison_directories()

    lines = [
        "# Model Comparison Report",
        "",
        "## Project",
        "",
        "Financial Portfolio Forecasting & Anomaly Detection",
        "",
        "## Report Information",
        "",
        f"- Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- Selection Metric: RMSE",
        "- Selection Rule: Lower RMSE is better",
        "",
        "## Compared Models",
        "",
        "- ARIMA",
        "- XGBoost",
        "- Prophet",
        "- LSTM",
        "",
        "## Model Comparison Table",
        "",
        "| Rank | Model | Status | MAE | RMSE | MAPE | R2 Score | Assets |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]

    if not comparison_dataframe.empty:
        for _, row in comparison_dataframe.iterrows():
            rank_value = row.get("rank")
            rank_display = "" if pd.isna(rank_value) else int(rank_value)

            lines.append(
                "| {rank} | {model} | {status} | {mae} | {rmse} | {mape} | {r2} | {assets} |".format(
                    rank=rank_display,
                    model=row.get("model_name"),
                    status=row.get("status"),
                    mae=row.get("mae"),
                    rmse=row.get("rmse"),
                    mape=row.get("mape"),
                    r2=row.get("r2_score"),
                    assets=row.get("trained_asset_count"),
                )
            )
    else:
        lines.append("| - | No model comparison data available | - | - | - | - | - | - |")

    lines.extend(
        [
            "",
            "## Best Forecasting Model",
            "",
            f"- Best Model: {best_model.get('best_model')}",
            f"- Selection Metric: {best_model.get('selection_metric')}",
            f"- Metric Value: {best_model.get('metric_value')}",
            f"- Reason: {best_model.get('reason')}",
            "",
            "## Interpretation Guide",
            "",
            "- MAE shows the average absolute prediction error.",
            "- RMSE penalizes larger forecasting errors more strongly.",
            "- MAPE shows percentage error, making it easier to explain to non-technical users.",
            "- R2 score shows how well the model explains variation in asset prices.",
            "",
            "## Recommendation",
            "",
        ]
    )

    if best_model.get("best_model"):
        lines.append(
            f"Use **{best_model.get('best_model')}** as the primary forecasting model because it achieved the best RMSE among available trained models."
        )
    else:
        lines.append(
            "No best model could be selected. Train the forecasting models first and rerun model comparison."
        )

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- Model Comparison CSV: `{MODEL_COMPARISON_CSV_PATH}`",
            f"- Model Comparison JSON: `{MODEL_COMPARISON_JSON_PATH}`",
            f"- Best Model JSON: `{BEST_MODEL_JSON_PATH}`",
            f"- Markdown Report: `{MODEL_COMPARISON_REPORT_PATH}`",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# Public Runner
# ============================================================

def run_model_comparison(
    selection_metric: str = "rmse",
    lower_is_better: bool = True,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """
    Run model comparison workflow.

    Args:
        selection_metric (str): Metric used for best model selection.
        lower_is_better (bool): Whether lower metric is better.
        save_outputs (bool): Whether to save outputs.

    Returns:
        Dict[str, Any]: Model comparison summary.
    """
    comparison_dataframe = create_model_comparison_dataframe(
        selection_metric=selection_metric,
        lower_is_better=lower_is_better,
    )

    best_model = select_best_model(
        comparison_dataframe=comparison_dataframe,
        selection_metric=selection_metric,
        lower_is_better=lower_is_better,
    )

    if save_outputs:
        save_model_comparison_outputs(
            comparison_dataframe=comparison_dataframe,
            best_model=best_model,
        )

        save_model_comparison_report(
            comparison_dataframe=comparison_dataframe,
            best_model=best_model,
        )

    return {
        "status": "success",
        "selection_metric": selection_metric,
        "lower_is_better": lower_is_better,
        "best_model": best_model,
        "model_count": int(len(comparison_dataframe)),
        "available_model_count": int(
            (comparison_dataframe["status"] == "available").sum()
        )
        if not comparison_dataframe.empty
        else 0,
        "comparison_records": comparison_dataframe.to_dict(orient="records"),
        "output_paths": {
            "model_comparison_csv": str(MODEL_COMPARISON_CSV_PATH),
            "model_comparison_json": str(MODEL_COMPARISON_JSON_PATH),
            "best_model_json": str(BEST_MODEL_JSON_PATH),
            "model_comparison_report": str(MODEL_COMPARISON_REPORT_PATH),
        },
    }


# ============================================================
# Script Usage
# ============================================================

if __name__ == "__main__":
    summary = run_model_comparison(
        selection_metric="rmse",
        lower_is_better=True,
        save_outputs=True,
    )

    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)

    print("Status:", summary.get("status"))
    print("Model Count:", summary.get("model_count"))
    print("Available Model Count:", summary.get("available_model_count"))

    best_model = summary.get("best_model", {})

    print("\nBest Model:")
    print("- Model:", best_model.get("best_model"))
    print("- Metric:", best_model.get("selection_metric"))
    print("- Value:", best_model.get("metric_value"))
    print("- Reason:", best_model.get("reason"))

    print("\nSaved Files:")
    for name, path in summary.get("output_paths", {}).items():
        print(f"- {name}: {path}")

    print("=" * 80)