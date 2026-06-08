"""
Run API server for the Financial Portfolio Forecasting
and Anomaly Detection project.

This script starts the FastAPI application using Uvicorn.

Main API app:

    api/main.py

Run from project root:

    python scripts/run_api.py

Then open:

    http://127.0.0.1:8000
    http://127.0.0.1:8000/docs
    http://127.0.0.1:8000/health/full

Optional command line usage:

    python scripts/run_api.py --host 127.0.0.1 --port 8000 --reload
    python scripts/run_api.py --host 0.0.0.0 --port 8000
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import uvicorn


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

API_MAIN_PATH = PROJECT_ROOT / "api" / "main.py"
API_ROUTES_DIR = PROJECT_ROOT / "api" / "routes"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"


# ============================================================
# Helper Functions
# ============================================================

def ensure_api_runtime_directories() -> None:
    """
    Create required directories before starting the API server.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def check_api_files() -> Dict[str, Any]:
    """
    Check important API files before running the server.

    Returns:
        Dict[str, Any]: API file status.
    """
    return {
        "api_main_exists": API_MAIN_PATH.exists(),
        "api_main_path": str(API_MAIN_PATH),
        "routes_dir_exists": API_ROUTES_DIR.exists(),
        "routes_dir_path": str(API_ROUTES_DIR),
        "dataset_exists": DATA_PATH.exists(),
        "dataset_path": str(DATA_PATH),
    }


def print_startup_banner(
    host: str,
    port: int,
    reload: bool,
    workers: int,
) -> None:
    """
    Print API startup information.

    Args:
        host (str): API host.
        port (int): API port.
        reload (bool): Whether reload mode is enabled.
        workers (int): Number of workers.
    """
    file_status = check_api_files()

    print("\n" + "=" * 80)
    print("FINANCIAL PORTFOLIO FORECASTING & ANOMALY DETECTION API")
    print("=" * 80)

    print("\nProject:")
    print("- Name: Financial Portfolio Forecasting & Anomaly Detection")
    print("- Month: Month 5 Practical")
    print("- Project Root:", PROJECT_ROOT)

    print("\nServer:")
    print("- Host:", host)
    print("- Port:", port)
    print("- Reload:", reload)
    print("- Workers:", workers)

    print("\nImportant URLs:")
    print(f"- Home:        http://{host}:{port}")
    print(f"- API Docs:    http://{host}:{port}/docs")
    print(f"- ReDoc:       http://{host}:{port}/redoc")
    print(f"- Health:      http://{host}:{port}/health/")
    print(f"- Full Health: http://{host}:{port}/health/full")

    print("\nForecast URLs:")
    print(f"- Forecast Health:  http://{host}:{port}/forecast/health")
    print(f"- Forecast Default: http://{host}:{port}/forecast/default")
    print(f"- Forecast Latest:  http://{host}:{port}/forecast/latest")

    print("\nAnomaly URLs:")
    print(f"- Anomaly Health:  http://{host}:{port}/anomaly/health")
    print(f"- Anomaly Default: http://{host}:{port}/anomaly/default")
    print(f"- Anomaly Latest:  http://{host}:{port}/anomaly/latest")

    print("\nFile Status:")
    print("- api/main.py exists:", file_status["api_main_exists"])
    print("- api/routes exists:", file_status["routes_dir_exists"])
    print("- Dataset exists:", file_status["dataset_exists"])
    print("- Dataset path:", file_status["dataset_path"])

    if not file_status["dataset_exists"]:
        print("\nDataset Warning:")
        print("- Your API can still start.")
        print("- But /forecast/default and /anomaly/default need this file:")
        print("  data/raw/financial_portfolio_data.csv")

    if not file_status["api_main_exists"]:
        print("\nAPI File Warning:")
        print("- api/main.py was not found.")
        print("- Please create api/main.py before running the API.")

    print("\nStop Server:")
    print("- Press CTRL + C")

    print("=" * 80 + "\n")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run FastAPI server for Financial Portfolio Forecasting "
            "and Anomaly Detection project."
        )
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address for the API server. Default: 127.0.0.1",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number for the API server. Default: 8000",
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes. Default: 1",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Uvicorn log level. Default: info",
    )

    return parser.parse_args()


def run_api_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
    workers: int = 1,
    log_level: str = "info",
) -> None:
    """
    Run FastAPI server using Uvicorn.

    Args:
        host (str): Server host.
        port (int): Server port.
        reload (bool): Whether to enable auto-reload.
        workers (int): Number of workers.
        log_level (str): Uvicorn log level.
    """
    ensure_api_runtime_directories()

    print_startup_banner(
        host=host,
        port=port,
        reload=reload,
        workers=workers,
    )

    if not API_MAIN_PATH.exists():
        raise FileNotFoundError(
            f"API main file not found at: {API_MAIN_PATH}\n"
            "Please create api/main.py first."
        )

    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
        log_level=log_level,
    )


# ============================================================
# Main Function
# ============================================================

def main() -> None:
    """
    Main entry point for API server.
    """
    args = parse_arguments()

    try:
        run_api_server(
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers,
            log_level=args.log_level,
        )

    except KeyboardInterrupt:
        print("\nAPI server stopped by user.")

    except Exception as error:
        print("\n" + "=" * 80)
        print("FAILED TO START API SERVER")
        print("=" * 80)
        print("Error Type:", error.__class__.__name__)
        print("Error Message:", error)
        print("\nCommon fixes:")
        print("1. Make sure api/main.py exists.")
        print("2. Make sure your virtual environment is activated.")
        print("3. Install requirements:")
        print("   pip install -r requirements.txt")
        print("4. Make sure FastAPI and Uvicorn are installed:")
        print("   pip install fastapi uvicorn")
        print("5. Run from project root:")
        print("   python scripts/run_api.py --reload")
        print("=" * 80)


# ============================================================
# Script Entry Point
# ============================================================

if __name__ == "__main__":
    main()