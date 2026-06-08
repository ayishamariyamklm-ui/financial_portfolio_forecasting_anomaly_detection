"""
Run Streamlit dashboard for the Financial Portfolio Forecasting
and Anomaly Detection project.

This script starts the Streamlit dashboard.

Main dashboard app:

    dashboard/app.py

Run from project root:

    python scripts/run_dashboard.py

Then open:

    http://localhost:8501

Optional command line usage:

    python scripts/run_dashboard.py --port 8501
    python scripts/run_dashboard.py --host 127.0.0.1 --port 8501
    python scripts/run_dashboard.py --server-headless true
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


# ============================================================
# Add Project Root to Python Path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ============================================================
# Project Paths
# ============================================================

DASHBOARD_APP_PATH = PROJECT_ROOT / "dashboard" / "app.py"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
DASHBOARD_PAGES_DIR = DASHBOARD_DIR / "pages"

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "financial_portfolio_data.csv"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"

LOGS_DIR = PROJECT_ROOT / "logs"


# ============================================================
# Helper Functions
# ============================================================

def ensure_dashboard_runtime_directories() -> None:
    """
    Create required directories before starting the dashboard.
    """
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    DASHBOARD_PAGES_DIR.mkdir(parents=True, exist_ok=True)

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


def check_dashboard_files() -> Dict[str, Any]:
    """
    Check important dashboard files before running Streamlit.

    Returns:
        Dict[str, Any]: Dashboard file status.
    """
    page_files = {
        "portfolio_overview": DASHBOARD_PAGES_DIR / "1_portfolio_overview.py",
        "price_forecasting": DASHBOARD_PAGES_DIR / "2_price_forecasting.py",
        "anomaly_detection": DASHBOARD_PAGES_DIR / "3_anomaly_detection.py",
        "model_performance": DASHBOARD_PAGES_DIR / "4_model_performance.py",
    }

    return {
        "dashboard_app_exists": DASHBOARD_APP_PATH.exists(),
        "dashboard_app_path": str(DASHBOARD_APP_PATH),
        "dashboard_dir_exists": DASHBOARD_DIR.exists(),
        "dashboard_pages_dir_exists": DASHBOARD_PAGES_DIR.exists(),
        "dataset_exists": DATA_PATH.exists(),
        "dataset_path": str(DATA_PATH),
        "page_files": {
            name: {
                "exists": path.exists(),
                "path": str(path),
            }
            for name, path in page_files.items()
        },
    }


def print_startup_banner(
    host: str,
    port: int,
    server_headless: str,
) -> None:
    """
    Print dashboard startup information.

    Args:
        host (str): Streamlit host.
        port (int): Streamlit port.
        server_headless (str): Streamlit headless mode.
    """
    file_status = check_dashboard_files()

    print("\n" + "=" * 80)
    print("FINANCIAL PORTFOLIO FORECASTING & ANOMALY DETECTION DASHBOARD")
    print("=" * 80)

    print("\nProject:")
    print("- Name: Financial Portfolio Forecasting & Anomaly Detection")
    print("- Month: Month 5 Practical")
    print("- Project Root:", PROJECT_ROOT)

    print("\nDashboard Server:")
    print("- Host:", host)
    print("- Port:", port)
    print("- Headless:", server_headless)

    print("\nImportant URLs:")
    print(f"- Dashboard: http://{host}:{port}")

    print("\nMain Dashboard:")
    print("- File:", DASHBOARD_APP_PATH)
    print("- Exists:", file_status["dashboard_app_exists"])

    print("\nDashboard Pages:")
    for page_name, page_status in file_status["page_files"].items():
        print(f"- {page_name}: {page_status['exists']}")

    print("\nDataset:")
    print("- Exists:", file_status["dataset_exists"])
    print("- Path:", file_status["dataset_path"])

    if not file_status["dataset_exists"]:
        print("\nDataset Warning:")
        print("- Dashboard can still open.")
        print("- But charts need this file:")
        print("  data/raw/financial_portfolio_data.csv")

    if not file_status["dashboard_app_exists"]:
        print("\nDashboard File Warning:")
        print("- dashboard/app.py was not found.")
        print("- Please create dashboard/app.py before running the dashboard.")

    print("\nRecommended Run Order:")
    print("1. python scripts/run_data_pipeline.py")
    print("2. python scripts/run_all_models.py")
    print("3. python scripts/evaluate_models.py")
    print("4. python scripts/run_dashboard.py")

    print("\nStop Dashboard:")
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
            "Run Streamlit dashboard for Financial Portfolio Forecasting "
            "and Anomaly Detection project."
        )
    )

    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host address for Streamlit server. Default: localhost",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port number for Streamlit server. Default: 8501",
    )

    parser.add_argument(
        "--server-headless",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Run Streamlit in headless mode. Default: false",
    )

    parser.add_argument(
        "--browser-gather-usage-stats",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Enable or disable Streamlit usage stats. Default: false",
    )

    parser.add_argument(
        "--theme-base",
        type=str,
        default="light",
        choices=["light", "dark"],
        help="Dashboard theme base. Default: light",
    )

    return parser.parse_args()


def build_streamlit_command(
    host: str,
    port: int,
    server_headless: str,
    browser_gather_usage_stats: str,
    theme_base: str,
) -> list[str]:
    """
    Build Streamlit run command.

    Args:
        host (str): Streamlit host.
        port (int): Streamlit port.
        server_headless (str): Headless mode.
        browser_gather_usage_stats (str): Usage stats option.
        theme_base (str): Theme base.

    Returns:
        list[str]: Command list.
    """
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(DASHBOARD_APP_PATH),
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--server.headless",
        server_headless,
        "--browser.gatherUsageStats",
        browser_gather_usage_stats,
        "--theme.base",
        theme_base,
    ]


def run_dashboard(
    host: str = "localhost",
    port: int = 8501,
    server_headless: str = "false",
    browser_gather_usage_stats: str = "false",
    theme_base: str = "light",
) -> None:
    """
    Run Streamlit dashboard.

    Args:
        host (str): Streamlit host.
        port (int): Streamlit port.
        server_headless (str): Streamlit headless mode.
        browser_gather_usage_stats (str): Usage stats option.
        theme_base (str): Streamlit theme base.
    """
    ensure_dashboard_runtime_directories()

    print_startup_banner(
        host=host,
        port=port,
        server_headless=server_headless,
    )

    if not DASHBOARD_APP_PATH.exists():
        raise FileNotFoundError(
            f"Dashboard app file not found at: {DASHBOARD_APP_PATH}\n"
            "Please create dashboard/app.py first."
        )

    command = build_streamlit_command(
        host=host,
        port=port,
        server_headless=server_headless,
        browser_gather_usage_stats=browser_gather_usage_stats,
        theme_base=theme_base,
    )

    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


# ============================================================
# Main Function
# ============================================================

def main() -> None:
    """
    Main entry point for dashboard server.
    """
    args = parse_arguments()

    try:
        run_dashboard(
            host=args.host,
            port=args.port,
            server_headless=args.server_headless,
            browser_gather_usage_stats=args.browser_gather_usage_stats,
            theme_base=args.theme_base,
        )

    except KeyboardInterrupt:
        print("\nDashboard stopped by user.")

    except FileNotFoundError as error:
        print("\n" + "=" * 80)
        print("FAILED TO START DASHBOARD")
        print("=" * 80)
        print("Error Type:", error.__class__.__name__)
        print("Error Message:", error)
        print("\nFix:")
        print("1. Make sure dashboard/app.py exists.")
        print("2. Make sure you created dashboard/pages/*.py files.")
        print("3. Run from project root:")
        print("   python scripts/run_dashboard.py")
        print("=" * 80)

    except subprocess.CalledProcessError as error:
        print("\n" + "=" * 80)
        print("STREAMLIT FAILED TO START")
        print("=" * 80)
        print("Error Type:", error.__class__.__name__)
        print("Return Code:", error.returncode)
        print("\nCommon fixes:")
        print("1. Activate your virtual environment.")
        print("2. Install Streamlit and Plotly:")
        print("   pip install streamlit plotly")
        print("3. Or install all project requirements:")
        print("   pip install -r requirements.txt")
        print("4. Run again:")
        print("   python scripts/run_dashboard.py")
        print("=" * 80)

    except Exception as error:
        print("\n" + "=" * 80)
        print("FAILED TO START DASHBOARD")
        print("=" * 80)
        print("Error Type:", error.__class__.__name__)
        print("Error Message:", error)
        print("\nCommon fixes:")
        print("1. Make sure your virtual environment is activated.")
        print("2. Install requirements:")
        print("   pip install -r requirements.txt")
        print("3. Make sure Streamlit is installed:")
        print("   pip install streamlit")
        print("4. Run from project root:")
        print("   python scripts/run_dashboard.py")
        print("=" * 80)


# ============================================================
# Script Entry Point
# ============================================================

if __name__ == "__main__":
    main()