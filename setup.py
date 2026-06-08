"""
Setup configuration for the Financial Portfolio Forecasting
and Anomaly Detection project.

This file allows the project to be installed as a local Python package.
"""

from setuptools import find_packages, setup


PROJECT_NAME = "financial-portfolio-forecasting-anomaly-detection"
VERSION = "0.1.0"
AUTHOR = "Ayisha Mariyam"
AUTHOR_EMAIL = "ayishamariyamklm@gmail.com"
DESCRIPTION = (
    "A data science project for financial portfolio forecasting "
    "and anomaly detection using time series analysis and machine learning."
)


def read_requirements(file_path: str = "requirements.txt") -> list:
    """
    Read package requirements from requirements.txt.

    Args:
        file_path (str): Path to the requirements file.

    Returns:
        list: A list of required packages.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            requirements = [
                line.strip()
                for line in file.readlines()
                if line.strip() and not line.startswith("#")
            ]
        return requirements
    except FileNotFoundError:
        return []


def read_long_description(file_path: str = "README.md") -> str:
    """
    Read the long project description from README.md.

    Args:
        file_path (str): Path to the README file.

    Returns:
        str: README content as a string.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return DESCRIPTION


setup(
    name=PROJECT_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=read_requirements(),
    python_requires=">=3.10",
    include_package_data=True,
    keywords=[
        "data-science",
        "machine-learning",
        "time-series",
        "forecasting",
        "anomaly-detection",
        "finance",
        "portfolio-analysis",
        "mlops",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Source": "https://github.com/ayishamariyamklm-ui/financial-portfolio-forecasting-anomaly-detection",
        "Author": "https://github.com/ayishamariyamklm-ui",
    },
)