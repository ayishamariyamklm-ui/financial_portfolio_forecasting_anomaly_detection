# ============================================================
# Financial Portfolio Forecasting & Anomaly Detection
# Dockerfile
# ============================================================

# Use lightweight Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Ensure Python output is shown immediately in logs
ENV PYTHONUNBUFFERED=1

# Set environment
ENV ENVIRONMENT=production

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for better Docker caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create required folders if they do not exist
RUN mkdir -p \
    data/raw \
    data/processed \
    data/interim \
    data/external \
    data/sample \
    models \
    reports \
    reports/figures \
    reports/tables \
    reports/predictions \
    outputs \
    logs

# Expose FastAPI port
EXPOSE 8000

# Expose Streamlit dashboard port
EXPOSE 8501

# Default command
# Runs FastAPI app
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]