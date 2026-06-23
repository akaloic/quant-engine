# Slim image that can serve either the REST API or the dashboard.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy only what the build needs first (better layer caching).
COPY pyproject.toml README.md ./
COPY src ./src
COPY dashboard ./dashboard
COPY examples ./examples

RUN pip install --upgrade pip && pip install ".[service,dashboard]"

EXPOSE 8000 8501

# Default: print a backtest summary. Override via docker-compose for api/dashboard.
CMD ["quant-engine", "backtest", "--strategy", "ma_crossover"]
