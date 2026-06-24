# Airflow orchestration

The DAG in [`dags/quant_pipeline_dag.py`](dags/quant_pipeline_dag.py) schedules the
medallion data pipeline in production. It is a **thin orchestration wrapper**: all
the real logic lives in the `quant_engine.pipeline` package and is exercised by the
test-suite in CI, so what Airflow runs nightly is exactly what runs under
`quant-engine pipeline` and in `pytest` — no second implementation to drift.

```
ingest ──▶ validate ──▶ curate
(raw)      (silver +     (gold /
            quarantine)   feature store)
```

- **`ingest`** lands raw OHLCV into the `raw/` layer (idempotent partition writes).
- **`validate`** applies the data contract, routes bad rows to `quarantine/`, and
  **fails the run** if the quarantine rate exceeds `QUANT_MAX_QUARANTINE_RATE` — so
  bad data never reaches `curated/`. Failed tasks retry per `default_args`.
- **`curate`** builds the modelling-ready feature layer consumed by the backtester
  and the ML signal.

## Run it locally

```bash
pip install "quant-engine[airflow]"            # installs Apache Airflow
export AIRFLOW_HOME="$PWD/airflow"
export QUANT_LAKE_ROOT="$PWD/data/lake"
airflow standalone                             # UI on http://localhost:8080
```

The DAG `quant_market_data_pipeline` then appears in the UI. Configuration is read
from environment variables (`QUANT_LAKE_ROOT`, `QUANT_SYMBOLS`, `QUANT_SOURCE`,
`QUANT_MAX_QUARANTINE_RATE`), so one DAG serves every environment.

> Airflow is an optional extra and is **not** installed in CI — the pipeline logic
> is validated directly against `quant_engine.pipeline`. A DAG-integrity test
> (`tests/test_pipeline_dag.py`) runs only when Airflow is present.
