"""Airflow DAG: daily market-data medallion pipeline.

Schedules the **same** functions that ``quant-engine pipeline`` runs locally
(`quant_engine.pipeline`), so the orchestrated production path and the
CLI/CI path share one implementation. The graph is three tasks --
``ingest -> validate -> curate`` -- with retries, and the ``validate`` task
enforces the data contract: if too many rows are quarantined it fails the run,
so bad data never reaches the curated layer or any downstream backtest.

Deploy by copying this file into your ``$AIRFLOW_HOME/dags`` folder with
``quant-engine`` installed in the workers' environment. Configuration is read
from environment variables so the same DAG serves every environment:

* ``QUANT_LAKE_ROOT``  -- lake root (default ``data/lake``)
* ``QUANT_SYMBOLS``    -- comma-separated universe (default ``AAA,BBB,CCC,DDD,EEE``)
* ``QUANT_SOURCE``     -- ``synthetic`` | ``yfinance`` | ``parquet``
* ``QUANT_MAX_QUARANTINE_RATE`` -- data-contract threshold (default ``0.02``)

Heavy imports live inside the tasks (not at module top level) so DAG parsing
stays fast, per Airflow best practice.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow.decorators import dag, task

LAKE_ROOT = os.environ.get("QUANT_LAKE_ROOT", "data/lake")
SYMBOLS = [s for s in os.environ.get("QUANT_SYMBOLS", "AAA,BBB,CCC,DDD,EEE").split(",") if s]
SOURCE = os.environ.get("QUANT_SOURCE", "synthetic")
MAX_QUARANTINE_RATE = float(os.environ.get("QUANT_MAX_QUARANTINE_RATE", "0.02"))

DEFAULT_ARGS = {
    "owner": "quant",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="quant_market_data_pipeline",
    description="Ingest -> validate -> curate market data into a medallion lake.",
    schedule="0 18 * * 1-5",  # 18:00 on weekdays, after the US close
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["quant", "market-data", "medallion"],
    doc_md=__doc__,
)
def quant_market_data_pipeline() -> None:
    @task
    def ingest() -> dict[str, object]:
        from quant_engine.config import DataConfig
        from quant_engine.pipeline.ingest import ingest_raw
        from quant_engine.pipeline.lake import LakeLayout

        layout = LakeLayout.from_root(LAKE_ROOT)
        result = ingest_raw(DataConfig(source=SOURCE, symbols=SYMBOLS), layout.raw)
        return {"rows": result.total_rows, "watermark": result.watermark}

    @task
    def validate(upstream: dict[str, object]) -> dict[str, object]:
        from quant_engine.pipeline.lake import LakeLayout
        from quant_engine.pipeline.validate import validate_layer

        layout = LakeLayout.from_root(LAKE_ROOT)
        report = validate_layer(layout.raw, layout.validated, layout.quarantine)
        if report.quarantine_rate > MAX_QUARANTINE_RATE:
            raise ValueError(
                f"data contract breached: {report.quarantine_rate:.2%} quarantined "
                f"(limit {MAX_QUARANTINE_RATE:.2%})"
            )
        return report.model_dump()

    @task
    def curate(upstream: dict[str, object]) -> dict[str, object]:
        from quant_engine.pipeline.curate import curate_layer
        from quant_engine.pipeline.lake import LakeLayout

        layout = LakeLayout.from_root(LAKE_ROOT)
        result = curate_layer(layout.validated, layout.curated)
        return {"rows": result.total_rows}

    curate(validate(ingest()))


quant_market_data_pipeline()
