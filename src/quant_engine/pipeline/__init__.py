"""Medallion data pipeline: raw -> validated -> curated, with data contracts.

A small but real data-engineering layer that sits *upstream* of the trading
engine. It ingests market data, enforces quality contracts (routing bad rows to
quarantine), and publishes a modelling-ready feature layer -- all reproducibly
and idempotently. ``quant-engine pipeline`` runs it locally; the Airflow DAG in
``airflow/dags/`` schedules the very same functions in production.
"""

from __future__ import annotations

from quant_engine.pipeline.contracts import (
    DataContractError,
    ValidationReport,
    detect_gaps,
    validate_frame,
)
from quant_engine.pipeline.flow import PipelineResult, run_pipeline
from quant_engine.pipeline.lake import LakeLayout

__all__ = [
    "DataContractError",
    "LakeLayout",
    "PipelineResult",
    "ValidationReport",
    "detect_gaps",
    "run_pipeline",
    "validate_frame",
]
