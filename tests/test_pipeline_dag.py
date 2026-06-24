"""Airflow DAG integrity. Skipped unless Airflow is installed (it isn't in CI)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("airflow")


def test_dag_imports_cleanly_with_three_tasks() -> None:
    from airflow.models import DagBag

    dag_folder = Path(__file__).resolve().parents[1] / "airflow" / "dags"
    bag = DagBag(dag_folder=str(dag_folder), include_examples=False)

    assert bag.import_errors == {}
    dag = bag.get_dag("quant_market_data_pipeline")
    assert dag is not None
    assert {task.task_id for task in dag.tasks} == {"ingest", "validate", "curate"}
