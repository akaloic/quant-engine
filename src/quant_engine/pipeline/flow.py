"""The end-to-end pipeline: ingest -> validate -> (contract gate) -> curate.

This is the orchestration-agnostic core. The CLI (``quant-engine pipeline``) and
the Airflow DAG both drive these same functions, so the logic is identical and
fully tested in CI whether it runs locally or on a scheduler -- the same
"research-to-live parity" idea applied to the data layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from quant_engine.config import DataConfig, PipelineConfig
from quant_engine.pipeline.contracts import DataContractError, ValidationReport
from quant_engine.pipeline.curate import CurateResult, curate_layer
from quant_engine.pipeline.ingest import IngestResult, ingest_raw
from quant_engine.pipeline.lake import LakeLayout, write_manifest
from quant_engine.pipeline.validate import validate_layer


@dataclass
class PipelineResult:
    """Outcome of a full pipeline run, with a human-readable summary."""

    ingest: IngestResult
    validation: ValidationReport
    curate: CurateResult
    manifest_path: Path

    def summary(self) -> str:
        report = self.validation
        return "\n".join(
            [
                "Pipeline run complete:",
                f"  raw       : {self.ingest.total_rows} rows ingested",
                f"  validated : {report.n_valid} rows "
                f"({report.n_quarantined} quarantined, {report.quarantine_rate:.2%})",
                f"  gaps      : {report.gaps} missing business day(s)",
                f"  curated   : {self.curate.total_rows} feature rows",
                f"  manifest  : {self.manifest_path}",
            ]
        )


def run_pipeline(data: DataConfig, pipeline: PipelineConfig) -> PipelineResult:
    """Run ingest -> validate -> curate, enforcing the data contract in between.

    Raises :class:`DataContractError` if the quarantine rate exceeds
    ``pipeline.max_quarantine_rate`` -- bad data never reaches the curated layer.
    """
    layout = LakeLayout.from_root(pipeline.root)

    ingest = ingest_raw(data, layout.raw)
    report = validate_layer(layout.raw, layout.validated, layout.quarantine)
    if report.quarantine_rate > pipeline.max_quarantine_rate:
        raise DataContractError(
            f"data contract breached: {report.quarantine_rate:.2%} of rows quarantined "
            f"(limit {pipeline.max_quarantine_rate:.2%})"
        )
    curate = curate_layer(layout.validated, layout.curated)

    manifest_path = write_manifest(
        layout,
        source=data.source,
        symbols=data.symbols,
        ingest=ingest,
        report=report,
        curate=curate,
    )
    return PipelineResult(
        ingest=ingest, validation=report, curate=curate, manifest_path=manifest_path
    )
