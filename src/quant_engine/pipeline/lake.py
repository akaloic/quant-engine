"""On-disk layout of the data lake and the run manifest.

The lake follows the **medallion** convention: data flows through three layers
of increasing quality -- ``raw`` (as ingested), ``validated`` (passed the data
contract) and ``curated`` (modelling-ready features), with a ``quarantine`` area
for rows that failed validation. Each run drops a JSON manifest recording row
counts, the per-symbol high-watermark and the validation report -- lightweight
data lineage you can read without a database.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quant_engine.pipeline.contracts import ValidationReport
    from quant_engine.pipeline.curate import CurateResult
    from quant_engine.pipeline.ingest import IngestResult


@dataclass(frozen=True)
class LakeLayout:
    """Resolves the four medallion layers and the manifest under one root."""

    root: Path

    @classmethod
    def from_root(cls, root: str | Path) -> LakeLayout:
        return cls(Path(root))

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def validated(self) -> Path:
        return self.root / "validated"

    @property
    def curated(self) -> Path:
        return self.root / "curated"

    @property
    def quarantine(self) -> Path:
        return self.root / "quarantine"

    @property
    def manifest(self) -> Path:
        return self.root / "_manifest.json"


def write_manifest(
    layout: LakeLayout,
    *,
    source: str,
    symbols: list[str],
    ingest: IngestResult,
    report: ValidationReport,
    curate: CurateResult,
) -> Path:
    """Write a JSON lineage manifest for one pipeline run and return its path."""
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": source,
        "symbols": symbols,
        "rows_ingested": ingest.rows_by_symbol,
        "watermark": ingest.watermark,
        "validation": {**report.model_dump(), "quarantine_rate": report.quarantine_rate},
        "curated_rows": curate.rows_by_symbol,
    }
    layout.root.mkdir(parents=True, exist_ok=True)
    layout.manifest.write_text(json.dumps(payload, indent=2))
    return layout.manifest


def read_manifest(layout: LakeLayout) -> dict[str, object]:
    """Read the lineage manifest back (raises if the pipeline never ran)."""
    data: dict[str, object] = json.loads(layout.manifest.read_text())
    return data
