"""Validation: promote raw -> validated, routing bad rows to quarantine.

This is the *silver* step. It reads the raw layer, applies the data contract
(:mod:`quant_engine.pipeline.contracts`) symbol by symbol, writes the clean rows
to ``validated`` and the rejects to ``quarantine`` with their failure reason,
and returns an aggregate :class:`ValidationReport`. The caller decides whether
the failure rate is tolerable.
"""

from __future__ import annotations

from pathlib import Path

from quant_engine.data.parquet_store import load_partitioned, save_partitioned
from quant_engine.pipeline.contracts import (
    CONTRACT_RULES,
    ValidationReport,
    detect_gaps,
    validate_frame,
)


def validate_layer(
    raw_root: str | Path,
    validated_root: str | Path,
    quarantine_root: str | Path,
) -> ValidationReport:
    """Validate every symbol in the raw layer and write the two output layers."""
    frames = load_partitioned(raw_root)

    valid_frames = {}
    quarantine_frames = {}
    n_input = n_valid = n_quarantined = gaps = 0
    failures = dict.fromkeys(CONTRACT_RULES, 0)

    for symbol, frame in frames.items():
        valid, quarantine, counts = validate_frame(frame)
        n_input += len(frame)
        n_valid += len(valid)
        n_quarantined += len(quarantine)
        gaps += detect_gaps(valid)
        for rule, count in counts.items():
            failures[rule] += count
        valid_frames[symbol] = valid
        if not quarantine.empty:
            quarantine_frames[symbol] = quarantine

    save_partitioned(valid_frames, validated_root)
    if quarantine_frames:
        save_partitioned(quarantine_frames, quarantine_root)

    return ValidationReport(
        n_input=n_input,
        n_valid=n_valid,
        n_quarantined=n_quarantined,
        gaps=gaps,
        failures_by_rule=failures,
    )
