"""Curation: promote validated -> curated (the modelling-ready feature layer).

This is the *gold* step and doubles as a small **feature store**: it turns clean
prices into the exact causal feature set the ML signal consumes
(:mod:`quant_engine.ml.features`), drops the warm-up rows that are NaN while
rolling windows fill, and persists the result partitioned by symbol/year. A
backtest or a training run then reads features straight off the curated layer
instead of recomputing them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from quant_engine.data.parquet_store import load_dataset, load_partitioned, save_partitioned
from quant_engine.ml.features import FEATURE_NAMES, build_feature_frame


@dataclass
class CurateResult:
    """What landed in the curated layer."""

    rows_by_symbol: dict[str, int]
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))

    @property
    def total_rows(self) -> int:
        return sum(self.rows_by_symbol.values())


def curate_layer(validated_root: str | Path, curated_root: str | Path) -> CurateResult:
    """Build the feature set for every validated symbol and persist it."""
    frames = load_partitioned(validated_root)

    curated: dict[str, pd.DataFrame] = {}
    for symbol, frame in frames.items():
        close = frame["close"].astype(float)
        features = build_feature_frame(close)
        features["close"] = close
        curated[symbol] = features.dropna()

    save_partitioned(curated, curated_root)
    rows = {symbol: len(frame) for symbol, frame in curated.items()}
    return CurateResult(rows_by_symbol=rows)


def load_curated(
    curated_root: str | Path, symbols: list[str] | None = None
) -> dict[str, pd.DataFrame]:
    """Read the curated feature layer back (features + ``close`` per symbol)."""
    return load_dataset(curated_root, symbols)
