"""Ingestion: land raw market data in the lake's ``raw`` layer.

This is the *bronze* step -- pull from a source and persist it as-is, with no
cleaning, so the raw record is always recoverable. Writes are idempotent at
partition granularity (``save_partitioned`` overwrites a symbol/year partition
rather than appending), so re-running a day's load never duplicates rows. The
returned :class:`IngestResult` exposes the per-symbol high-watermark, which a
scheduled run uses to reason about incremental loads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant_engine.config import DataConfig
from quant_engine.data.parquet_store import load_partitioned, save_partitioned
from quant_engine.data.synthetic import generate_cointegrated_pair, generate_prices


@dataclass
class IngestResult:
    """What landed in the raw layer."""

    rows_by_symbol: dict[str, int]
    watermark: dict[str, str]  # symbol -> ISO timestamp of the most recent bar

    @property
    def total_rows(self) -> int:
        return sum(self.rows_by_symbol.values())


def _resolve_frames(config: DataConfig) -> dict[str, pd.DataFrame]:
    """Fetch raw OHLCV frames from the configured source.

    Self-contained on purpose: the data layer is upstream of (and independent
    from) the backtest engine, so ingestion never imports it.
    """
    if config.source == "synthetic":
        if config.params.get("kind") == "cointegrated":
            pair = (
                (config.symbols[0], config.symbols[1])
                if len(config.symbols) >= 2
                else ("PEP", "KO")
            )
            return generate_cointegrated_pair(symbols=pair, n_bars=config.bars, seed=config.seed)
        return generate_prices(
            config.symbols,
            n_bars=config.bars,
            seed=config.seed,
            mu=config.params.get("mu", 0.08),
            sigma=config.params.get("sigma", 0.20),
            correlation=config.params.get("correlation", 0.0),
        )

    if config.source == "yfinance":
        from quant_engine.data.providers import download_yfinance

        frames = download_yfinance(config.symbols, config.start, config.end)
        if not frames:
            raise RuntimeError(f"yfinance returned no data for {config.symbols}")
        return frames

    if config.source == "parquet":
        frames = load_partitioned(config.data_dir, config.symbols)
        if not frames:
            raise FileNotFoundError(
                f"no data found under {config.data_dir!r} for symbols {config.symbols}"
            )
        return frames

    raise ValueError(f"unknown data source {config.source!r}")


def ingest_raw(config: DataConfig, raw_root: str | Path) -> IngestResult:
    """Resolve frames from ``config`` and write them to the raw layer."""
    frames = _resolve_frames(config)
    save_partitioned(frames, raw_root)
    rows = {symbol: len(frame) for symbol, frame in frames.items()}
    watermark = {
        symbol: pd.Timestamp(frame.index.max()).isoformat()
        for symbol, frame in frames.items()
        if len(frame)
    }
    return IngestResult(rows_by_symbol=rows, watermark=watermark)
