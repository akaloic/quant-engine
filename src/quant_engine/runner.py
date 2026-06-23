"""High-level orchestration: turn a :class:`RunConfig` into a backtest result.

This is the single entry point shared by the CLI, the REST service and the
dashboard, so a run behaves identically however it is launched.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from quant_engine.config import DataConfig, RunConfig
from quant_engine.data.base import HistoricDataHandler
from quant_engine.data.parquet_store import load_partitioned
from quant_engine.data.synthetic import (
    generate_cointegrated_pair,
    generate_prices,
    make_handler,
)
from quant_engine.engine.backtest import BacktestEngine, BacktestResult
from quant_engine.strategy.registry import create_strategy


def _slice_dates(
    frames: dict[str, pd.DataFrame], start: date | None, end: date | None
) -> dict[str, pd.DataFrame]:
    if start is None and end is None:
        return frames
    sliced = {}
    for symbol, frame in frames.items():
        mask = pd.Series(True, index=frame.index)
        if start is not None:
            mask &= frame.index >= pd.Timestamp(start)
        if end is not None:
            mask &= frame.index <= pd.Timestamp(end)
        sliced[symbol] = frame[mask]
    return sliced


def build_frames(config: DataConfig) -> dict[str, pd.DataFrame]:
    """Resolve a data configuration into raw OHLCV frames (one per symbol)."""
    params: dict[str, Any] = config.params

    if config.source == "synthetic":
        if params.get("kind") == "cointegrated":
            if len(config.symbols) >= 2:
                pair = (config.symbols[0], config.symbols[1])
            else:
                pair = ("PEP", "KO")
            return generate_cointegrated_pair(symbols=pair, n_bars=config.bars, seed=config.seed)
        return generate_prices(
            config.symbols,
            n_bars=config.bars,
            seed=config.seed,
            mu=params.get("mu", 0.08),
            sigma=params.get("sigma", 0.20),
            correlation=params.get("correlation", 0.0),
        )

    if config.source == "parquet":
        frames = load_partitioned(config.data_dir, config.symbols)
        if not frames:
            raise FileNotFoundError(
                f"no data found under {config.data_dir!r} for symbols {config.symbols}"
            )
        return _slice_dates(frames, config.start, config.end)

    if config.source == "yfinance":
        from quant_engine.data.providers import download_yfinance

        frames = download_yfinance(config.symbols, config.start, config.end)
        if not frames:
            raise RuntimeError(f"yfinance returned no data for {config.symbols}")
        return frames

    raise ValueError(f"unknown data source {config.source!r}")


def build_data_handler(config: DataConfig) -> HistoricDataHandler:
    """Construct a :class:`HistoricDataHandler` from a data configuration."""
    return make_handler(build_frames(config))


def run_backtest(config: RunConfig) -> BacktestResult:
    """Build data + strategy from ``config`` and run a full backtest."""
    data = build_data_handler(config.data)
    strategy = create_strategy(config.strategy.name, config.strategy.params)
    engine = BacktestEngine(data, strategy, config.backtest)
    return engine.run()


def load_run_config(path: str | Path) -> RunConfig:
    """Load a :class:`RunConfig` from a YAML file."""
    with open(path) as handle:
        raw = yaml.safe_load(handle) or {}
    return RunConfig(**raw)
