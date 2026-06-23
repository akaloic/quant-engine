"""Shared fixtures: deterministic synthetic data and handlers."""

from __future__ import annotations

from collections import deque

import pandas as pd
import pytest

from quant_engine.data.base import HistoricDataHandler
from quant_engine.data.synthetic import (
    generate_cointegrated_pair,
    generate_prices,
    make_handler,
)


@pytest.fixture
def frames() -> dict[str, pd.DataFrame]:
    return generate_prices(["AAA", "BBB", "CCC"], n_bars=300, seed=42)


@pytest.fixture
def handler(frames: dict[str, pd.DataFrame]) -> HistoricDataHandler:
    # Bind a throwaway queue so the handler can be stepped standalone in tests
    # (the engine binds its own shared queue in real use).
    feed = make_handler(frames)
    feed.bind_queue(deque())
    return feed


@pytest.fixture
def pair_frames() -> dict[str, pd.DataFrame]:
    return generate_cointegrated_pair(("XXX", "YYY"), n_bars=400, seed=3)
