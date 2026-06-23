from __future__ import annotations

import numpy as np
import pandas as pd

from quant_engine.data.parquet_store import load_partitioned, save_partitioned
from quant_engine.data.synthetic import generate_prices


def test_generation_is_deterministic():
    a = generate_prices(["AAA"], n_bars=120, seed=1)["AAA"]
    b = generate_prices(["AAA"], n_bars=120, seed=1)["AAA"]
    pd.testing.assert_frame_equal(a, b)
    c = generate_prices(["AAA"], n_bars=120, seed=2)["AAA"]
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_ohlc_is_internally_consistent():
    df = generate_prices(["AAA"], n_bars=80, seed=5)["AAA"]
    assert (df["high"] >= df["low"]).all()
    assert (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (df["close"] > 0).all()


def test_handler_never_reveals_the_future(handler):
    while handler.update_bars():
        current = handler.current_datetime()
        bars = handler.get_latest_bars("AAA", 10)
        assert bars, "expected at least one bar once started"
        assert bars[-1].timestamp == current
        assert all(bar.timestamp <= current for bar in bars)


def test_get_latest_bars_window_length(handler):
    for _ in range(30):
        handler.update_bars()
    assert len(handler.get_latest_bars("AAA", 5)) == 5
    assert len(handler.get_latest_bars("AAA", 10000)) == 30  # capped at available history


def test_parquet_roundtrip(tmp_path, frames):
    save_partitioned(frames, tmp_path / "store")
    loaded = load_partitioned(tmp_path / "store", ["AAA"])
    assert set(loaded) == {"AAA"}
    np.testing.assert_allclose(
        loaded["AAA"]["close"].to_numpy(), frames["AAA"]["close"].to_numpy()
    )
