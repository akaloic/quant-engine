"""Deterministic synthetic market data.

Real market data (Yahoo Finance) is great for demos but useless for tests and
CI: it changes daily and needs the network. So the engine ships a synthetic
generator based on **geometric Brownian motion** -- the standard textbook model
where log-returns are normal and prices compound multiplicatively.

Given a seed it is fully reproducible, which means backtests over synthetic data
are deterministic and can be asserted on in unit tests.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from quant_engine.core.events import Bar
from quant_engine.data.base import HistoricDataHandler

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def _ohlcv_from_close(close: np.ndarray, index: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """Build a plausible OHLCV frame from a close-price path.

    Open is the previous close (gap-free); high/low extend beyond the
    open/close range by a small random amount so candles look realistic.
    """
    open_ = np.empty_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0.0, 0.004, size=close.shape))
    high = np.maximum(open_, close) * (1.0 + wick)
    low = np.minimum(open_, close) * (1.0 - wick)
    volume = rng.lognormal(mean=12.0, sigma=0.4, size=close.shape)
    frame = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )
    frame.index.name = "timestamp"
    return frame


def generate_prices(
    symbols: list[str],
    n_bars: int = 756,
    start: str = "2021-01-04",
    mu: float = 0.08,
    sigma: float = 0.20,
    s0: float = 100.0,
    correlation: float = 0.0,
    periods_per_year: int = 252,
    seed: int = 7,
) -> dict[str, pd.DataFrame]:
    """Generate correlated GBM OHLCV frames, one per symbol.

    ``mu``/``sigma`` are *annualised* drift and volatility; ``correlation`` is
    the pairwise correlation of returns across symbols (equicorrelation).
    """
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start=start, periods=n_bars)
    k = len(symbols)
    dt = 1.0 / periods_per_year

    # Correlated standard-normal shocks via Cholesky of the correlation matrix.
    corr = np.full((k, k), correlation, dtype=float)
    np.fill_diagonal(corr, 1.0)
    chol = np.linalg.cholesky(corr)
    shocks = rng.standard_normal((n_bars, k)) @ chol.T

    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    frames: dict[str, pd.DataFrame] = {}
    for j, symbol in enumerate(symbols):
        log_returns = drift + diffusion * shocks[:, j]
        close = s0 * np.exp(np.cumsum(log_returns))
        frames[symbol] = _ohlcv_from_close(close, index, rng)
    return frames


def generate_cointegrated_pair(
    symbols: tuple[str, str] = ("PEP", "KO"),
    n_bars: int = 756,
    start: str = "2021-01-04",
    sigma: float = 0.20,
    s0: float = 100.0,
    spread_vol: float = 0.02,
    spread_halflife: float = 15.0,
    periods_per_year: int = 252,
    seed: int = 11,
) -> dict[str, pd.DataFrame]:
    """Generate two prices that share a common trend plus a mean-reverting spread.

    The first symbol follows GBM; the second is the first plus a stationary
    Ornstein-Uhlenbeck spread (an AR(1) in log-space). This is exactly the
    setup a pairs-trading strategy is designed to exploit, so it gives the
    strategy something real to trade against in tests and demos.
    """
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start=start, periods=n_bars)
    dt = 1.0 / periods_per_year

    drift = -0.5 * sigma**2 * dt
    diffusion = sigma * np.sqrt(dt)
    log_close_a = np.cumsum(drift + diffusion * rng.standard_normal(n_bars))

    # OU spread: phi sets the mean-reversion speed via the requested half-life.
    phi = float(np.exp(-np.log(2.0) / spread_halflife))
    spread = np.empty(n_bars)
    spread[0] = 0.0
    for t in range(1, n_bars):
        spread[t] = phi * spread[t - 1] + spread_vol * rng.standard_normal()

    close_a = s0 * np.exp(log_close_a)
    close_b = s0 * np.exp(log_close_a + spread)
    return {
        symbols[0]: _ohlcv_from_close(close_a, index, rng),
        symbols[1]: _ohlcv_from_close(close_b, index, rng),
    }


def frames_to_bars(
    frames: dict[str, pd.DataFrame],
) -> tuple[dict[str, list[Bar]], list[datetime]]:
    """Align frames on their common timestamps and convert to :class:`Bar` lists."""
    common: pd.Index | None = None
    for frame in frames.values():
        common = frame.index if common is None else common.intersection(frame.index)
    if common is None or len(common) == 0:
        raise ValueError("frames share no common timestamps")
    index = pd.DatetimeIndex(common.sort_values())
    timestamps = [ts.to_pydatetime() for ts in index]

    bars_by_symbol: dict[str, list[Bar]] = {}
    for symbol, frame in frames.items():
        aligned = frame.loc[index]
        opens = aligned["open"].to_numpy(dtype=float)
        highs = aligned["high"].to_numpy(dtype=float)
        lows = aligned["low"].to_numpy(dtype=float)
        closes = aligned["close"].to_numpy(dtype=float)
        volumes = aligned["volume"].to_numpy(dtype=float)
        bars_by_symbol[symbol] = [
            Bar(
                timestamp=timestamps[k],
                symbol=symbol,
                open=float(opens[k]),
                high=float(highs[k]),
                low=float(lows[k]),
                close=float(closes[k]),
                volume=float(volumes[k]),
            )
            for k in range(len(timestamps))
        ]
    return bars_by_symbol, timestamps


def make_handler(frames: dict[str, pd.DataFrame]) -> HistoricDataHandler:
    """Convenience: OHLCV frames -> a ready-to-replay :class:`HistoricDataHandler`."""
    bars_by_symbol, timestamps = frames_to_bars(frames)
    return HistoricDataHandler(bars_by_symbol, timestamps)
