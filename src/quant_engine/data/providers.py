"""Real market data via Yahoo Finance (optional dependency).

Install with ``pip install 'quant-engine[data]'``. Everything here is optional:
the engine runs fully on synthetic data without ``yfinance`` installed.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from quant_engine.data.synthetic import OHLCV_COLUMNS


def download_yfinance(
    symbols: list[str],
    start: date | str | None = None,
    end: date | str | None = None,
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """Download OHLCV bars and normalise them to the engine's schema.

    Returns a ``{symbol: DataFrame}`` mapping with lowercase
    ``open/high/low/close/volume`` columns indexed by timestamp.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "yfinance is required for the 'yfinance' data source. "
            "Install it with: pip install 'quant-engine[data]'"
        ) from exc

    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        raw = yf.download(
            symbol, start=start, end=end, interval=interval, auto_adjust=True, progress=False
        )
        if raw is None or raw.empty:
            continue
        # yfinance may return a MultiIndex column frame for a single ticker.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower)
        frame = raw[["open", "high", "low", "close", "volume"]].copy()
        frame.index.name = "timestamp"
        frames[symbol] = frame[OHLCV_COLUMNS].astype(float)
    return frames
