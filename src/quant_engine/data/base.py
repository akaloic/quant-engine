"""The :class:`DataHandler` abstraction.

A data handler is the engine's *only* window onto the market. It exposes two
things to the rest of the system:

* ``update_bars()`` -- advance time by one heartbeat and push a
  :class:`MarketEvent` onto the queue (returns ``False`` when data is
  exhausted, which ends a backtest).
* ``get_latest_bars(symbol, n)`` -- the last ``n`` bars *up to and including*
  the current heartbeat. Strategies build their indicators from this, which is
  what mechanically prevents look-ahead bias.

Backtests subclass this with a historical handler; live/paper trading subclasses
it with a handler that polls a broker/data feed. The rest of the engine cannot
tell the difference -- that is the research-to-live parity guarantee.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime

from quant_engine.core.events import Bar, Event, MarketEvent


class DataHandler(ABC):
    """Abstract market-data feed."""

    symbols: list[str]
    continue_backtest: bool = True

    def __init__(self) -> None:
        self._events: deque[Event] | None = None

    def bind_queue(self, events: deque[Event]) -> None:
        """Give the handler the shared event queue (called by the engine)."""
        self._events = events

    def _emit(self, event: Event) -> None:
        if self._events is None:
            raise RuntimeError("DataHandler is not bound to an event queue; call bind_queue().")
        self._events.append(event)

    @abstractmethod
    def update_bars(self) -> bool:
        """Advance one heartbeat and emit a :class:`MarketEvent`.

        Returns ``False`` when there is no more data (ends the backtest).
        """

    @abstractmethod
    def get_latest_bars(self, symbol: str, n: int = 1) -> list[Bar]:
        """Return the last ``n`` bars for ``symbol`` (oldest first)."""

    @abstractmethod
    def current_datetime(self) -> datetime:
        """Timestamp of the current heartbeat."""

    # -- convenience helpers built on the abstract methods ------------------
    def latest_close(self, symbol: str) -> float | None:
        """Most recent close for ``symbol`` (``None`` before the first bar)."""
        bars = self.get_latest_bars(symbol, 1)
        return bars[-1].close if bars else None

    def has_history(self, symbol: str, n: int) -> bool:
        """True once at least ``n`` bars are available for ``symbol``."""
        return len(self.get_latest_bars(symbol, n)) >= n


class HistoricDataHandler(DataHandler):
    """Replays in-memory OHLCV frames bar-by-bar.

    Symbols are aligned on their **common timestamps** so that every symbol has
    a genuine (non-imputed) bar at each heartbeat -- this keeps multi-asset
    strategies honest and the accounting simple.
    """

    def __init__(self, bars_by_symbol: dict[str, list[Bar]], timestamps: list[datetime]) -> None:
        super().__init__()
        self.symbols = list(bars_by_symbol.keys())
        self._bars = bars_by_symbol
        self.timestamps = timestamps
        self._i = -1
        for sym, bars in bars_by_symbol.items():
            if len(bars) != len(timestamps):
                raise ValueError(
                    f"symbol {sym!r} has {len(bars)} bars but {len(timestamps)} timestamps"
                )

    def update_bars(self) -> bool:
        if self._i + 1 >= len(self.timestamps):
            self.continue_backtest = False
            return False
        self._i += 1
        self._emit(MarketEvent(self.timestamps[self._i]))
        return True

    def get_latest_bars(self, symbol: str, n: int = 1) -> list[Bar]:
        if self._i < 0:
            return []
        start = max(0, self._i - n + 1)
        return self._bars[symbol][start : self._i + 1]

    def current_datetime(self) -> datetime:
        if self._i < 0:
            raise RuntimeError("No bars have been emitted yet; call update_bars() first.")
        return self.timestamps[self._i]
