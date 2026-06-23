"""The :class:`Strategy` interface.

A strategy's only job is to turn market data into *target weights*. It reads
history through the :class:`DataHandler` (so it can never see the future) and
emits :class:`SignalEvent` s via :meth:`set_target`. It knows nothing about
account size, position sizing, costs, or order routing -- those belong to the
portfolio, risk and execution layers. That clean separation is what lets the
identical strategy object run unchanged in a backtest and in live paper trading.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque

import numpy as np

from quant_engine.core.events import Event, MarketEvent, SignalEvent
from quant_engine.data.base import DataHandler


class Strategy(ABC):
    """Base class for all trading strategies."""

    strategy_id: str = "strategy"
    #: Minimum bars of history before the strategy should act (engine/warmup hint).
    required_history: int = 1

    def __init__(self) -> None:
        self.data: DataHandler | None = None
        self._events: deque[Event] | None = None
        self._targets: dict[str, float] = {}

    def bind(self, data: DataHandler, events: deque[Event]) -> None:
        """Wire the strategy to the data feed and the shared event queue."""
        self.data = data
        self._events = events
        self._targets = {}

    @property
    def symbols(self) -> list[str]:
        return self.data.symbols if self.data is not None else []

    def on_start(self) -> None:
        """Optional hook called once before the first bar."""

    @abstractmethod
    def on_market(self, event: MarketEvent) -> None:
        """React to a new bar by calling :meth:`set_target` for symbols to trade."""

    # -- helpers ------------------------------------------------------------
    def closes(self, symbol: str, n: int) -> np.ndarray:
        """Last ``n`` closing prices for ``symbol`` (oldest first)."""
        assert self.data is not None
        return np.array([bar.close for bar in self.data.get_latest_bars(symbol, n)], dtype=float)

    def set_target(self, symbol: str, weight: float, *, eps: float = 1e-9) -> None:
        """Request a target portfolio weight in ``[-1, 1]`` for ``symbol``.

        A :class:`SignalEvent` is emitted only when the target actually changes,
        which keeps turnover (and order count) tied to genuine signal changes
        rather than to every heartbeat.
        """
        assert self.data is not None and self._events is not None
        weight = float(np.clip(weight, -1.0, 1.0))
        if abs(self._targets.get(symbol, 0.0) - weight) > eps:
            self._targets[symbol] = weight
            self._events.append(
                SignalEvent(self.data.current_datetime(), symbol, weight, self.strategy_id)
            )
