"""Live / paper-trading loop -- the research-to-live parity guarantee in code.

:class:`LivePaperEngine` subclasses :class:`BacktestEngine` and reuses its
component wiring and its per-heartbeat ``_step`` logic *unchanged*. The only
differences are operational: it can pace itself in real time and it calls an
``on_bar`` hook after every bar so an external process (dashboard, logger,
alerting) can observe live state.

To go from paper to real money you would swap two objects -- a broker-backed
``DataHandler`` and a broker-backed ``ExecutionHandler`` -- and the strategy,
portfolio, risk and analytics code would not change at all.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from quant_engine.config import BacktestConfig
from quant_engine.core.events import MarketEvent
from quant_engine.data.base import DataHandler
from quant_engine.engine.backtest import BacktestEngine, BacktestResult
from quant_engine.strategy.base import Strategy

Snapshot = dict[str, Any]


class LivePaperEngine(BacktestEngine):
    def __init__(
        self,
        data: DataHandler,
        strategy: Strategy,
        config: BacktestConfig | None = None,
        poll_interval: float = 0.0,
        on_bar: Callable[[Snapshot], None] | None = None,
    ) -> None:
        super().__init__(data, strategy, config)
        self.poll_interval = poll_interval
        self.on_bar = on_bar

    def snapshot(self, timestamp: datetime) -> Snapshot:
        """Current portfolio state, suitable for logging or a live dashboard."""
        return {
            "timestamp": timestamp,
            "equity": self.portfolio.total_equity(),
            "cash": self.portfolio.cash,
            "positions": {s: q for s, q in self.portfolio.positions.items() if q},
            "target_weights": {s: w for s, w in self.portfolio.target_weights.items() if w},
        }

    def run_live(self, max_bars: int | None = None) -> BacktestResult:
        """Run the live/paper loop, pacing at ``poll_interval`` seconds per bar."""
        self.strategy.on_start()
        processed = 0
        while self.data.update_bars():
            event = self.events.popleft()
            if not isinstance(event, MarketEvent):
                raise RuntimeError(f"expected MARKET at heartbeat, got {event.type}")
            self._step(event)

            if self.on_bar is not None:
                self.on_bar(self.snapshot(event.timestamp))

            processed += 1
            if max_bars is not None and processed >= max_bars:
                break
            if self.poll_interval > 0:
                time.sleep(self.poll_interval)
        return self._build_result()
