"""The backtest engine: the event loop that ties everything together.

For each historical bar (a *heartbeat*) the engine:

1. checks stop-losses on existing positions and settles any forced exits;
2. lets the strategy emit fresh target weights;
3. drains the resulting Signal -> Order -> Fill cascade through the queue;
4. marks the book to market and records the equity point.

Settling stops and strategy trades in two separate drains keeps the accounting
unambiguous. The whole loop is deterministic: same data + config => same result.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from quant_engine.analytics.performance import PerformanceMetrics, compute_metrics
from quant_engine.analytics.tearsheet import save_tearsheet
from quant_engine.config import BacktestConfig
from quant_engine.core.events import Event, FillEvent, MarketEvent, OrderEvent, SignalEvent
from quant_engine.data.base import DataHandler
from quant_engine.execution.simulated import SimulatedExecutionHandler
from quant_engine.portfolio.portfolio import Portfolio
from quant_engine.strategy.base import Strategy


@dataclass
class BacktestResult:
    """Everything produced by a run, plus lazily-computed analytics."""

    equity_curve: pd.DataFrame
    fills: list[FillEvent]
    config: BacktestConfig
    strategy_id: str
    _metrics: PerformanceMetrics | None = field(default=None, init=False, repr=False)

    @property
    def metrics(self) -> PerformanceMetrics:
        if self._metrics is None:
            self._metrics = compute_metrics(self.equity_curve, self.config.periods_per_year)
        return self._metrics

    @property
    def n_trades(self) -> int:
        return len(self.fills)

    @property
    def total_costs(self) -> float:
        return float(sum(f.commission + f.slippage for f in self.fills))

    @property
    def annual_turnover(self) -> float:
        """One-way traded notional per year, as a multiple of average equity."""
        if self.equity_curve.empty:
            return 0.0
        traded = sum(abs(f.notional) for f in self.fills)
        avg_equity = float(self.equity_curve["equity"].mean())
        years = len(self.equity_curve) / self.config.periods_per_year
        if avg_equity <= 0 or years <= 0:
            return 0.0
        return traded / avg_equity / years

    def tearsheet(self, path: str | Path, title: str | None = None) -> Path:
        return save_tearsheet(
            self.equity_curve,
            self.metrics,
            path,
            title or self.strategy_id,
            self.config.periods_per_year,
        )

    def summary(self) -> str:
        lines = [
            f"Strategy            : {self.strategy_id}",
            self.metrics.summary(),
            f"Trades (fills)      : {self.n_trades:>9d}",
            f"Total costs         : {self.total_costs:>9,.0f}",
            f"Annual turnover     : {self.annual_turnover:>9.2f}x",
        ]
        return "\n".join(lines)


class BacktestEngine:
    def __init__(
        self,
        data: DataHandler,
        strategy: Strategy,
        config: BacktestConfig | None = None,
    ) -> None:
        self.data = data
        self.strategy = strategy
        self.config = config or BacktestConfig()
        self.events: deque[Event] = deque()

        self.portfolio = Portfolio(
            data, self.config.initial_capital, self.config.risk, self.config.periods_per_year
        )
        self.execution = SimulatedExecutionHandler(self.config.execution)

        # Wire every component to the single shared event queue.
        self.data.bind_queue(self.events)
        self.strategy.bind(data, self.events)
        self.portfolio.bind_queue(self.events)
        self.execution.bind(data, self.events)

    def _drain(self) -> None:
        """Process the queue until empty, routing each event to its handler."""
        while self.events:
            event = self.events.popleft()
            if isinstance(event, SignalEvent):
                self.portfolio.on_signal(event)
            elif isinstance(event, OrderEvent):
                self.execution.execute_order(event)
            elif isinstance(event, FillEvent):
                self.portfolio.on_fill(event)
            else:
                raise RuntimeError(f"unexpected {event.type} while draining queue")

    def _step(self, market_event: MarketEvent) -> None:
        """Process one heartbeat (shared by backtest and live/paper loops)."""
        self.portfolio.check_stops(market_event)
        self._drain()
        self.strategy.on_market(market_event)
        self._drain()
        self.portfolio.record_equity(market_event.timestamp)

    def _build_result(self) -> BacktestResult:
        return BacktestResult(
            equity_curve=self.portfolio.equity_frame(),
            fills=self.portfolio.fills,
            config=self.config,
            strategy_id=self.strategy.strategy_id,
        )

    def run(self) -> BacktestResult:
        self.strategy.on_start()
        while self.data.update_bars():
            event = self.events.popleft()
            if not isinstance(event, MarketEvent):
                raise RuntimeError(f"expected MARKET at heartbeat, got {event.type}")
            self._step(event)
        return self._build_result()
