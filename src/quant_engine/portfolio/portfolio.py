"""The portfolio: the system's book of record.

It consumes :class:`SignalEvent` s (target weights) and :class:`FillEvent` s
(executed trades), and it owns position sizing: turning a target weight into an
order quantity given current equity and price, after the risk manager has had
its say. Each heartbeat it marks positions to market and appends to the equity
curve, which is what all downstream performance analytics are computed from.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime

import pandas as pd

from quant_engine.config import RiskConfig
from quant_engine.core.events import Event, FillEvent, MarketEvent, OrderEvent, SignalEvent
from quant_engine.data.base import DataHandler
from quant_engine.risk.manager import RiskManager


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


class Portfolio:
    def __init__(
        self,
        data: DataHandler,
        initial_capital: float,
        risk_config: RiskConfig,
        periods_per_year: int,
    ) -> None:
        self.data = data
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.positions: dict[str, float] = {}
        self.entry_price: dict[str, float] = {}
        self.target_weights: dict[str, float] = {}
        self.risk = RiskManager(risk_config, periods_per_year)
        self.stop_loss_pct = risk_config.stop_loss_pct
        self._events: deque[Event] | None = None
        self.equity_history: list[dict[str, float | datetime]] = []
        self.fills: list[FillEvent] = []

    def bind_queue(self, events: deque[Event]) -> None:
        self._events = events

    def _emit(self, event: Event) -> None:
        assert self._events is not None
        self._events.append(event)

    # -- valuation ----------------------------------------------------------
    def market_value(self) -> float:
        total = 0.0
        for symbol, qty in self.positions.items():
            if qty == 0:
                continue
            price = self.data.latest_close(symbol)
            if price is not None:
                total += qty * price
        return total

    def total_equity(self) -> float:
        return self.cash + self.market_value()

    # -- event handlers -----------------------------------------------------
    def on_signal(self, signal: SignalEvent) -> None:
        self.target_weights[signal.symbol] = signal.target_weight
        self._rebalance(signal.symbol, signal.timestamp)

    def _rebalance(self, symbol: str, timestamp: datetime) -> None:
        price = self.data.latest_close(symbol)
        if price is None or price <= 0:
            return
        equity = self.total_equity()
        if equity <= 0:
            return
        weight = self.risk.adjusted_weight(symbol, self.target_weights, self.data)
        desired_qty = weight * equity / price
        delta = desired_qty - self.positions.get(symbol, 0.0)
        # Skip dust trades so floating-point drift doesn't churn the book.
        if abs(delta * price) < max(1.0, equity * 1e-6):
            return
        self._emit(OrderEvent(timestamp, symbol, delta))

    def check_stops(self, market_event: MarketEvent) -> None:
        """Flatten any position whose loss from entry exceeds the stop threshold."""
        if self.stop_loss_pct is None:
            return
        for symbol, qty in list(self.positions.items()):
            if qty == 0:
                continue
            price = self.data.latest_close(symbol)
            entry = self.entry_price.get(symbol)
            if price is None or entry is None or entry <= 0:
                continue
            pnl_return = (price / entry - 1.0) * _sign(qty)
            if pnl_return <= -self.stop_loss_pct:
                self.target_weights[symbol] = 0.0
                self._emit(OrderEvent(market_event.timestamp, symbol, -qty))

    def on_fill(self, fill: FillEvent) -> None:
        symbol = fill.symbol
        prev = self.positions.get(symbol, 0.0)
        new = prev + fill.quantity

        if prev == 0:
            self.entry_price[symbol] = fill.fill_price
        elif _sign(fill.quantity) == _sign(prev):  # adding to the position
            self.entry_price[symbol] = (
                self.entry_price[symbol] * prev + fill.fill_price * fill.quantity
            ) / new
        elif new == 0:  # fully closed
            self.entry_price.pop(symbol, None)
        elif _sign(new) != _sign(prev):  # flipped through zero
            self.entry_price[symbol] = fill.fill_price
        # else: reducing without flipping -> keep original entry price

        self.cash -= fill.notional  # signed: buy reduces cash, sell increases it
        self.cash -= fill.commission
        self.positions[symbol] = new
        self.fills.append(fill)

    def record_equity(self, timestamp: datetime) -> None:
        equity = self.total_equity()
        gross = 0.0
        net = 0.0
        for symbol, qty in self.positions.items():
            if qty == 0:
                continue
            price = self.data.latest_close(symbol)
            if price is None:
                continue
            value = qty * price
            gross += abs(value)
            net += value
        self.equity_history.append(
            {
                "timestamp": timestamp,
                "equity": equity,
                "cash": self.cash,
                "gross_exposure": gross / equity if equity > 0 else 0.0,
                "net_exposure": net / equity if equity > 0 else 0.0,
            }
        )

    def equity_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.equity_history)
        if frame.empty:
            return frame
        return frame.set_index("timestamp")
