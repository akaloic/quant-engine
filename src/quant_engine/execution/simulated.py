"""Simulated execution with a realistic cost model.

Real fills are never free or at the mid: you pay commission, and your own order
pushes the price against you (slippage + market impact). Ignoring these is the
single most common way backtests lie. This handler fills market orders at the
current bar's close (equivalent to a market-on-close order) and charges:

* **commission** = ``commission_bps`` of notional, floored at ``min_commission``;
* **slippage**   = ``slippage_bps`` applied *against* the trade direction;
* **market impact** (optional) = extra bps proportional to how large the order
  is versus recent average volume (participation rate).
"""

from __future__ import annotations

import numpy as np

from quant_engine.config import ExecutionConfig
from quant_engine.core.events import FillEvent, OrderEvent
from quant_engine.execution.base import ExecutionHandler

_BPS = 1e-4


class SimulatedExecutionHandler(ExecutionHandler):
    def __init__(self, config: ExecutionConfig, adv_window: int = 20) -> None:
        super().__init__()
        self.config = config
        self.adv_window = adv_window

    def _impact_bps(self, order: OrderEvent) -> float:
        if self.config.impact_coefficient <= 0:
            return 0.0
        assert self.data is not None
        bars = self.data.get_latest_bars(order.symbol, self.adv_window)
        if not bars:
            return 0.0
        adv = float(np.mean([b.volume for b in bars]))
        if adv <= 0:
            return 0.0
        participation = abs(order.quantity) / adv
        return self.config.impact_coefficient * participation

    def execute_order(self, order: OrderEvent) -> None:
        assert self.data is not None
        price = self.data.latest_close(order.symbol)
        if price is None or price <= 0 or order.quantity == 0:
            return

        direction = 1 if order.quantity > 0 else -1
        slippage_bps = self.config.slippage_bps + self._impact_bps(order)
        fill_price = price * (1.0 + direction * slippage_bps * _BPS)

        notional = abs(order.quantity * fill_price)
        commission = max(self.config.min_commission, self.config.commission_bps * _BPS * notional)
        slippage_cost = abs(order.quantity) * abs(fill_price - price)

        self._emit(
            FillEvent(
                timestamp=order.timestamp,
                symbol=order.symbol,
                quantity=order.quantity,
                fill_price=fill_price,
                commission=commission,
                slippage=slippage_cost,
            )
        )
