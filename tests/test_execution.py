from __future__ import annotations

from collections import deque

import pytest

from quant_engine.config import ExecutionConfig
from quant_engine.core.events import FillEvent, OrderEvent
from quant_engine.execution.simulated import SimulatedExecutionHandler


def _advance(handler, n=5):
    for _ in range(n):
        handler.update_bars()


def test_buy_pays_slippage_and_commission(handler):
    _advance(handler)
    events: deque = deque()
    ex = SimulatedExecutionHandler(ExecutionConfig(commission_bps=10, slippage_bps=20))
    ex.bind(handler, events)

    price = handler.latest_close("AAA")
    ex.execute_order(OrderEvent(handler.current_datetime(), "AAA", 100))
    fill = events[-1]
    assert isinstance(fill, FillEvent)

    expected_price = price * (1 + 20e-4)  # buy fills above the close
    assert fill.fill_price == pytest.approx(expected_price)
    assert fill.commission == pytest.approx(10e-4 * 100 * expected_price)
    assert fill.slippage == pytest.approx(100 * (expected_price - price))


def test_sell_receives_worse_price(handler):
    _advance(handler)
    events: deque = deque()
    ex = SimulatedExecutionHandler(ExecutionConfig(commission_bps=0, slippage_bps=20))
    ex.bind(handler, events)

    price = handler.latest_close("AAA")
    ex.execute_order(OrderEvent(handler.current_datetime(), "AAA", -100))
    fill = events[-1]
    assert fill.fill_price < price  # sell fills below the close


def test_min_commission_floor(handler):
    _advance(handler)
    events: deque = deque()
    ex = SimulatedExecutionHandler(ExecutionConfig(commission_bps=0.1, min_commission=5.0))
    ex.bind(handler, events)
    ex.execute_order(OrderEvent(handler.current_datetime(), "AAA", 1))
    assert events[-1].commission == pytest.approx(5.0)


def test_zero_quantity_emits_nothing(handler):
    _advance(handler)
    events: deque = deque()
    ex = SimulatedExecutionHandler(ExecutionConfig())
    ex.bind(handler, events)
    ex.execute_order(OrderEvent(handler.current_datetime(), "AAA", 0))
    assert not events
