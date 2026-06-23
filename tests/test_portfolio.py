from __future__ import annotations

from collections import deque

import pytest

from quant_engine.config import RiskConfig
from quant_engine.core.events import FillEvent
from quant_engine.portfolio.portfolio import Portfolio


def _portfolio(handler, capital=100_000.0):
    for _ in range(5):
        handler.update_bars()
    pf = Portfolio(handler, capital, RiskConfig(), 252)
    pf.bind_queue(deque())
    return pf


def test_fill_updates_cash_and_position(handler):
    pf = _portfolio(handler)
    price = handler.latest_close("AAA")
    pf.on_fill(FillEvent(handler.current_datetime(), "AAA", 10, price, commission=5.0))
    assert pf.positions["AAA"] == 10
    assert pf.cash == pytest.approx(100_000.0 - 10 * price - 5.0)
    assert pf.entry_price["AAA"] == pytest.approx(price)


def test_weighted_average_entry_price(handler):
    pf = _portfolio(handler)
    pf.on_fill(FillEvent(handler.current_datetime(), "AAA", 10, 100.0))
    pf.on_fill(FillEvent(handler.current_datetime(), "AAA", 10, 110.0))
    assert pf.positions["AAA"] == 20
    assert pf.entry_price["AAA"] == pytest.approx(105.0)


def test_flip_resets_entry_price(handler):
    pf = _portfolio(handler)
    pf.on_fill(FillEvent(handler.current_datetime(), "AAA", 10, 100.0))
    pf.on_fill(FillEvent(handler.current_datetime(), "AAA", -15, 120.0))  # flip to short
    assert pf.positions["AAA"] == -5
    assert pf.entry_price["AAA"] == pytest.approx(120.0)


def test_mark_to_market_equity(handler):
    pf = _portfolio(handler)
    price = handler.latest_close("AAA")
    pf.on_fill(FillEvent(handler.current_datetime(), "AAA", 10, price))
    pf.record_equity(handler.current_datetime())
    snapshot = pf.equity_history[-1]
    # No price move since the fill -> equity unchanged (ignoring zero commission here).
    assert snapshot["equity"] == pytest.approx(100_000.0)
    assert snapshot["net_exposure"] == pytest.approx(10 * price / 100_000.0)
