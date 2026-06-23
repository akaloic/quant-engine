from __future__ import annotations

from datetime import datetime

import pytest

from quant_engine.core.events import (
    Direction,
    EventType,
    FillEvent,
    MarketEvent,
    OrderEvent,
    SignalEvent,
)

TS = datetime(2024, 1, 1)


def test_event_types_are_tagged():
    assert MarketEvent(TS).type == EventType.MARKET
    assert SignalEvent(TS, "AAA", 0.5).type == EventType.SIGNAL
    assert OrderEvent(TS, "AAA", 10).type == EventType.ORDER
    assert FillEvent(TS, "AAA", 10, 100.0).type == EventType.FILL


def test_signal_weight_must_be_bounded():
    with pytest.raises(ValueError):
        SignalEvent(TS, "AAA", 1.5)
    with pytest.raises(ValueError):
        SignalEvent(TS, "AAA", -1.5)


def test_order_direction():
    assert OrderEvent(TS, "AAA", 5).direction == Direction.LONG
    assert OrderEvent(TS, "AAA", -5).direction == Direction.SHORT
    assert OrderEvent(TS, "AAA", 0).direction == Direction.FLAT


def test_fill_notional_is_signed():
    assert FillEvent(TS, "AAA", 10, 100.0).notional == pytest.approx(1000.0)
    assert FillEvent(TS, "AAA", -10, 100.0).notional == pytest.approx(-1000.0)
