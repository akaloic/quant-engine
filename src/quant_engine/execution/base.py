"""The :class:`ExecutionHandler` abstraction.

It consumes :class:`OrderEvent` s and produces :class:`FillEvent` s. A backtest
uses the simulated handler (models costs against historical prices); live paper
trading swaps in a handler that sends orders to a broker. Because both speak the
same Order/Fill event contract, nothing upstream changes between the two.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque

from quant_engine.core.events import Event, OrderEvent
from quant_engine.data.base import DataHandler


class ExecutionHandler(ABC):
    def __init__(self) -> None:
        self.data: DataHandler | None = None
        self._events: deque[Event] | None = None

    def bind(self, data: DataHandler, events: deque[Event]) -> None:
        self.data = data
        self._events = events

    def _emit(self, event: Event) -> None:
        assert self._events is not None
        self._events.append(event)

    @abstractmethod
    def execute_order(self, order: OrderEvent) -> None:
        """Fill ``order`` and emit a :class:`FillEvent`."""
