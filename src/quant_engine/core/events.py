"""Event types that flow through the engine's central queue.

The whole system is *event-driven*: components never call each other directly.
Instead they push typed events onto a single FIFO queue and the engine drains
that queue one event at a time. This is the same architecture used by
production engines such as NautilusTrader, and it buys us three things:

1. **No look-ahead bias.** A strategy only ever reacts to a ``MarketEvent`` for
   the *current* bar; it physically cannot see future data.
2. **Research-to-live parity.** The exact same event types and handlers run in
   the historical backtest and in the live paper-trading loop. Only the data
   source and the execution handler are swapped.
3. **Separation of concerns.** Data -> Strategy -> Risk/Portfolio -> Execution
   are decoupled stages that each consume one event type and emit another.

The event flow for a single trade is::

    MarketEvent  --(Strategy)-->  SignalEvent
    SignalEvent  --(Portfolio + RiskManager)-->  OrderEvent
    OrderEvent   --(ExecutionHandler)-->  FillEvent
    FillEvent    --(Portfolio)-->  position & cash updated
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum, StrEnum


class EventType(StrEnum):
    """Discriminator carried by every event."""

    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"


class Direction(IntEnum):
    """Sign convention shared across signals, orders and positions."""

    LONG = 1
    FLAT = 0
    SHORT = -1


class OrderType(StrEnum):
    """Supported order types.

    Only ``MARKET`` is simulated today; the enum exists so that limit/stop
    orders can be added without touching the event contract.
    """

    MARKET = "MARKET"


@dataclass(frozen=True, slots=True)
class Bar:
    """A single OHLCV price bar for one symbol.

    Bars are the atomic unit of market data. ``close`` is what the engine marks
    positions to and what the simulated execution handler fills against.
    """

    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass(slots=True)
class Event:
    """Base class for all events (used only for typing / ``isinstance``)."""

    type: EventType = field(init=False)


@dataclass(slots=True)
class MarketEvent(Event):
    """Emitted once per heartbeat when a new bar (or set of bars) is available.

    It carries no payload beyond the timestamp: handlers pull the data they are
    allowed to see from the :class:`DataHandler`, which guarantees they only get
    history up to and including ``timestamp``.
    """

    timestamp: datetime

    def __post_init__(self) -> None:
        self.type = EventType.MARKET


@dataclass(slots=True)
class SignalEvent(Event):
    """A strategy's *desired exposure* for one symbol.

    ``target_weight`` is the fraction of total portfolio equity the strategy
    wants allocated to ``symbol``: ``+1.0`` = fully long, ``-1.0`` = fully
    short, ``0.0`` = flat. Expressing intent as a target weight (rather than a
    raw quantity) keeps strategies decoupled from account size and lets the
    portfolio/risk layer own position sizing.
    """

    timestamp: datetime
    symbol: str
    target_weight: float
    strategy_id: str = "strategy"

    def __post_init__(self) -> None:
        self.type = EventType.SIGNAL
        if not -1.0 <= self.target_weight <= 1.0:
            raise ValueError(
                f"target_weight must be in [-1, 1], got {self.target_weight!r}"
            )


@dataclass(slots=True)
class OrderEvent(Event):
    """A concrete instruction to trade ``quantity`` units of ``symbol``.

    ``quantity`` is signed: positive = buy, negative = sell. It is produced by
    the portfolio after translating a target weight into a position delta and
    after the risk manager has vetted/clamped it.
    """

    timestamp: datetime
    symbol: str
    quantity: float
    order_type: OrderType = OrderType.MARKET

    def __post_init__(self) -> None:
        self.type = EventType.ORDER

    @property
    def direction(self) -> Direction:
        if self.quantity > 0:
            return Direction.LONG
        if self.quantity < 0:
            return Direction.SHORT
        return Direction.FLAT


@dataclass(slots=True)
class FillEvent(Event):
    """The result of an order hitting the (simulated) market.

    ``fill_price`` already includes slippage. ``commission`` and ``slippage``
    are reported separately so they can be aggregated into transaction-cost
    analytics.
    """

    timestamp: datetime
    symbol: str
    quantity: float
    fill_price: float
    commission: float = 0.0
    slippage: float = 0.0

    def __post_init__(self) -> None:
        self.type = EventType.FILL

    @property
    def notional(self) -> float:
        """Signed cash value of the fill, *before* commission."""
        return self.quantity * self.fill_price
