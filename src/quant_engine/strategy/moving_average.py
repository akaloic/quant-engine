"""Moving-average crossover -- the canonical trend-following strategy.

When the fast moving average is above the slow one the trend is up, so we go
long; when it is below, we go short (or flat if ``allow_short`` is off). Simple,
transparent, and easy to reason about in an interview.
"""

from __future__ import annotations

from quant_engine.core.events import MarketEvent
from quant_engine.strategy.base import Strategy


class MovingAverageCrossover(Strategy):
    strategy_id = "ma_crossover"

    def __init__(self, fast: int = 20, slow: int = 50, allow_short: bool = True) -> None:
        super().__init__()
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be < slow ({slow})")
        self.fast = fast
        self.slow = slow
        self.allow_short = allow_short
        self.required_history = slow

    def on_market(self, event: MarketEvent) -> None:
        n = len(self.symbols)
        weight = 1.0 / n  # equal-weight the universe so gross exposure stays ~1
        for symbol in self.symbols:
            closes = self.closes(symbol, self.slow)
            if len(closes) < self.slow:
                continue
            fast_ma = closes[-self.fast :].mean()
            slow_ma = closes.mean()
            if fast_ma > slow_ma:
                self.set_target(symbol, weight)
            elif fast_ma < slow_ma:
                self.set_target(symbol, -weight if self.allow_short else 0.0)
