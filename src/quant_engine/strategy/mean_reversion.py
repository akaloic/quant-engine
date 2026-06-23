"""Mean reversion via a rolling z-score (Bollinger-band logic).

We standardise the latest price against its recent mean and standard deviation::

    z = (price - rolling_mean) / rolling_std

A very negative z means the price is unusually cheap, so we buy; a very positive
z means it is unusually rich, so we sell/short. We close the position once the
price reverts back near its mean (``|z| <= exit_z``).
"""

from __future__ import annotations

from quant_engine.core.events import MarketEvent
from quant_engine.strategy.base import Strategy


class MeanReversion(Strategy):
    strategy_id = "mean_reversion"

    def __init__(
        self,
        lookback: int = 20,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        allow_short: bool = True,
    ) -> None:
        super().__init__()
        if exit_z >= entry_z:
            raise ValueError("exit_z must be < entry_z")
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.allow_short = allow_short
        self.required_history = lookback

    def on_market(self, event: MarketEvent) -> None:
        n = len(self.symbols)
        weight = 1.0 / n
        for symbol in self.symbols:
            closes = self.closes(symbol, self.lookback)
            if len(closes) < self.lookback:
                continue
            std = closes.std(ddof=1)
            if std == 0:
                continue
            z = (closes[-1] - closes.mean()) / std
            current = self._targets.get(symbol, 0.0)
            if z <= -self.entry_z:
                target = weight
            elif z >= self.entry_z:
                target = -weight if self.allow_short else 0.0
            elif abs(z) <= self.exit_z:
                target = 0.0
            else:
                target = current  # inside the band: hold whatever we have
            self.set_target(symbol, target)
