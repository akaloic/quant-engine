"""Cross-sectional momentum -- a classic multi-asset factor strategy.

Every ``holding`` bars we rank the universe by trailing return over ``lookback``
bars, go long the strongest names and short the weakest, equally weighted within
each leg. This is the "winners keep winning" effect documented since
Jegadeesh & Titman (1993) and is the bread-and-butter of equity quant desks.
"""

from __future__ import annotations

from quant_engine.core.events import MarketEvent
from quant_engine.strategy.base import Strategy


class CrossSectionalMomentum(Strategy):
    strategy_id = "xs_momentum"

    def __init__(
        self,
        lookback: int = 126,
        holding: int = 21,
        long_frac: float = 0.3,
        allow_short: bool = True,
    ) -> None:
        super().__init__()
        if not 0 < long_frac <= 0.5:
            raise ValueError("long_frac must be in (0, 0.5]")
        self.lookback = lookback
        self.holding = holding
        self.long_frac = long_frac
        self.allow_short = allow_short
        self.required_history = lookback + 1
        self._bar = 0

    def on_market(self, event: MarketEvent) -> None:
        rebalance_now = self._bar % self.holding == 0
        self._bar += 1
        if not rebalance_now:
            return

        assert self.data is not None
        eligible = [s for s in self.symbols if self.data.has_history(s, self.lookback + 1)]
        if len(eligible) < 2:
            return

        trailing_return = {}
        for symbol in eligible:
            closes = self.closes(symbol, self.lookback + 1)
            trailing_return[symbol] = closes[-1] / closes[0] - 1.0

        ranked = sorted(eligible, key=lambda s: trailing_return[s])
        k = max(1, int(len(ranked) * self.long_frac))
        longs = ranked[-k:]
        shorts = ranked[:k] if self.allow_short else []

        targets = dict.fromkeys(self.symbols, 0.0)
        if shorts:
            # Dollar-neutral: 50% gross long, 50% gross short.
            for symbol in longs:
                targets[symbol] = 0.5 / k
            for symbol in shorts:
                targets[symbol] = -0.5 / k
        else:
            for symbol in longs:
                targets[symbol] = 1.0 / k

        for symbol, weight in targets.items():
            self.set_target(symbol, weight)
