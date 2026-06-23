"""Pairs trading -- a market-neutral statistical-arbitrage strategy.

Two historically-linked assets tend to move together. We estimate their linkage
with an ordinary least-squares hedge ratio ``beta`` (regress A on B) and define
the *spread*::

    spread = price_A - beta * price_B

When the spread is far from its recent mean (high z-score) we bet it reverts:
short the rich leg, long the cheap leg. The two legs are normalised to a fixed
gross exposure, so the book is roughly market-neutral.
"""

from __future__ import annotations

import numpy as np

from quant_engine.core.events import MarketEvent
from quant_engine.strategy.base import Strategy


class PairsTrading(Strategy):
    strategy_id = "pairs"

    def __init__(
        self,
        symbol_a: str,
        symbol_b: str,
        lookback: int = 60,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        gross: float = 1.0,
    ) -> None:
        super().__init__()
        if exit_z >= entry_z:
            raise ValueError("exit_z must be < entry_z")
        self.symbol_a = symbol_a
        self.symbol_b = symbol_b
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.gross = gross
        self.required_history = lookback
        self._state = 0  # +1 = long spread, -1 = short spread, 0 = flat

    def on_market(self, event: MarketEvent) -> None:
        a = self.closes(self.symbol_a, self.lookback)
        b = self.closes(self.symbol_b, self.lookback)
        if len(a) < self.lookback or len(b) < self.lookback:
            return

        var_b = np.var(b, ddof=1)
        if var_b == 0:
            return
        beta = float(np.cov(a, b, ddof=1)[0, 1] / var_b)
        if beta <= 0:
            return  # not a sensible long/short hedge; stay out

        spread = a - beta * b
        std = spread.std(ddof=1)
        if std == 0:
            return
        z = (spread[-1] - spread.mean()) / std

        if self._state == 0:
            if z >= self.entry_z:
                self._state = -1  # spread rich -> short A, long B
            elif z <= -self.entry_z:
                self._state = 1  # spread cheap -> long A, short B
        elif abs(z) <= self.exit_z:
            self._state = 0

        if self._state == 0:
            weight_a = weight_b = 0.0
        else:
            raw_a = float(self._state)
            raw_b = -self._state * beta
            scale = self.gross / (abs(raw_a) + abs(raw_b))
            weight_a = raw_a * scale
            weight_b = raw_b * scale

        self.set_target(self.symbol_a, weight_a)
        self.set_target(self.symbol_b, weight_b)
