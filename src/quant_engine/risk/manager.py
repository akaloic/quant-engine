"""Risk manager: turns a strategy's *desired* weights into *allowed* weights.

Strategies express intent; the risk manager enforces the firm's guard rails
before any order is sent:

* **Per-symbol cap** -- no single name exceeds ``max_weight_per_symbol``.
* **Gross-exposure cap** -- the sum of absolute weights is scaled down to
  ``max_gross_exposure`` (1.0 = no leverage).
* **Volatility targeting** (optional) -- scale a position so its annualised
  volatility matches ``target_volatility``: calmer assets get more capital,
  wilder ones get less.

Stop-losses live on the portfolio (they need live P&L) and call back here only
for the limit checks.
"""

from __future__ import annotations

import numpy as np

from quant_engine.config import RiskConfig
from quant_engine.data.base import DataHandler
from quant_engine.risk.metrics import realized_volatility

# Cap how far vol-targeting can lever a single position, before the gross cap.
_MAX_VOL_SCALE = 3.0


class RiskManager:
    def __init__(self, config: RiskConfig, periods_per_year: int) -> None:
        self.config = config
        self.periods_per_year = periods_per_year

    def _vol_scale(self, symbol: str, data: DataHandler) -> float:
        target = self.config.target_volatility
        if target is None:
            return 1.0
        bars = data.get_latest_bars(symbol, self.config.vol_lookback + 1)
        closes = np.array([b.close for b in bars])
        if closes.size < 3:
            return 1.0
        returns = np.diff(closes) / closes[:-1]
        realised = realized_volatility(returns, self.periods_per_year)
        if realised <= 0:
            return 1.0
        return float(min(target / realised, _MAX_VOL_SCALE))

    def adjusted_weight(
        self, symbol: str, target_weights: dict[str, float], data: DataHandler
    ) -> float:
        """Apply per-symbol cap, vol-targeting and the gross-exposure cap."""
        cap = self.config.max_weight_per_symbol
        weight = float(np.clip(target_weights.get(symbol, 0.0), -cap, cap))
        weight *= self._vol_scale(symbol, data)

        # Gross-exposure cap, using the capped (pre-vol-scale) snapshot as the base.
        gross = sum(min(abs(w), cap) for w in target_weights.values())
        if gross > self.config.max_gross_exposure and gross > 0:
            weight *= self.config.max_gross_exposure / gross
        return weight
