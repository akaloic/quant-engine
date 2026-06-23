"""Risk statistics computed from a return series.

Value-at-Risk (VaR) answers "how bad is a bad day?": the loss that is only
exceeded ``1 - level`` of the time. Conditional VaR / Expected Shortfall (CVaR)
answers "and when it's that bad, how bad on average?". Both are reported as
positive numbers representing losses. We use the **historical** (empirical
percentile) method -- no distributional assumption, easy to explain.
"""

from __future__ import annotations

import numpy as np


def historical_var(returns: np.ndarray, level: float = 0.95) -> float:
    """Historical VaR at confidence ``level`` (e.g. 0.95), as a positive loss."""
    if returns.size == 0:
        return 0.0
    return float(-np.quantile(returns, 1.0 - level))


def historical_cvar(returns: np.ndarray, level: float = 0.95) -> float:
    """Historical CVaR / Expected Shortfall: mean loss beyond the VaR threshold."""
    if returns.size == 0:
        return 0.0
    threshold = np.quantile(returns, 1.0 - level)
    tail = returns[returns <= threshold]
    return float(-tail.mean()) if tail.size else float(-threshold)


def realized_volatility(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Annualised standard deviation of returns."""
    if returns.size < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))
