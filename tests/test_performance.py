from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_engine.analytics.performance import compute_metrics


def _frame(values: list[float]) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=len(values), freq="B")
    return pd.DataFrame({"equity": values}, index=index)


def test_monotonic_growth_has_no_drawdown_and_positive_sharpe():
    equity = _frame([100.0 * (1.001**i) for i in range(300)])
    metrics = compute_metrics(equity, periods_per_year=252)
    assert metrics.total_return > 0
    assert metrics.sharpe > 0
    assert metrics.max_drawdown == 0.0
    assert metrics.hit_rate == 1.0


def test_max_drawdown_matches_known_path():
    equity = _frame([100.0, 120.0, 150.0, 75.0, 90.0])
    metrics = compute_metrics(equity, periods_per_year=252)
    # Peak 150 -> trough 75 = -50%.
    assert metrics.max_drawdown == pytest.approx(-0.5)


def test_var_and_cvar_are_positive_losses():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0, 0.01, size=1000)
    equity = _frame(list(100.0 * np.cumprod(1 + returns)))
    metrics = compute_metrics(equity, periods_per_year=252)
    assert metrics.var_95 > 0
    assert metrics.cvar_95 >= metrics.var_95  # tail loss is at least the VaR
