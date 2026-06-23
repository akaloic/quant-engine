from __future__ import annotations

import pytest

from quant_engine.config import BacktestConfig
from quant_engine.data.synthetic import generate_prices, make_handler
from quant_engine.engine.backtest import BacktestEngine
from quant_engine.strategy.cross_sectional import CrossSectionalMomentum
from quant_engine.strategy.mean_reversion import MeanReversion
from quant_engine.strategy.moving_average import MovingAverageCrossover
from quant_engine.strategy.pairs import PairsTrading


def _run(handler, strategy, config=None):
    return BacktestEngine(handler, strategy, config or BacktestConfig()).run()


def test_ma_crossover_trades_and_runs(handler):
    result = _run(handler, MovingAverageCrossover(fast=5, slow=20))
    assert len(result.equity_curve) == 300
    assert result.n_trades > 0


def test_ma_crossover_profits_in_strong_uptrend():
    frames = generate_prices(["UP"], n_bars=300, seed=1, mu=0.5, sigma=0.08)
    result = _run(make_handler(frames), MovingAverageCrossover(fast=5, slow=20, allow_short=False))
    assert result.metrics.total_return > 0


def test_mean_reversion_runs(handler):
    result = _run(handler, MeanReversion(lookback=20, entry_z=1.5, exit_z=0.5))
    assert len(result.equity_curve) == 300
    assert result.n_trades > 0


def test_cross_sectional_momentum_runs(handler):
    result = _run(handler, CrossSectionalMomentum(lookback=60, holding=21, long_frac=0.3))
    assert len(result.equity_curve) == 300


def test_pairs_trading_runs(pair_frames):
    result = _run(
        make_handler(pair_frames), PairsTrading("XXX", "YYY", lookback=60, entry_z=1.5, exit_z=0.5)
    )
    assert len(result.equity_curve) == 400


def test_gross_exposure_respects_limit(handler):
    config = BacktestConfig()
    config.risk.max_gross_exposure = 1.0
    result = _run(handler, MovingAverageCrossover(fast=5, slow=20), config)
    # Allow a small tolerance for intrabar price drift between rebalances.
    assert result.equity_curve["gross_exposure"].max() <= 1.25


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        MovingAverageCrossover(fast=50, slow=20)
    with pytest.raises(ValueError):
        MeanReversion(entry_z=1.0, exit_z=2.0)
