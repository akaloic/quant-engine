from __future__ import annotations

from quant_engine.config import BacktestConfig, ExecutionConfig
from quant_engine.core.events import MarketEvent
from quant_engine.data.synthetic import generate_prices, make_handler
from quant_engine.engine.backtest import BacktestEngine
from quant_engine.strategy.base import Strategy
from quant_engine.strategy.moving_average import MovingAverageCrossover


class _NoopStrategy(Strategy):
    strategy_id = "noop"

    def on_market(self, event: MarketEvent) -> None:  # never trades
        return None


def _handler(seed=11, n=250):
    return make_handler(generate_prices(["AAA", "BBB"], n_bars=n, seed=seed))


def test_flat_strategy_conserves_capital():
    result = BacktestEngine(_handler(), _NoopStrategy(), BacktestConfig()).run()
    assert result.n_trades == 0
    assert (result.equity_curve["equity"] == 100_000.0).all()


def test_equity_curve_has_one_point_per_bar():
    result = BacktestEngine(_handler(n=250), MovingAverageCrossover(5, 20), BacktestConfig()).run()
    assert len(result.equity_curve) == 250


def test_backtest_is_deterministic():
    cfg = BacktestConfig()
    first = BacktestEngine(_handler(seed=3), MovingAverageCrossover(5, 20), cfg).run()
    second = BacktestEngine(_handler(seed=3), MovingAverageCrossover(5, 20), cfg).run()
    assert first.equity_curve["equity"].iloc[-1] == second.equity_curve["equity"].iloc[-1]


def test_transaction_costs_reduce_returns():
    free = BacktestConfig(execution=ExecutionConfig(commission_bps=0, slippage_bps=0))
    costly = BacktestConfig(execution=ExecutionConfig(commission_bps=20, slippage_bps=30))
    free_result = BacktestEngine(_handler(seed=7), MovingAverageCrossover(5, 20), free).run()
    costly_result = BacktestEngine(_handler(seed=7), MovingAverageCrossover(5, 20), costly).run()
    assert costly_result.metrics.total_return < free_result.metrics.total_return
    assert costly_result.total_costs > 0


def test_result_summary_is_printable():
    result = BacktestEngine(_handler(), MovingAverageCrossover(5, 20), BacktestConfig()).run()
    text = result.summary()
    assert "Sharpe" in text and "Max drawdown" in text
