"""Minimal programmatic example: build data, run a backtest, print + plot.

    python examples/run_backtest.py
"""

from __future__ import annotations

from quant_engine.config import BacktestConfig, ExecutionConfig, RiskConfig
from quant_engine.data.synthetic import generate_prices, make_handler
from quant_engine.engine.backtest import BacktestEngine
from quant_engine.strategy.moving_average import MovingAverageCrossover


def main() -> None:
    data = make_handler(generate_prices(["AAA", "BBB", "CCC", "DDD", "EEE"], n_bars=1260, seed=7))
    strategy = MovingAverageCrossover(fast=20, slow=50)
    config = BacktestConfig(
        initial_capital=100_000.0,
        execution=ExecutionConfig(commission_bps=1.0, slippage_bps=2.0),
        risk=RiskConfig(max_gross_exposure=1.0),
    )

    result = BacktestEngine(data, strategy, config).run()
    print(result.summary())
    result.tearsheet("artifacts/example_tearsheet.png")
    print("\nTearsheet written to artifacts/example_tearsheet.png")


if __name__ == "__main__":
    main()
