"""Run several strategies on the same data and overlay their equity curves.

    python examples/compare_strategies.py
"""

from __future__ import annotations

import pandas as pd

from quant_engine.analytics.tearsheet import save_comparison
from quant_engine.config import BacktestConfig
from quant_engine.data.synthetic import generate_prices, make_handler
from quant_engine.engine.backtest import BacktestEngine
from quant_engine.strategy.base import Strategy
from quant_engine.strategy.cross_sectional import CrossSectionalMomentum
from quant_engine.strategy.mean_reversion import MeanReversion
from quant_engine.strategy.moving_average import MovingAverageCrossover

SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]


def _strategies() -> dict[str, Strategy]:
    return {
        "MA crossover (20/50)": MovingAverageCrossover(fast=20, slow=50),
        "Mean reversion (z=2)": MeanReversion(lookback=20, entry_z=2.0, exit_z=0.5),
        "XS momentum (126/21)": CrossSectionalMomentum(lookback=126, holding=21, long_frac=0.3),
    }


def main() -> None:
    curves: dict[str, pd.Series] = {}
    for name, strategy in _strategies().items():
        # Fresh data handler per run (handlers are stateful, single-pass).
        data = make_handler(generate_prices(SYMBOLS, n_bars=1260, seed=7))
        result = BacktestEngine(data, strategy, BacktestConfig()).run()
        curves[name] = result.equity_curve["equity"]
        m = result.metrics
        print(f"{name:<22} Sharpe {m.sharpe:5.2f}  CAGR {m.cagr:7.2%}  MaxDD {m.max_drawdown:7.2%}")

    path = save_comparison(curves, "assets/strategy_comparison.png")
    print(f"\nComparison chart written to {path}")


if __name__ == "__main__":
    main()
