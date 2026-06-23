from __future__ import annotations

from quant_engine.config import BacktestConfig
from quant_engine.data.synthetic import generate_prices, make_handler
from quant_engine.engine.backtest import BacktestEngine
from quant_engine.strategy.moving_average import MovingAverageCrossover


def test_tearsheet_is_written(tmp_path):
    frames = generate_prices(["AAA", "BBB"], n_bars=250, seed=4)
    result = BacktestEngine(make_handler(frames), MovingAverageCrossover(5, 20), BacktestConfig()).run()
    path = result.tearsheet(tmp_path / "tearsheet.png", title="Test")
    assert path.exists()
    assert path.stat().st_size > 0
