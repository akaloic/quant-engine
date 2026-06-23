from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("xgboost")

from quant_engine.config import BacktestConfig
from quant_engine.data.synthetic import generate_prices, make_handler
from quant_engine.engine.backtest import BacktestEngine
from quant_engine.strategy.ml_signal import MLSignalStrategy


def test_ml_signal_strategy_runs_out_of_sample():
    frames = generate_prices(["AAA"], n_bars=800, seed=9)
    result = BacktestEngine(
        make_handler(frames), MLSignalStrategy(train_size=300, allow_short=False), BacktestConfig()
    ).run()
    assert len(result.equity_curve) == 800


def test_walk_forward_reports_valid_metrics():
    pytest.importorskip("sklearn")
    from quant_engine.ml.train import walk_forward

    close = pd.Series(generate_prices(["AAA"], n_bars=1000, seed=2)["AAA"]["close"])
    report = walk_forward(close, symbol="AAA", n_splits=4)
    assert report.n_splits >= 1
    assert 0.0 <= report.cv_accuracy <= 1.0
    assert 0.0 <= report.cv_roc_auc <= 1.0
