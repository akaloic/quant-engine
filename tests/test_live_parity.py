"""The headline guarantee: the live/paper loop matches the backtest exactly."""

from __future__ import annotations

import pandas as pd

from quant_engine.config import BacktestConfig
from quant_engine.data.synthetic import generate_prices, make_handler
from quant_engine.engine.backtest import BacktestEngine
from quant_engine.engine.live import LivePaperEngine
from quant_engine.strategy.moving_average import MovingAverageCrossover


def test_live_paper_matches_backtest():
    frames = generate_prices(["AAA", "BBB", "CCC"], n_bars=250, seed=5)
    config = BacktestConfig()

    backtest = BacktestEngine(make_handler(frames), MovingAverageCrossover(5, 20), config).run()
    live = LivePaperEngine(
        make_handler(frames), MovingAverageCrossover(5, 20), config, poll_interval=0.0
    ).run_live()

    pd.testing.assert_frame_equal(backtest.equity_curve, live.equity_curve)


def test_snapshot_reports_live_state():
    frames = generate_prices(["AAA"], n_bars=120, seed=1)
    seen = []
    engine = LivePaperEngine(
        make_handler(frames),
        MovingAverageCrossover(5, 20),
        on_bar=lambda snap: seen.append(snap),
    )
    engine.run_live(max_bars=30)
    assert len(seen) == 30
    assert {"timestamp", "equity", "cash", "positions"} <= set(seen[-1])
