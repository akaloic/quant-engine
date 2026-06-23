"""quant-engine: an event-driven backtesting & paper-trading engine.

The public surface re-exports the handful of objects most users need to wire up
and run a backtest. Everything else lives in clearly named sub-packages
(``core``, ``data``, ``strategy``, ``portfolio``, ``risk``, ``execution``,
``engine``, ``analytics``).
"""

from __future__ import annotations

__version__ = "0.1.0"

from quant_engine.config import BacktestConfig
from quant_engine.engine.backtest import BacktestEngine, BacktestResult

__all__ = ["BacktestConfig", "BacktestEngine", "BacktestResult", "__version__"]
