"""Trading strategies and the name->class registry."""

from quant_engine.strategy.base import Strategy
from quant_engine.strategy.cross_sectional import CrossSectionalMomentum
from quant_engine.strategy.mean_reversion import MeanReversion
from quant_engine.strategy.moving_average import MovingAverageCrossover
from quant_engine.strategy.pairs import PairsTrading
from quant_engine.strategy.registry import available_strategies, create_strategy

__all__ = [
    "CrossSectionalMomentum",
    "MeanReversion",
    "MovingAverageCrossover",
    "PairsTrading",
    "Strategy",
    "available_strategies",
    "create_strategy",
]
