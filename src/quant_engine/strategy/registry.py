"""Map strategy names to classes so runs can be described in YAML/CLI."""

from __future__ import annotations

from typing import Any

from quant_engine.strategy.base import Strategy
from quant_engine.strategy.cross_sectional import CrossSectionalMomentum
from quant_engine.strategy.mean_reversion import MeanReversion
from quant_engine.strategy.moving_average import MovingAverageCrossover
from quant_engine.strategy.pairs import PairsTrading

_REGISTRY: dict[str, type[Strategy]] = {
    cls.strategy_id: cls
    for cls in (MovingAverageCrossover, MeanReversion, CrossSectionalMomentum, PairsTrading)
}


def available_strategies() -> list[str]:
    """Names accepted by :func:`create_strategy` (``ml_signal`` needs the ml extra)."""
    return sorted([*_REGISTRY, "ml_signal"])


def create_strategy(name: str, params: dict[str, Any] | None = None) -> Strategy:
    """Instantiate a strategy by registry name with keyword ``params``."""
    params = params or {}
    key = name.lower()
    if key == "ml_signal":
        # Imported lazily so the optional xgboost dependency isn't required to
        # import the rest of the package.
        from quant_engine.strategy.ml_signal import MLSignalStrategy

        return MLSignalStrategy(**params)
    if key not in _REGISTRY:
        raise KeyError(f"unknown strategy {name!r}; available: {available_strategies()}")
    return _REGISTRY[key](**params)
