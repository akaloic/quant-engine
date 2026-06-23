"""Typed configuration models.

Everything that parametrises a run lives here as ``pydantic`` models. The
engine consumes :class:`BacktestConfig`; the CLI additionally reads
:class:`RunConfig` from a YAML file so that an experiment is fully described by
a single, version-controllable document.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field


class ExecutionConfig(BaseModel):
    """Cost model for the simulated execution handler.

    Costs are expressed in basis points (1 bp = 0.01%) of traded notional,
    which is how brokers and TCA reports usually quote them.
    """

    commission_bps: float = Field(1.0, ge=0, description="Per-trade commission (bps of notional).")
    min_commission: float = Field(0.0, ge=0, description="Floor on per-trade commission, in cash.")
    slippage_bps: float = Field(2.0, ge=0, description="Fixed slippage (bps), applied adversely.")
    impact_coefficient: float = Field(
        0.0,
        ge=0,
        description="Linear impact: extra bps = coef * (order_notional / ADV).",
    )


class RiskConfig(BaseModel):
    """Portfolio-level guard rails enforced before every order."""

    max_weight_per_symbol: float = Field(
        1.0, gt=0, le=1.0, description="Cap on |weight| in any single symbol."
    )
    max_gross_exposure: float = Field(
        1.0, gt=0, description="Cap on sum of |weights| (1.0 = no leverage)."
    )
    stop_loss_pct: float | None = Field(
        None, gt=0, description="Per-position trailing stop, as a fraction (e.g. 0.1 = 10%)."
    )
    target_volatility: float | None = Field(
        None,
        gt=0,
        description="Annualised vol target; if set, weights are scaled by target/realised vol.",
    )
    vol_lookback: int = Field(20, gt=1, description="Lookback (bars) for realised-vol scaling.")


class BacktestConfig(BaseModel):
    """Everything the engine needs that is independent of data and strategy."""

    initial_capital: float = Field(100_000.0, gt=0)
    periods_per_year: int = Field(
        252, gt=0, description="Annualisation factor (252 daily, 52 weekly, 12 monthly)."
    )
    execution: ExecutionConfig = ExecutionConfig()
    risk: RiskConfig = RiskConfig()


class DataConfig(BaseModel):
    """Where price data comes from."""

    source: Literal["synthetic", "parquet", "yfinance"] = "synthetic"
    symbols: list[str] = Field(default_factory=lambda: ["AAA"])
    start: date | None = None
    end: date | None = None
    data_dir: str = "data"
    # Synthetic-only knobs.
    bars: int = Field(756, gt=1, description="Number of bars to generate (synthetic source).")
    seed: int = 7
    params: dict[str, Any] = Field(default_factory=dict)


class StrategyConfig(BaseModel):
    """Strategy selection by registry name plus its keyword arguments."""

    name: str = "ma_crossover"
    params: dict[str, Any] = Field(default_factory=dict)


class RunConfig(BaseModel):
    """Top-level CLI configuration (a whole experiment in one document)."""

    data: DataConfig = DataConfig()
    strategy: StrategyConfig = StrategyConfig()
    backtest: BacktestConfig = BacktestConfig()
    output_dir: str = "artifacts"
