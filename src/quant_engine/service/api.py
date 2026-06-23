"""REST API for launching backtests (optional ``service`` extra).

Run with ``quant-engine serve`` or ``uvicorn quant_engine.service.api:app``.
``RunConfig`` is a pydantic model, so FastAPI validates the request body for free
and exposes the schema at ``/docs``.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from quant_engine import __version__
from quant_engine.config import RunConfig
from quant_engine.runner import run_backtest
from quant_engine.strategy.registry import available_strategies

app = FastAPI(
    title="quant-engine",
    version=__version__,
    description="Event-driven backtesting & paper-trading engine for systematic strategies.",
)


class EquityPoint(BaseModel):
    timestamp: str
    equity: float


class BacktestResponse(BaseModel):
    strategy_id: str
    metrics: dict[str, float]
    n_trades: int
    total_costs: float
    annual_turnover: float
    equity_curve: list[EquityPoint]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/strategies")
def strategies() -> dict[str, list[str]]:
    return {"strategies": available_strategies()}


@app.post("/backtest", response_model=BacktestResponse)
def backtest(config: RunConfig) -> BacktestResponse:
    try:
        result = run_backtest(config)
    except Exception as exc:  # surface config/data errors as 400s
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    equity = result.equity_curve
    values = equity["equity"].to_numpy(dtype=float)
    curve = [
        EquityPoint(timestamp=str(ts), equity=float(value))
        for ts, value in zip(equity.index, values, strict=True)
    ]
    return BacktestResponse(
        strategy_id=result.strategy_id,
        metrics=result.metrics.as_dict(),
        n_trades=result.n_trades,
        total_costs=result.total_costs,
        annual_turnover=result.annual_turnover,
        equity_curve=curve,
    )
