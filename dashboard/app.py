"""Interactive dashboard for quant-engine (requires the 'dashboard' extra).

Run with::

    streamlit run dashboard/app.py

Pick a strategy and parameters in the sidebar, run a backtest, and inspect the
equity curve, drawdown and full metric set -- all powered by the same engine the
CLI and API use.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quant_engine.config import (
    BacktestConfig,
    DataConfig,
    ExecutionConfig,
    RiskConfig,
    RunConfig,
    StrategyConfig,
)
from quant_engine.runner import run_backtest
from quant_engine.strategy.registry import available_strategies


def _coerce(value: str) -> Any:
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def _parse_params(text: str) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, raw = line.split("=", 1)
        params[key.strip()] = _coerce(raw.strip())
    return params


st.set_page_config(page_title="quant-engine", layout="wide")
st.title("quant-engine — systematic trading backtester")
st.caption("Event-driven backtesting with realistic costs, risk limits and full analytics.")

with st.sidebar:
    st.header("Configuration")
    strategy_name = st.selectbox("Strategy", available_strategies(), index=0)
    symbols = st.text_input("Symbols (comma-separated)", "AAA,BBB,CCC,DDD,EEE")
    kind = st.selectbox("Synthetic data", ["gbm", "cointegrated"])
    bars = st.slider("Bars", min_value=120, max_value=2520, value=756, step=60)
    seed = st.number_input("Seed", value=7, step=1)
    st.subheader("Costs (bps)")
    commission = st.number_input("Commission", value=1.0, step=0.5)
    slippage = st.number_input("Slippage", value=2.0, step=0.5)
    st.subheader("Risk")
    max_gross = st.number_input("Max gross exposure", value=1.0, step=0.1)
    stop_loss = st.number_input("Stop-loss (0 = off)", value=0.0, step=0.01)
    target_vol = st.number_input("Target volatility (0 = off)", value=0.0, step=0.05)
    params_text = st.text_area("Strategy params (key=value per line)", "")
    run = st.button("Run backtest", type="primary")

if run:
    config = RunConfig(
        data=DataConfig(
            source="synthetic",
            symbols=[s.strip() for s in symbols.split(",") if s.strip()],
            bars=int(bars),
            seed=int(seed),
            params={"kind": kind},
        ),
        strategy=StrategyConfig(name=strategy_name, params=_parse_params(params_text)),
        backtest=BacktestConfig(
            execution=ExecutionConfig(commission_bps=commission, slippage_bps=slippage),
            risk=RiskConfig(
                max_gross_exposure=max_gross,
                stop_loss_pct=stop_loss or None,
                target_volatility=target_vol or None,
            ),
        ),
    )

    try:
        result = run_backtest(config)
    except Exception as exc:
        st.error(f"Backtest failed: {exc}")
        st.stop()

    metrics = result.metrics
    cols = st.columns(4)
    cols[0].metric("CAGR", f"{metrics.cagr:.1%}")
    cols[1].metric("Sharpe", f"{metrics.sharpe:.2f}")
    cols[2].metric("Max drawdown", f"{metrics.max_drawdown:.1%}")
    cols[3].metric("Annual vol", f"{metrics.annual_volatility:.1%}")

    equity = result.equity_curve["equity"]
    drawdown = equity / equity.cummax() - 1.0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=equity.index, y=equity.to_numpy(), name="Equity", line={"color": "#1f77b4"})
    )
    fig.update_layout(title="Equity curve", height=380, margin={"t": 40})
    st.plotly_chart(fig, use_container_width=True)

    dd_fig = go.Figure()
    dd_fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=(drawdown * 100).to_numpy(),
            fill="tozeroy",
            line={"color": "#d62728"},
        )
    )
    dd_fig.update_layout(title="Drawdown (%)", height=300, margin={"t": 40})
    st.plotly_chart(dd_fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Metrics")
        st.dataframe(
            pd.Series(metrics.as_dict(), name="value").to_frame(), use_container_width=True
        )
    with right:
        st.subheader("Execution")
        st.write(
            {
                "trades (fills)": result.n_trades,
                "total costs": round(result.total_costs, 2),
                "annual turnover": round(result.annual_turnover, 2),
            }
        )
else:
    st.info("Configure a run in the sidebar and click **Run backtest**.")
