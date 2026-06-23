"""Render a one-page performance tearsheet (PNG) from a backtest.

Uses the non-interactive Agg backend so it works headless (CI, servers, Docker).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quant_engine.analytics.performance import PerformanceMetrics


def save_tearsheet(
    equity_frame: pd.DataFrame,
    metrics: PerformanceMetrics,
    path: str | Path,
    title: str = "Backtest",
    periods_per_year: int = 252,
) -> Path:
    """Write a 4-panel tearsheet (equity, drawdown, rolling Sharpe, returns) to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    equity = equity_frame["equity"].astype(float)
    returns = equity.pct_change().dropna()
    drawdown = equity / equity.cummax() - 1.0
    window = max(min(63, len(returns) // 4), 5)
    rolling_sharpe = (
        returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(periods_per_year)
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        f"{title}   |   CAGR {metrics.cagr:.1%}   Sharpe {metrics.sharpe:.2f}   "
        f"MaxDD {metrics.max_drawdown:.1%}   Vol {metrics.annual_volatility:.1%}",
        fontsize=13,
        fontweight="bold",
    )

    ax = axes[0, 0]
    ax.plot(equity.index, equity.to_numpy(), color="#1f77b4", lw=1.4)
    ax.set_title("Equity curve")
    ax.set_ylabel("Portfolio value")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.fill_between(drawdown.index, drawdown.to_numpy() * 100, 0, color="#d62728", alpha=0.4)
    ax.set_title("Drawdown")
    ax.set_ylabel("%")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(rolling_sharpe.index, rolling_sharpe.to_numpy(), color="#2ca02c", lw=1.2)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_title(f"Rolling Sharpe ({window} periods)")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.hist(returns.to_numpy() * 100, bins=50, color="#9467bd", alpha=0.8)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_title("Distribution of period returns")
    ax.set_xlabel("%")
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def save_comparison(
    curves: dict[str, pd.Series],
    path: str | Path,
    title: str = "Strategy comparison",
) -> Path:
    """Overlay several normalised equity curves on one chart."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    for label, equity in curves.items():
        normalised = equity / equity.iloc[0]
        ax.plot(normalised.index, normalised.to_numpy(), lw=1.4, label=label)
    ax.set_title(title)
    ax.set_ylabel("Growth of 1 unit")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
