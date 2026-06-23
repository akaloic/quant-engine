"""Performance & risk metrics computed from an equity curve.

Every number a portfolio manager or risk officer would ask for, defined the
standard way:

* **Sharpe**  -- excess return per unit of total volatility.
* **Sortino** -- like Sharpe but only penalises *downside* volatility.
* **Calmar**  -- annual return divided by the worst peak-to-trough loss.
* **Max drawdown** -- the deepest equity decline, and how long it lasted.
* **VaR / CVaR (95%)** -- tail-loss measures (see :mod:`quant_engine.risk.metrics`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from quant_engine.risk.metrics import historical_cvar, historical_var


@dataclass
class PerformanceMetrics:
    total_return: float
    cagr: float
    annual_volatility: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    max_drawdown_duration: int
    var_95: float
    cvar_95: float
    hit_rate: float
    best_period: float
    worst_period: float
    avg_gross_exposure: float
    n_periods: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def summary(self) -> str:
        rows = [
            ("Total return", f"{self.total_return:>9.2%}"),
            ("CAGR", f"{self.cagr:>9.2%}"),
            ("Annual volatility", f"{self.annual_volatility:>9.2%}"),
            ("Sharpe ratio", f"{self.sharpe:>9.2f}"),
            ("Sortino ratio", f"{self.sortino:>9.2f}"),
            ("Calmar ratio", f"{self.calmar:>9.2f}"),
            ("Max drawdown", f"{self.max_drawdown:>9.2%}"),
            ("Max DD duration", f"{self.max_drawdown_duration:>7d} p"),
            ("VaR 95% (1-period)", f"{self.var_95:>9.2%}"),
            ("CVaR 95% (1-period)", f"{self.cvar_95:>9.2%}"),
            ("Hit rate", f"{self.hit_rate:>9.2%}"),
            ("Avg gross exposure", f"{self.avg_gross_exposure:>9.2%}"),
            ("Periods", f"{self.n_periods:>9d}"),
        ]
        width = max(len(label) for label, _ in rows)
        return "\n".join(f"{label:<{width}} : {value}" for label, value in rows)


def _max_drawdown(equity: pd.Series) -> tuple[float, int]:
    """Return (max drawdown as a negative fraction, longest underwater run)."""
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0

    underwater = (equity < running_max).to_numpy()
    longest = current = 0
    for is_under in underwater:
        current = current + 1 if is_under else 0
        longest = max(longest, current)
    return max_dd, longest


def compute_metrics(
    equity_frame: pd.DataFrame,
    periods_per_year: int = 252,
    risk_free: float = 0.0,
) -> PerformanceMetrics:
    """Compute the full metric set from a portfolio equity frame."""
    equity = equity_frame["equity"].astype(float)
    returns = equity.pct_change().dropna()
    r = returns.to_numpy()
    n = int(r.size)
    ann = float(np.sqrt(periods_per_year))

    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    total_return = end / start - 1.0 if len(equity) > 1 and start > 0 else 0.0
    years = n / periods_per_year if n > 0 else 0.0
    cagr = (end / start) ** (1.0 / years) - 1.0 if years > 0 and start > 0 else 0.0

    std = float(r.std(ddof=1)) if n > 1 else 0.0
    annual_vol = std * ann
    excess_mean = float(r.mean()) - risk_free / periods_per_year if n > 0 else 0.0
    sharpe = excess_mean / std * ann if std > 0 else 0.0

    downside_dev = float(np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))) if n > 0 else 0.0
    sortino = excess_mean / downside_dev * ann if downside_dev > 0 else 0.0

    max_dd, dd_duration = _max_drawdown(equity)
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

    avg_gross = (
        float(equity_frame["gross_exposure"].mean())
        if "gross_exposure" in equity_frame.columns
        else 0.0
    )

    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        annual_volatility=annual_vol,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=max_dd,
        max_drawdown_duration=dd_duration,
        var_95=historical_var(r, 0.95),
        cvar_95=historical_cvar(r, 0.95),
        hit_rate=float((r > 0).mean()) if n > 0 else 0.0,
        best_period=float(r.max()) if n > 0 else 0.0,
        worst_period=float(r.min()) if n > 0 else 0.0,
        avg_gross_exposure=avg_gross,
        n_periods=n,
    )
