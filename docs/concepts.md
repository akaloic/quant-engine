# Concepts, in plain language

This is the "explain it to me like I'm in an interview" guide. Every concept the
engine implements, why it matters, and the one-line version you'd say out loud.

## Event-driven backtesting

A backtest can be written two ways. The *vectorised* way computes signals over a
whole price history at once with array maths -- fast, but dangerously easy to peek
at the future by accident. The *event-driven* way replays history one bar at a
time and pushes typed events (`MarketEvent`, `SignalEvent`, `OrderEvent`,
`FillEvent`) through a single queue. Components react only to the event in front
of them.

> **Say it:** "I built it event-driven so the same code runs in backtest and live,
> and so a strategy structurally cannot see the future."

## Look-ahead bias

The cardinal sin of backtesting: using information that wouldn't have been
available at decision time. Here, a strategy can only read history through
`DataHandler.get_latest_bars(symbol, n)`, which returns bars *up to and including*
the current heartbeat -- never beyond. That's enforced by the data handler, not by
discipline.

> **Say it:** "The data handler only ever hands back history up to the current bar,
> so look-ahead is impossible by construction."

## Target weights vs. raw orders

Strategies output a **target weight** per symbol in `[-1, 1]` (the fraction of
equity to hold), not a number of shares. The portfolio converts weight into a
share delta given current equity and price, and the risk manager vets it. This
decouples strategy logic from account size and from sizing/risk concerns.

> **Say it:** "Strategies express intent as a target weight; sizing and risk live in
> one place downstream."

## Transaction costs: commission, slippage, market impact

Ignoring costs is the most common way a backtest lies.

- **Commission** -- the broker's fee, here a few basis points of notional.
- **Slippage** -- you don't fill exactly at the quoted price; a buy fills a little
  higher, a sell a little lower. Modelled as a fixed bps applied against you.
- **Market impact** -- big orders move the price; modelled (optionally) as extra
  bps proportional to your order size versus recent volume (participation rate).

> **Say it:** "I fill at the close like a market-on-close order, then charge
> commission, slippage and an optional size-based impact -- so the equity curve is
> net of realistic costs."

## Risk management

- **Per-symbol cap** -- never bet too much on one name.
- **Gross-exposure cap** -- the sum of absolute weights; 1.0 means no leverage.
- **Volatility targeting** -- scale positions so each contributes a similar amount
  of risk: give calm assets more capital, wild ones less. Computed from realised
  volatility over a lookback.
- **Stop-loss** -- if a position's loss from entry exceeds a threshold, flatten it.

> **Say it:** "Strategies propose, the risk manager disposes -- caps, vol-targeting
> and stops are enforced before any order is sent."

## Performance metrics

- **Sharpe** = excess return / total volatility. Risk-adjusted return.
- **Sortino** = excess return / *downside* volatility. Doesn't punish upside swings.
- **Calmar** = annual return / worst drawdown. Return per unit of pain.
- **Max drawdown** = largest peak-to-trough equity drop, with how long it lasted.
- **VaR 95%** = the loss you only exceed 5% of the time (historical percentile).
- **CVaR 95%** = the average loss *on those worst 5% of days*.
- **Turnover** = how much you trade per year; high turnover eats returns via costs.

> **Say it:** "I report the standard risk-adjusted ratios plus tail risk (VaR/CVaR)
> and turnover, all computed from the mark-to-market equity curve."

## Walk-forward validation (for the ML signal)

You can never test a time-series model on data from before its training window.
Walk-forward training expands the window forward: train on `[0, t)`, test on the
next block, roll forward, repeat. The reported accuracy/AUC is the average over
out-of-sample blocks.

A sanity check baked into the demo: on synthetic geometric-Brownian-motion data
(a pure random walk) the model scores ~0.50 AUC -- i.e. it correctly finds **no**
edge. That's the point: if a pipeline reports high accuracy on a random walk, it's
leaking the future. Ours doesn't.

> **Say it:** "I validate the model walk-forward so train always precedes test, and
> on a random walk it correctly finds zero edge -- proof there's no leakage."

## Research-to-live parity

The backtest engine and the paper-trading engine share the *same* event loop,
strategy, portfolio, risk and analytics code (`LivePaperEngine` subclasses
`BacktestEngine`). To trade for real you swap two objects -- a broker-backed data
feed and a broker-backed execution handler -- and nothing else changes.

> **Say it:** "Research and live run the same code path; going live is swapping the
> data feed and the execution handler, not rewriting the strategy."
