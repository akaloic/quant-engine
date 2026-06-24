"""Command-line interface: ``quant-engine <command>``.

Commands
--------
* ``gen-data``  generate deterministic synthetic data into a Parquet store
* ``pipeline``  run the medallion data pipeline (ingest -> validate -> curate)
* ``backtest``  run a strategy (from flags or a YAML config) and write a tearsheet
* ``train``     walk-forward train the XGBoost signal (optional ml extra)
* ``serve``     launch the REST API (optional service extra)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from quant_engine import __version__
from quant_engine.config import (
    BacktestConfig,
    DataConfig,
    ExecutionConfig,
    PipelineConfig,
    RiskConfig,
    RunConfig,
    StrategyConfig,
)
from quant_engine.strategy.registry import available_strategies


def _coerce(value: str) -> Any:
    """Best-effort string -> bool/int/float/str coercion for --param values."""
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def _parse_params(items: list[str] | None) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--param expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        params[key.strip()] = _coerce(raw.strip())
    return params


def _run_config_from_args(args: argparse.Namespace) -> RunConfig:
    if args.config:
        from quant_engine.runner import load_run_config

        return load_run_config(args.config)

    data = DataConfig(
        source=args.source,
        symbols=[s.strip() for s in args.symbols.split(",") if s.strip()],
        data_dir=args.data_dir,
        start=args.start,
        end=args.end,
        bars=args.bars,
        seed=args.seed,
        params={"kind": args.kind},
    )
    risk = RiskConfig(
        max_weight_per_symbol=args.max_weight,
        max_gross_exposure=args.max_gross,
        stop_loss_pct=args.stop_loss,
        target_volatility=args.target_vol,
    )
    backtest = BacktestConfig(
        initial_capital=args.capital,
        periods_per_year=args.periods_per_year,
        execution=ExecutionConfig(
            commission_bps=args.commission_bps, slippage_bps=args.slippage_bps
        ),
        risk=risk,
    )
    strategy = StrategyConfig(name=args.strategy, params=_parse_params(args.param))
    return RunConfig(data=data, strategy=strategy, backtest=backtest, output_dir=args.output)


def _cmd_backtest(args: argparse.Namespace) -> int:
    from quant_engine.runner import run_backtest

    config = _run_config_from_args(args)
    result = run_backtest(config)
    print(result.summary())

    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / f"{result.strategy_id}_metrics.json"
    metrics_path.write_text(json.dumps(result.metrics.as_dict(), indent=2))
    result.equity_curve.to_csv(output / f"{result.strategy_id}_equity.csv")
    print(f"\nMetrics  -> {metrics_path}")
    if not args.no_tearsheet:
        png = result.tearsheet(output / f"{result.strategy_id}_tearsheet.png")
        print(f"Tearsheet -> {png}")
    return 0


def _cmd_gen_data(args: argparse.Namespace) -> int:
    from quant_engine.data.parquet_store import save_partitioned
    from quant_engine.runner import build_frames

    config = DataConfig(
        source="synthetic",
        symbols=[s.strip() for s in args.symbols.split(",") if s.strip()],
        bars=args.bars,
        seed=args.seed,
        params={"kind": args.kind, "correlation": args.correlation},
    )
    frames = build_frames(config)
    root = save_partitioned(frames, args.out)
    rows = sum(len(f) for f in frames.values())
    print(f"Wrote {rows} rows for {list(frames)} -> {root} (partitioned by symbol/year)")
    return 0


def _cmd_pipeline(args: argparse.Namespace) -> int:
    from quant_engine.pipeline.flow import run_pipeline

    data = DataConfig(
        source=args.source,
        symbols=[s.strip() for s in args.symbols.split(",") if s.strip()],
        data_dir=args.data_dir,
        start=args.start,
        end=args.end,
        bars=args.bars,
        seed=args.seed,
        params={"kind": args.kind, "correlation": args.correlation},
    )
    pipeline = PipelineConfig(root=args.root, max_quarantine_rate=args.max_quarantine_rate)
    result = run_pipeline(data, pipeline)
    print(result.summary())
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    import pandas as pd

    from quant_engine.ml.train import walk_forward
    from quant_engine.runner import build_frames

    config = DataConfig(
        source=args.source,
        symbols=[args.symbol],
        data_dir=args.data_dir,
        bars=args.bars,
        seed=args.seed,
    )
    frames = build_frames(config)
    if args.symbol not in frames:
        raise SystemExit(f"symbol {args.symbol!r} not found in data ({list(frames)})")
    close = pd.Series(frames[args.symbol]["close"])
    report = walk_forward(
        close, symbol=args.symbol, n_splits=args.n_splits, mlflow_experiment=args.experiment
    )
    print(
        f"[{report.symbol}] walk-forward over {report.n_splits} folds "
        f"({report.n_samples} samples)\n"
        f"  CV accuracy : {report.cv_accuracy:.3f}\n"
        f"  CV ROC-AUC  : {report.cv_roc_auc:.3f}"
    )
    if args.experiment:
        print(f"  logged to MLflow experiment {args.experiment!r}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        raise SystemExit("uvicorn is required: pip install 'quant-engine[service]'") from None
    uvicorn.run("quant_engine.service.api:app", host=args.host, port=args.port, reload=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quant-engine", description=__doc__)
    parser.add_argument("--version", action="version", version=f"quant-engine {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    bt = sub.add_parser("backtest", help="run a backtest and write a tearsheet")
    bt.add_argument("--config", help="YAML RunConfig (overrides the flags below)")
    bt.add_argument("--strategy", default="ma_crossover", help=f"one of {available_strategies()}")
    bt.add_argument("--symbols", default="AAA,BBB,CCC,DDD,EEE")
    bt.add_argument("--source", default="synthetic", choices=["synthetic", "parquet", "yfinance"])
    bt.add_argument("--kind", default="gbm", choices=["gbm", "cointegrated"])
    bt.add_argument("--bars", type=int, default=756)
    bt.add_argument("--seed", type=int, default=7)
    bt.add_argument("--data-dir", default="data")
    bt.add_argument("--start", default=None)
    bt.add_argument("--end", default=None)
    bt.add_argument("--capital", type=float, default=100_000.0)
    bt.add_argument("--periods-per-year", type=int, default=252)
    bt.add_argument("--commission-bps", type=float, default=1.0)
    bt.add_argument("--slippage-bps", type=float, default=2.0)
    bt.add_argument("--max-weight", type=float, default=1.0)
    bt.add_argument("--max-gross", type=float, default=1.0)
    bt.add_argument("--stop-loss", type=float, default=None)
    bt.add_argument("--target-vol", type=float, default=None)
    bt.add_argument("--param", action="append", help="strategy param key=value (repeatable)")
    bt.add_argument("--output", default="artifacts")
    bt.add_argument("--no-tearsheet", action="store_true")
    bt.set_defaults(func=_cmd_backtest)

    gd = sub.add_parser("gen-data", help="generate synthetic data into a Parquet store")
    gd.add_argument("--symbols", default="AAA,BBB,CCC,DDD,EEE")
    gd.add_argument("--bars", type=int, default=756)
    gd.add_argument("--seed", type=int, default=7)
    gd.add_argument("--kind", default="gbm", choices=["gbm", "cointegrated"])
    gd.add_argument("--correlation", type=float, default=0.0)
    gd.add_argument("--out", default="data")
    gd.set_defaults(func=_cmd_gen_data)

    pl = sub.add_parser(
        "pipeline", help="run the medallion data pipeline (ingest -> validate -> curate)"
    )
    pl.add_argument("--source", default="synthetic", choices=["synthetic", "parquet", "yfinance"])
    pl.add_argument("--symbols", default="AAA,BBB,CCC,DDD,EEE")
    pl.add_argument("--kind", default="gbm", choices=["gbm", "cointegrated"])
    pl.add_argument("--correlation", type=float, default=0.0)
    pl.add_argument("--bars", type=int, default=756)
    pl.add_argument("--seed", type=int, default=7)
    pl.add_argument("--data-dir", default="data", help="source dir when --source parquet")
    pl.add_argument("--start", default=None)
    pl.add_argument("--end", default=None)
    pl.add_argument("--root", default="data/lake", help="root of the medallion data lake")
    pl.add_argument("--max-quarantine-rate", type=float, default=0.02)
    pl.set_defaults(func=_cmd_pipeline)

    tr = sub.add_parser("train", help="walk-forward train the XGBoost signal")
    tr.add_argument("--symbol", default="AAA")
    tr.add_argument("--source", default="synthetic", choices=["synthetic", "parquet", "yfinance"])
    tr.add_argument("--bars", type=int, default=1260)
    tr.add_argument("--seed", type=int, default=7)
    tr.add_argument("--data-dir", default="data")
    tr.add_argument("--n-splits", type=int, default=5)
    tr.add_argument("--experiment", default=None, help="MLflow experiment name (enables logging)")
    tr.set_defaults(func=_cmd_train)

    sv = sub.add_parser("serve", help="launch the REST API")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(func=_cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
