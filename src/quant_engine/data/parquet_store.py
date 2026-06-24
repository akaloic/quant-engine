"""Columnar storage for price data, partitioned by ``symbol`` and ``year``.

Parquet + Hive-style partitioning is the standard layout for analytical price
stores: column pruning and partition filtering mean a backtest over one symbol
and a few years only reads the bytes it needs, instead of scanning a monolithic
CSV. ``load_partitioned`` pushes the symbol filter down into the dataset scan.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from quant_engine.data.synthetic import OHLCV_COLUMNS


def _frames_to_long(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for symbol, frame in frames.items():
        part = frame.reset_index().rename(columns={frame.index.name or "index": "timestamp"})
        part["symbol"] = symbol
        part["year"] = pd.to_datetime(part["timestamp"]).dt.year
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def save_partitioned(frames: dict[str, pd.DataFrame], root: str | Path) -> Path:
    """Write OHLCV frames as a Parquet dataset partitioned by symbol/year."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(_frames_to_long(frames), preserve_index=False)
    pq.write_to_dataset(
        table,
        root_path=str(root),
        partition_cols=["symbol", "year"],
        existing_data_behavior="delete_matching",
    )
    return root


def load_partitioned(root: str | Path, symbols: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Read a partitioned dataset back into per-symbol OHLCV frames.

    ``symbols`` is pushed down as a partition filter, so unrequested symbols are
    never read off disk.
    """
    dataset = ds.dataset(str(root), format="parquet", partitioning="hive")
    scan_filter = ds.field("symbol").isin(symbols) if symbols else None
    table = dataset.to_table(filter=scan_filter)
    long = table.to_pandas()
    if long.empty:
        return {}

    long["timestamp"] = pd.to_datetime(long["timestamp"])
    frames: dict[str, pd.DataFrame] = {}
    for symbol, group in long.groupby("symbol", sort=True):
        frame = (
            group.set_index("timestamp")[OHLCV_COLUMNS].sort_index().astype(float)
        )
        frame.index.name = "timestamp"
        frames[str(symbol)] = frame
    return frames
