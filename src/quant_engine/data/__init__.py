"""Market-data layer: handlers, synthetic generation and Parquet storage."""

from quant_engine.data.base import DataHandler, HistoricDataHandler
from quant_engine.data.parquet_store import load_partitioned, save_partitioned
from quant_engine.data.synthetic import (
    generate_cointegrated_pair,
    generate_prices,
    make_handler,
)

__all__ = [
    "DataHandler",
    "HistoricDataHandler",
    "generate_cointegrated_pair",
    "generate_prices",
    "load_partitioned",
    "make_handler",
    "save_partitioned",
]
