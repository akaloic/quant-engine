"""Data contracts: the quality rules every bar must satisfy.

A *data contract* is just an explicit, enforced agreement about what valid data
looks like. Here it is a handful of vectorised checks over an OHLCV frame. Rows
that violate any rule are not silently dropped -- they are routed to a
``quarantine`` area with a ``reason`` column, so a human can audit *why* data was
rejected. The pipeline then refuses to publish if too large a fraction failed
(see :class:`DataContractError`), which stops a bad upstream feed from quietly
poisoning every downstream backtest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel

from quant_engine.data.synthetic import OHLCV_COLUMNS

# The rule names, in a fixed order so reports are stable across runs.
CONTRACT_RULES: list[str] = [
    "null_values",  # any OHLCV field is NaN
    "non_positive_price",  # a price <= 0 is never legitimate
    "negative_volume",  # volume cannot be negative
    "ohlc_inconsistent",  # high/low must bracket open/close
    "duplicate_timestamp",  # the same bar delivered twice
]


class DataContractError(RuntimeError):
    """Raised when the share of rows failing validation exceeds the limit."""


class ValidationReport(BaseModel):
    """Aggregate outcome of validating a batch, across all symbols."""

    n_input: int
    n_valid: int
    n_quarantined: int
    gaps: int
    failures_by_rule: dict[str, int]

    @property
    def quarantine_rate(self) -> float:
        return self.n_quarantined / self.n_input if self.n_input else 0.0


def _failure_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Boolean array per rule (``True`` == the row violates that rule).

    Everything is numpy so boolean indexing stays positional and survives a
    duplicated timestamp index. Comparisons against NaN yield ``False`` (the
    null check catches those rows instead), so the warning is silenced.
    """
    prices = frame[["open", "high", "low", "close"]]
    hi = frame["high"].to_numpy()
    lo = frame["low"].to_numpy()
    op = frame["open"].to_numpy()
    cl = frame["close"].to_numpy()
    with np.errstate(invalid="ignore"):
        ohlc_bad = (hi < lo) | (hi < op) | (hi < cl) | (lo > op) | (lo > cl)
    return {
        "null_values": frame[OHLCV_COLUMNS].isna().any(axis=1).to_numpy(),
        "non_positive_price": (prices <= 0).any(axis=1).to_numpy(),
        "negative_volume": (frame["volume"] < 0).to_numpy(),
        "ohlc_inconsistent": np.asarray(ohlc_bad, dtype=bool),
        "duplicate_timestamp": frame.index.duplicated(keep="first"),
    }


def validate_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Split one symbol's OHLCV frame into (valid, quarantined, failure counts).

    The quarantined frame carries a ``reason`` column listing every rule it
    broke. A row can break several rules; each is counted independently.
    """
    masks = _failure_masks(frame)
    mask_df = pd.DataFrame(masks, index=frame.index)
    fail_any = mask_df.any(axis=1).to_numpy()

    valid = frame[~fail_any].copy()
    quarantine = frame[fail_any].copy()
    if not quarantine.empty:
        broken = mask_df[fail_any]
        reasons = broken.apply(
            lambda row: ",".join(rule for rule in CONTRACT_RULES if bool(row[rule])), axis=1
        )
        quarantine["reason"] = reasons.to_numpy()

    counts = {rule: int(mask_df[rule].sum()) for rule in CONTRACT_RULES}
    return valid, quarantine, counts


def detect_gaps(frame: pd.DataFrame) -> int:
    """Count business days missing between the first and last bar.

    A clean daily series has no gaps; a hole usually means a dropped delivery or
    a half-ingested batch. Reported (not quarantined) since a *missing* row has
    nothing to route.
    """
    if len(frame) < 2:
        return 0
    index = pd.DatetimeIndex(frame.index).normalize().unique().sort_values()
    expected = pd.bdate_range(index.min(), index.max())
    return len(expected.difference(index))
