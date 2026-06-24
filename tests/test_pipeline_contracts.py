"""Data-contract validation: clean data passes, every violation is caught."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_engine.data.synthetic import generate_prices
from quant_engine.pipeline.contracts import detect_gaps, validate_frame


def _frame(n: int = 120) -> pd.DataFrame:
    return generate_prices(["AAA"], n_bars=n, seed=1)["AAA"]


def test_clean_frame_passes_fully() -> None:
    frame = _frame()
    valid, quarantine, counts = validate_frame(frame)
    assert len(valid) == len(frame)
    assert quarantine.empty
    assert sum(counts.values()) == 0


def test_each_violation_is_detected_and_quarantined() -> None:
    frame = _frame(60).copy()
    frame.iloc[5, frame.columns.get_loc("close")] = np.nan  # null_values
    frame.iloc[10, frame.columns.get_loc("close")] = -5.0  # non_positive_price
    frame.iloc[15, frame.columns.get_loc("volume")] = -1.0  # negative_volume
    low_at_20 = float(frame.iloc[20]["low"])
    frame.iloc[20, frame.columns.get_loc("high")] = low_at_20 - 1.0  # ohlc_inconsistent

    valid, quarantine, counts = validate_frame(frame)

    assert counts["null_values"] >= 1
    assert counts["non_positive_price"] >= 1
    assert counts["negative_volume"] == 1
    assert counts["ohlc_inconsistent"] >= 1
    assert len(quarantine) >= 4
    assert "reason" in quarantine.columns
    assert len(valid) + len(quarantine) == len(frame)


def test_duplicate_timestamp_is_quarantined() -> None:
    frame = _frame(30)
    with_dupe = pd.concat([frame, frame.iloc[[0]]])  # re-deliver the first bar
    valid, _, counts = validate_frame(with_dupe)
    assert counts["duplicate_timestamp"] == 1
    assert len(valid) == len(frame)


def test_gap_detection_counts_missing_business_days() -> None:
    frame = _frame(60)
    holed = frame.drop(frame.index[10:13])  # remove three consecutive business days
    assert detect_gaps(frame) == 0
    assert detect_gaps(holed) == 3
