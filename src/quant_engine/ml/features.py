"""Causal technical features for the ML signal.

Every feature at row *t* is computed only from prices at *t* or earlier (via
``shift`` / ``rolling``), so feeding these into a model never leaks the future.
The label is the *next* bar's direction, which is what we want to predict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_NAMES: list[str] = [
    "r1",
    "r5",
    "r21",
    "mom_63",
    "ma_ratio_fast",
    "ma_ratio_slow",
    "vol_21",
    "z_20",
    "rsi_14",
]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(window).mean()
    loss = (-delta.clip(upper=0.0)).rolling(window).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def build_feature_frame(close: pd.Series) -> pd.DataFrame:
    """Return a DataFrame of :data:`FEATURE_NAMES` aligned to ``close``."""
    close = close.astype(float)
    r1 = close.pct_change()
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    features = pd.DataFrame(
        {
            "r1": r1,
            "r5": close.pct_change(5),
            "r21": close.pct_change(21),
            "mom_63": close.pct_change(63),
            "ma_ratio_fast": close / close.rolling(10).mean() - 1.0,
            "ma_ratio_slow": close / close.rolling(50).mean() - 1.0,
            "vol_21": r1.rolling(21).std(),
            "z_20": (close - sma20) / std20,
            "rsi_14": _rsi(close, 14),
        },
        index=close.index,
    )
    return features[FEATURE_NAMES]


def make_labels(close: pd.Series) -> pd.Series:
    """1 if the next bar closes higher than the current bar, else 0."""
    return (close.shift(-1) > close).astype("Int64").rename("up")


def build_training_set(close: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Aligned (X, y) with rows that have any NaN feature/label dropped."""
    features = build_feature_frame(close)
    labels = make_labels(close)
    frame = features.join(labels).dropna()
    return frame[FEATURE_NAMES], frame["up"].astype(int)
