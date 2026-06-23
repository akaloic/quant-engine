"""XGBoost signal strategy (optional ``ml`` extra).

For each symbol the strategy trains one gradient-boosted classifier **once**, on
the first ``train_size`` bars (the in-sample window), then trades purely
out-of-sample: at every later bar it predicts the probability that the next bar
closes up and takes a long/short/flat position accordingly.

Because the model is frozen after the in-sample fit and predictions only ever use
data up to the current bar, there is no look-ahead leakage into the traded period.

Requires ``pip install 'quant-engine[ml]'`` (xgboost).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from quant_engine.core.events import MarketEvent
from quant_engine.ml.features import FEATURE_NAMES, build_feature_frame, build_training_set
from quant_engine.strategy.base import Strategy


class MLSignalStrategy(Strategy):
    strategy_id = "ml_signal"

    def __init__(
        self,
        train_size: int = 252,
        prob_threshold: float = 0.05,
        allow_short: bool = True,
        predict_window: int = 128,
        model_params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.train_size = train_size
        self.prob_threshold = prob_threshold
        self.allow_short = allow_short
        self.predict_window = max(predict_window, 70)  # enough history for all features
        self.model_params = model_params or {
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
        }
        self.required_history = train_size
        self._models: dict[str, Any] = {}
        self._trained: set[str] = set()

    def _fit(self, symbol: str) -> None:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover - only without the ml extra
            raise ImportError(
                "xgboost is required for MLSignalStrategy. "
                "Install it with: pip install 'quant-engine[ml]'"
            ) from exc

        closes = pd.Series(self.closes(symbol, self.train_size))
        x, y = build_training_set(closes)
        self._trained.add(symbol)
        if len(x) < 30 or y.nunique() < 2:
            self._models[symbol] = None  # not enough signal; this symbol stays flat
            return
        model = XGBClassifier(**self.model_params)
        model.fit(x, y)
        self._models[symbol] = model

    def on_market(self, event: MarketEvent) -> None:
        assert self.data is not None
        n = len(self.symbols)
        weight = 1.0 / n
        for symbol in self.symbols:
            if symbol not in self._trained:
                if self.data.has_history(symbol, self.train_size):
                    self._fit(symbol)
                continue  # never trade on the bar we trained on

            model = self._models.get(symbol)
            if model is None:
                continue

            closes = pd.Series(self.closes(symbol, self.predict_window))
            features = build_feature_frame(closes).iloc[[-1]]
            if features[FEATURE_NAMES].isna().to_numpy().any():
                continue
            prob_up = float(model.predict_proba(features[FEATURE_NAMES])[0, 1])

            if prob_up > 0.5 + self.prob_threshold:
                self.set_target(symbol, weight)
            elif prob_up < 0.5 - self.prob_threshold:
                self.set_target(symbol, -weight if self.allow_short else 0.0)
            else:
                self.set_target(symbol, 0.0)
