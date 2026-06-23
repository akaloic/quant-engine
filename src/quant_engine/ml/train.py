"""Offline training & evaluation of the XGBoost signal, with MLflow tracking.

Financial time series must be validated *forward in time*: a model is trained on
the past and tested on the future, never the reverse. ``walk_forward`` does this
with an expanding window and logs parameters + cross-validated metrics
(accuracy, ROC-AUC) to MLflow so experiments are comparable and reproducible.

Requires ``pip install 'quant-engine[ml]'``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quant_engine.ml.features import build_training_set

DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
}


@dataclass
class WalkForwardReport:
    symbol: str
    n_splits: int
    cv_accuracy: float
    cv_roc_auc: float
    fold_accuracy: list[float] = field(default_factory=list)
    fold_roc_auc: list[float] = field(default_factory=list)
    n_samples: int = 0


def _expanding_splits(
    n: int, n_splits: int, min_train_frac: float = 0.4
) -> list[tuple[int, int, int]]:
    """Expanding-window splits: each fold trains on ``[0, tr_end)`` and tests on the next block."""
    start = max(int(n * min_train_frac), 30)
    if start >= n:
        return []
    fold = max((n - start) // n_splits, 1)
    splits = []
    for i in range(n_splits):
        tr_end = start + i * fold
        test_end = n if i == n_splits - 1 else tr_end + fold
        if tr_end >= n or test_end <= tr_end:
            break
        splits.append((tr_end, tr_end, test_end))
    return splits


def walk_forward(
    close: pd.Series,
    symbol: str = "ASSET",
    n_splits: int = 5,
    model_params: dict[str, Any] | None = None,
    mlflow_experiment: str | None = None,
) -> WalkForwardReport:
    """Run an expanding-window walk-forward evaluation and (optionally) log to MLflow."""
    from sklearn.metrics import accuracy_score, roc_auc_score
    from xgboost import XGBClassifier

    params = {**DEFAULT_PARAMS, **(model_params or {})}
    x, y = build_training_set(close)
    x = x.reset_index(drop=True)
    y = y.reset_index(drop=True)

    accuracies: list[float] = []
    aucs: list[float] = []
    for tr_end, test_start, test_end in _expanding_splits(len(x), n_splits):
        y_train = y.iloc[:tr_end]
        y_test = y.iloc[test_start:test_end]
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue
        model = XGBClassifier(**params)
        model.fit(x.iloc[:tr_end], y_train)
        proba = model.predict_proba(x.iloc[test_start:test_end])[:, 1]
        accuracies.append(float(accuracy_score(y_test, (proba > 0.5).astype(int))))
        aucs.append(float(roc_auc_score(y_test, proba)))

    report = WalkForwardReport(
        symbol=symbol,
        n_splits=len(accuracies),
        cv_accuracy=float(np.mean(accuracies)) if accuracies else float("nan"),
        cv_roc_auc=float(np.mean(aucs)) if aucs else float("nan"),
        fold_accuracy=accuracies,
        fold_roc_auc=aucs,
        n_samples=len(x),
    )
    _maybe_log_mlflow(mlflow_experiment, params, report)
    return report


def _maybe_log_mlflow(
    experiment: str | None, params: dict[str, Any], report: WalkForwardReport
) -> None:
    if experiment is None:
        return
    try:
        import mlflow
    except ImportError:  # pragma: no cover - only without the ml extra
        warnings.warn("mlflow not installed; skipping experiment logging", stacklevel=2)
        return
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=f"{report.symbol}-xgb-walkforward"):
        mlflow.log_params(params)
        mlflow.log_param("symbol", report.symbol)
        mlflow.log_param("n_splits", report.n_splits)
        mlflow.log_metric("cv_accuracy", report.cv_accuracy)
        mlflow.log_metric("cv_roc_auc", report.cv_roc_auc)
        mlflow.log_metric("n_samples", report.n_samples)
