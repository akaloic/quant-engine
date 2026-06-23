from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from quant_engine.service.api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_strategies_listed():
    response = client.get("/strategies")
    assert "ma_crossover" in response.json()["strategies"]


def test_backtest_endpoint():
    payload = {
        "data": {"source": "synthetic", "symbols": ["AAA", "BBB"], "bars": 200, "seed": 1},
        "strategy": {"name": "ma_crossover", "params": {"fast": 5, "slow": 20}},
    }
    response = client.post("/backtest", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["strategy_id"] == "ma_crossover"
    assert len(body["equity_curve"]) == 200
    assert "sharpe" in body["metrics"]
