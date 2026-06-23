.PHONY: install test lint type check backtest compare dashboard api docker clean

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

type:
	mypy

check: lint type test

backtest:
	quant-engine backtest --strategy ma_crossover --symbols AAA,BBB,CCC,DDD,EEE --bars 1260

compare:
	python examples/compare_strategies.py

dashboard:
	streamlit run dashboard/app.py

api:
	quant-engine serve

docker:
	docker compose up --build

clean:
	rm -rf artifacts data mlruns .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
