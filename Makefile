.PHONY: install dev lint format type test ingest eval api ui clean

install:
	pip install -e .

dev:
	pip install -e ".[dev,eval]"
	pre-commit install

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff check --fix src tests
	ruff format src tests

type:
	mypy src

test:
	pytest -m "not integration"

test-all:
	pytest

ingest:
	python scripts/ingest.py

eval:
	python scripts/run_eval.py

api:
	uvicorn medrag.api.main:app --reload --port 8000

ui:
	streamlit run src/medrag/ui/app.py

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
