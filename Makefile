.PHONY: install dev lint format type test test-all ingest reingest eval plot api ui clean

# Both install exactly what uv.lock pins into .venv (created if missing).
install:
	uv sync --frozen

dev:
	uv sync --frozen --extra dev --extra eval
	uv run --no-sync pre-commit install

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

type:
	mypy src

test:
	pytest -m "not integration"

test-all:
	pytest

ingest:
	python scripts/ingest.py

# Rebuild full text, chunks and vectors for the papers already stored.
reingest:
	python scripts/ingest.py --in-place

eval:
	python scripts/run_eval.py

plot:
	python scripts/plot_eval.py

api:
	uvicorn medrag.api.main:app --reload --port 8000

ui:
	streamlit run src/medrag/ui/app.py

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
