.PHONY: help install install-dev backend frontend-install frontend test lint typecheck validate test-dialectic demo clean

help:
	@echo "legal research generator — make targets"
	@echo "  install         Install backend (runtime deps) in the current venv"
	@echo "  install-dev     Install backend with dev + test deps"
	@echo "  backend         Run the FastAPI backend (uvicorn, reload)"
	@echo "  frontend-install Install frontend node deps"
	@echo "  frontend        Run the Next.js dev server"
	@echo "  test            Run the pytest suite (mock backend, no GPU/network)"
	@echo "  lint            Ruff lint"
	@echo "  typecheck       mypy strict typecheck"
	@echo "  validate        Lint, typecheck dialectic module, and run dialectic tests"
	@echo "  typecheck-dialectic  mypy strict typecheck of modules/dialectic"
	@echo "  test-dialectic  Run dialectic module tests only"
	@echo "  demo            Run the end-to-end CLI demo (raw idea -> verified paper + PDF)"

install:
	python -m pip install -e .

install-dev:
	python -m pip install -e ".[dev]"

backend:
	python -m uvicorn legal_research.api.app:app --host $${LRG_HOST:-127.0.0.1} --port $${LRG_PORT:-8000} --reload

frontend-install:
	cd frontend && npm install

frontend:
	cd frontend && npm run dev

test:
	python -m pytest

lint:
	python -m ruff check src tests modules/dialectic evals probe.py

typecheck:
	python -m mypy src modules/dialectic

validate: lint typecheck-dialectic test-dialectic test-eval-harness

typecheck-dialectic:
	python -m mypy modules/dialectic evals

test-dialectic:
	python -m pytest tests/test_dialectic.py -v

test-eval-harness:
	python -m pytest tests/test_eval_harness.py -q

eval:
	python evals/run_eval.py

demo:
	python -m legal_research.demo

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache paper
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
