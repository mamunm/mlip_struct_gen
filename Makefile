.PHONY: help install install-dev format lint type-check test clean pre-commit venv sync

help:  ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

venv:  ## Create a virtual environment with uv
	uv venv

install:  ## Install the package in development mode with uv
	uv pip install -e .

install-dev:  ## Install the package with development dependencies using uv
	uv pip install -e ".[dev]"

sync:  ## Sync dependencies with uv
	uv pip sync requirements.txt

format:  ## Format code with black and isort
	uv run black mlip_struct_gen/
	uv run isort mlip_struct_gen/ --profile black --line-length 100

lint:  ## Run ruff linter
	uv run ruff check mlip_struct_gen/ --fix

type-check:  ## Run mypy type checker
	uv run mypy mlip_struct_gen/

test:  ## Run tests (if available)
	@echo "Tests not yet implemented"

clean:  ## Clean up build artifacts and cache files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete

pre-commit:  ## Install and run pre-commit hooks
	uv run pre-commit install
	uv run pre-commit run --all-files

quality:  ## Run all code quality checks
	@echo "Running black..."
	uv run black --check mlip_struct_gen/
	@echo "Running ruff..."
	uv run ruff check mlip_struct_gen/
	@echo "Running mypy..."
	uv run mypy mlip_struct_gen/
	@echo "All quality checks passed!"

fix:  ## Fix all auto-fixable issues
	uv run black mlip_struct_gen/
	uv run isort mlip_struct_gen/ --profile black --line-length 100
	uv run ruff check mlip_struct_gen/ --fix