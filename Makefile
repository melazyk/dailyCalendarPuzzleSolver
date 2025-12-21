.PHONY: test test-fast test-slow test-cov test-html clean help install

# Python interpreter
PYTHON := python3

# Default target
.DEFAULT_GOAL := help

help: ## Show help for all commands
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36mmake %-15s\033[0m %s\n", $$1, $$2}'

install: ## Install development dependencies
	$(PYTHON) -m pip install -r requirements-dev.txt

test: ## Run all tests with coverage
	$(PYTHON) -m pytest -v

test-fast: ## Run only fast tests (skip slow ones)
	$(PYTHON) -m pytest -v -m "not slow"

test-slow: ## Run only slow tests
	$(PYTHON) -m pytest -v -m "slow"

test-cov: ## Run tests with detailed coverage report
	$(PYTHON) -m pytest -v --cov-report=term-missing

test-html: ## Run tests and open HTML coverage report
	$(PYTHON) -m pytest -v
	@echo "\nOpening HTML coverage report..."
	@xdg-open htmlcov/index.html 2>/dev/null || open htmlcov/index.html 2>/dev/null || echo "Open htmlcov/index.html in your browser"

test-parallel: ## Run tests in parallel (faster)
	$(PYTHON) -m pytest -v -n auto

test-watch: ## Run tests on file changes (requires pytest-watch)
	$(PYTHON) -m pytest-watch

clean: ## Clean temporary files and cache
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf coverage.json
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "Temporary files removed"

coverage-report: ## Show coverage report from last run
	$(PYTHON) -m coverage report

coverage-json: ## Export coverage to JSON
	$(PYTHON) -m coverage json
	@echo "Report saved to coverage.json"
