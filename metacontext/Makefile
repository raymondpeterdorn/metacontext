.PHONY: help install test lint format type-check fix clean run analyze version build
.DEFAULT_GOAL := help

# Colors for output
YELLOW := \033[33m
GREEN := \033[32m
BLUE := \033[34m
RED := \033[31m
RESET := \033[0m

help: ## Show this help message
	@echo "$(BLUE)Metacontext Development Commands$(RESET)"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(YELLOW)%-20s$(RESET) %s\n", $$1, $$2}'

install: ## Install dependencies with Poetry
	@echo "$(GREEN)Installing dependencies...$(RESET)"
	poetry install

install-dev: ## Install with development dependencies
	@echo "$(GREEN)Installing with development dependencies...$(RESET)"
	poetry install --extras dev

test: ## Run tests with pytest
	@echo "$(GREEN)Running tests...$(RESET)"
	poetry run pytest tests/ -v

test-cov: ## Run tests with coverage report
	@echo "$(GREEN)Running tests with coverage...$(RESET)"
	poetry run pytest tests/ -v --cov=src/metacontext --cov-report=html --cov-report=term

lint: ## Run ruff linter (check only)
	@echo "$(GREEN)Running ruff linter...$(RESET)"
	poetry run ruff check .

lint-fix: ## Run ruff linter with auto-fix
	@echo "$(GREEN)Running ruff linter with auto-fix...$(RESET)"
	poetry run ruff check . --fix

lint-fix-unsafe: ## Run ruff linter with unsafe auto-fix (use with caution)
	@echo "$(RED)Running ruff linter with UNSAFE auto-fix...$(RESET)"
	poetry run ruff check . --fix --unsafe-fixes

format: ## Format code with ruff
	@echo "$(GREEN)Formatting code with ruff...$(RESET)"
	poetry run ruff format .

format-check: ## Check code formatting without making changes
	@echo "$(GREEN)Checking code formatting...$(RESET)"
	poetry run ruff format --check .

type-check: ## Run mypy type checker
	@echo "$(GREEN)Running mypy type checker...$(RESET)"
	poetry run mypy .

type-install: ## Install missing type stubs for mypy
	@echo "$(GREEN)Installing missing type stubs...$(RESET)"
	poetry run mypy --install-types --non-interactive src/

fix: lint-fix format ## Fix linting issues and format code
	@echo "$(GREEN)Code fixing completed!$(RESET)"

check: lint format-check type-check ## Run all checks (lint, format, type)
	@echo "$(GREEN)All checks completed!$(RESET)"

clean: ## Clean up cache files and build artifacts
	@echo "$(GREEN)Cleaning up...$(RESET)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	rm -f .coverage

run: ## Run metacontext on current project
	@echo "$(GREEN)Running metacontext on current project...$(RESET)"
	poetry run metacontext . --verbose

analyze: ## Analyze current project structure
	@echo "$(GREEN)Analyzing current project...$(RESET)"
	poetry run python -m metacontext.cli analyze .

version: ## Show version information
	@echo "$(GREEN)Version information:$(RESET)"
	poetry run python -m metacontext.cli version

build: ## Build the package with Poetry
	@echo "$(GREEN)Building package...$(RESET)"
	poetry build

publish-test: build ## Publish to test PyPI
	@echo "$(GREEN)Publishing to test PyPI...$(RESET)"
	poetry publish --repository testpypi

publish: build ## Publish to PyPI
	@echo "$(GREEN)Publishing to PyPI...$(RESET)"
	poetry publish

shell: ## Start Poetry shell
	@echo "$(GREEN)Starting Poetry shell...$(RESET)"
	poetry shell

deps: ## Show dependency tree
	@echo "$(GREEN)Dependency tree:$(RESET)"
	poetry show --tree

deps-outdated: ## Show outdated dependencies
	@echo "$(GREEN)Checking for outdated dependencies...$(RESET)"
	poetry show --outdated

update: ## Update dependencies
	@echo "$(GREEN)Updating dependencies...$(RESET)"
	poetry update

lock: ## Update lock file
	@echo "$(GREEN)Updating lock file...$(RESET)"
	poetry lock

env-info: ## Show environment information
	@echo "$(GREEN)Environment information:$(RESET)"
	poetry env info

ci: check test ## Run CI checks locally (lint, format, type-check, test)
	@echo "$(GREEN)CI checks completed successfully!$(RESET)"

pre-commit: fix test ## Prepare for commit (fix issues and run tests)
	@echo "$(GREEN)Pre-commit checks completed!$(RESET)"

# Development workflow shortcuts
dev-setup: install-dev ## Complete development setup
	@echo "$(GREEN)Development setup completed!$(RESET)"
	@echo "$(YELLOW)Next steps:$(RESET)"
	@echo "  make shell    # Start Poetry shell"
	@echo "  make test     # Run tests"
	@echo "  make run      # Run the tool"

# Quick shortcuts
q-test: ## Quick test run
	@poetry run pytest tests/ -x -q

q-lint: ## Quick lint check
	@poetry run ruff check . --quiet

q-format: ## Quick format
	@poetry run ruff format . --quiet
