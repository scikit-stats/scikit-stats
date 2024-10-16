.PHONY: help prepare sync doc test
.DEFAULT_GOAL := help
SHELL:=/bin/bash

# Add help text after each target name starting with '\#\#'
help:   ## show this help
	@echo -e "Help for this makefile\n"
	@echo "Possible commands are:"
	@grep -h "##" $(MAKEFILE_LIST) | grep -v grep | sed -e 's/\(.*\):.*##\(.*\)/    \1: \2/'

prepare:  ## Install dependencies and pre-commit hook
	curl -LsSf https://astral.sh/uv/install.sh | sh
	uv python install 3.12
	uv sync --all-extras --dev
	uv run pre-commit install
	uv pip install -e .

sync:  ## Sync project deps with pyproject.toml
	uv sync --all-extras --dev

# Doc and tests

clean-doc:
	rm -rf docs/html docs/jupyter_execute

doc: clean-doc  ## Build Sphinx documentation
	uv run sphinx-build -b html docs docs/html;open docs/html/index.html

test:  ## Run tests with coverage
	uv run pytest --cov skstats --cov-report term-missing
