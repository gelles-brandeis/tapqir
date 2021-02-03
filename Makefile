.PHONY: install docs lint format FORCE

install:
	pip install -e .[dev]

lint: FORCE
	flake8
	black --check .

format: FORCE
	black .

test: lint FORCE
	pytest -v -n auto

FORCE:
