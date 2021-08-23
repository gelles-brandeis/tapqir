.PHONY: install docs lint license format test FORCE

install: FORCE
	pip install -e .[dev]

uninstall: FORCE
	pip uninstall tapqir

docs: FORCE
	$(MAKE) -C docs html

lint: FORCE
	flake8
	black --check .
	isort --check .
	python scripts/update_headers.py --check

license: FORCE
	python scripts/update_headers.py

format: license FORCE
	black .
	isort .

test: lint FORCE
	pytest
	tapqir --version

FORCE:
