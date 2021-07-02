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
	python examples/cosmos_simulations.py --gain 7 --pi 0.15 --lamda 0.15 \
		--proximity 0.2 --height 3000 -N 2 -F 5 -it 1 --path cputest1
	python examples/cosmos_simulations.py -N 2 -F 5 -it 1 --path cputest2
	python examples/kinetic_simulations.py -N 2 -F 5 -it 1 --path cputest3
	# python examples/cosmos_simulations.py --gain 7 --pi 0.15 --lamda 0.15 \
	#   --proximity 0.2 --height 3000 -N 2 -F 5 -it 1 --funsor
	# python examples/cosmos_simulations.py -N 2 -F 5 -it 1 --funsor
	# python examples/kinetic_simulations.py -N 2 -F 5 -it 1 --funsor

test-cuda: lint FORCE
	python examples/cosmos_simulations.py --gain 7 --pi 0.15 --lamda 0.15 \
		--proximity 0.2 --height 3000 -N 2 -F 5 -it 1 --cuda --path cudatest1
	python examples/cosmos_simulations.py -N 2 -F 5 -it 1 --cuda --path cudatest2
	python examples/kinetic_simulations.py -N 2 -F 5 -it 1 --cuda --path cudatest3
	# python examples/cosmos_simulations.py --gain 7 --pi 0.15 --lamda 0.15 \
	# 	--proximity 0.2 --height 3000 -N 2 -F 5 -it 1 --funsor --cuda
	# python examples/cosmos_simulations.py -N 2 -F 5 -it 1 --funsor --cuda
	# python examples/kinetic_simulations.py -N 2 -F 5 -it 1 --funsor --cuda

FORCE:
