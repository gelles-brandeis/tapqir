import sys
from pathlib import Path
from subprocess import check_call

import pytest
import torch

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cuda is not available"
)

TESTS_DIR = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = TESTS_DIR / "examples"

CPU_EXAMPLES = [
    "height_simulations.py -N 2 -F 5 -it 1 -infer 1",
    "randomized_simulations.py -N 2 -F 5 -it 1 -infer 1",
    "ratej_simulations.py -N 2 -F 5 -it 1 -infer 1",
    "kinetic_simulations.py -N 2 -F 5 -it 1 -infer 1",
    "height_simulations.py -N 2 -F 5 -it 1 -infer 1 --funsor",
    "randomized_simulations.py -N 2 -F 5 -it 1 -infer 1 --funsor",
    "ratej_simulations.py -N 2 -F 5 -it 1 -infer 1 --funsor",
    "kinetic_simulations.py -N 2 -F 5 -it 1 -infer 1 --funsor",
]

CUDA_EXAMPLES = [
    "height_simulations.py -N 2 -F 5 -it 1 -infer 1 --cuda",
    "randomized_simulations.py -N 2 -F 5 -it 1 -infer 1 --cuda",
    "ratej_simulations.py -N 2 -F 5 -it 1 -infer 1 --cuda",
    "kinetic_simulations.py -N 2 -F 5 -it 1 -infer 1 --cuda",
    "height_simulations.py -N 2 -F 5 -it 1 -infer 1 --cuda --funsor",
    "randomized_simulations.py -N 2 -F 5 -it 1 -infer 1 --cuda --funsor",
    "ratej_simulations.py -N 2 -F 5 -it 1 -infer 1 --cuda --funsor",
    "kinetic_simulations.py -N 2 -F 5 -it 1 -infer 1 --cuda --funsor",
]


@pytest.mark.parametrize("example", CPU_EXAMPLES)
def test_cpu(example, tmp_path):
    example = example.split()
    filename, args = example[0], example[1:]
    filename = EXAMPLES_DIR / filename
    args += ["--path", tmp_path]
    check_call([sys.executable, filename] + args)


@requires_cuda
@pytest.mark.parametrize("example", CUDA_EXAMPLES)
def test_cuda(example, tmp_path):
    example = example.split()
    filename, args = example[0], example[1:]
    filename = EXAMPLES_DIR / filename
    args += ["--path", tmp_path]
    check_call([sys.executable, filename] + args)
