import os
import sys
from subprocess import check_call

import pytest
import torch

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cuda is not available"
)

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.join(os.path.dirname(TESTS_DIR), "examples")

CPU_EXAMPLES = [
    "randomized_simulations.py -it 1 -infer 1 -bs 4",
    "height_simulations.py -it 1 -infer 1 -bs 4",
]

CUDA_EXAMPLES = [
    "randomized_simulations.py -it 1 -infer 1 -bs 4 --cuda",
    "height_simulations.py -it 1 -infer 1 -bs 4",
]


@pytest.mark.parametrize("example", CPU_EXAMPLES)
def test_cpu(example):
    example = example.split()
    filename, args = example[0], example[1:]
    filename = os.path.join(EXAMPLES_DIR, filename)
    check_call([sys.executable, filename] + args)


@requires_cuda
@pytest.mark.parametrize("example", CUDA_EXAMPLES)
def test_cuda(example):
    example = example.split()
    filename, args = example[0], example[1:]
    filename = os.path.join(EXAMPLES_DIR, filename)
    check_call([sys.executable, filename] + args)
