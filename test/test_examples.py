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
    "cosmos_simulations.py \
            --gain 7 --probsz 0.15 --ratej 0.15 --proximity 0.2 \
            --height 3000 -N 2 -F 5 -it 1 -infer 1",
    "cosmos_simulations.py -N 2 -F 5 -it 1 -infer 1",
    "kinetic_simulations.py -N 2 -F 5 -it 1 -infer 1",
    "cosmos_simulations.py \
            --gain 7 --probsz 0.15 --ratej 0.15 --proximity 0.2 \
            --height 3000 -N 2 -F 5 -it 1 -infer 1 --funsor",
    "cosmos_simulations.py -N 2 -F 5 -it 1 -infer 1 --funsor",
    "kinetic_simulations.py -N 2 -F 5 -it 1 -infer 1 --funsor",
]

CUDA_EXAMPLES = [
    "cosmos_simulations.py \
            --gain 7 --probsz 0.15 --ratej 0.15 --proximity 0.2 \
            --height 3000 -N 2 -F 5 -it 1 -infer 1 --cuda",
    "cosmos_simulations.py -N 2 -F 5 -it 1 -infer 1 --cuda",
    "kinetic_simulations.py -N 2 -F 5 -it 1 -infer 1 --cuda",
    "cosmos_simulations.py \
            --gain 7 --probsz 0.15 --ratej 0.15 --proximity 0.2 \
            --height 3000 -N 2 -F 5 -it 1 -infer 1 --funsor --cuda",
    "cosmos_simulations.py -N 2 -F 5 -it 1 -infer 1 --funsor --cuda",
    "kinetic_simulations.py -N 2 -F 5 -it 1 -infer 1 --funsor --cuda",
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
