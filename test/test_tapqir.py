# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typer.testing import CliRunner

from tapqir.main import app
from tapqir.models import cosmos, crosstalk, hmm
from tapqir.utils.dataset import save
from tapqir.utils.simulate import simulate

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cuda is not available"
)

runner = CliRunner()


@pytest.fixture(params=[cosmos, crosstalk, hmm])
def dataset_path(request, tmp_path):
    params = {}
    if request.param == cosmos:
        model = request.param()
        params["pi"] = 0.15
    elif request.param == crosstalk:
        model = request.param()
        params["pi"] = 0.15
        params["alpha"] = [[1.0]]
    elif request.param == hmm:
        model = request.param(vectorized=False)
        params["kon"] = 0.2
        params["koff"] = 0.2
    params["width"] = 1.4
    params["gain"] = 7.0
    params["lamda"] = 0.15
    params["proximity"] = 0.2
    params["offset"] = 90.0
    params["height"] = 3000
    params["background"] = 150
    N = 2
    F = 5
    C = 1
    P = 14

    data = simulate(model, N, F, C, P, params=params)

    # save data
    save(data, tmp_path)
    return tmp_path


@pytest.mark.parametrize("model", ["cosmos", "crosstalk"])
def test_commands_cpu(dataset_path, model):

    commands = [
        [
            "--cd",
            dataset_path,
            "fit",
            "--model",
            model,
            "--learning-rate",
            "0.005",
            "--nbatch-size",
            "2",
            "--fbatch-size",
            "5",
            "--num-iter",
            "1",
            "--cpu",
            "--no-input",
        ],
        [
            "--cd",
            dataset_path,
            "stats",
            "--model",
            model,
            "--nbatch-size",
            "2",
            "--fbatch-size",
            "5",
            "--cpu",
            "--matlab",
            "--no-input",
        ],
    ]

    for command in commands:
        result = runner.invoke(app, command)
        assert result.exit_code == 0


@requires_cuda
@pytest.mark.parametrize("model", ["cosmos", "crosstalk", "cosmos+hmm"])
def test_commands_cuda(dataset_path, model):
    commands = [
        [
            "--cd",
            dataset_path,
            "fit",
            "--model",
            model,
            "--learning-rate",
            "0.005",
            "--cuda",
            "--nbatch-size",
            "2",
            "--fbatch-size",
            "5",
            "--num-iter",
            "1",
            "--no-input",
        ],
    ]

    for command in commands:
        result = runner.invoke(app, command)
        assert result.exit_code == 0
