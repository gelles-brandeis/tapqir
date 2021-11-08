# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typer.testing import CliRunner

from tapqir.main import app
from tapqir.models import Cosmos
from tapqir.utils.dataset import save
from tapqir.utils.simulate import simulate

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cuda is not available"
)

runner = CliRunner()


@pytest.fixture
def dataset_path(tmp_path):
    params = {}
    params["width"] = 1.4
    params["gain"] = 7.0
    params["pi"] = 0.15
    params["lamda"] = 0.15
    params["proximity"] = 0.2
    params["offset"] = 90.0
    params["height"] = 3000
    params["background"] = 150
    N = 2
    F = 5
    C = 1
    P = 14

    model = Cosmos()
    data = simulate(model, N, F, C, P, params=params)

    # save data
    save(data, tmp_path)
    return tmp_path


def test_commands_cpu(dataset_path, qtbot):

    commands = [
        ["--cd", dataset_path, "init"],
        [
            "--cd",
            dataset_path,
            "fit",
            "--model",
            "cosmos",
            "--marginal",
            "--learning-rate",
            "0.005",
            "--nbatch-size",
            "2",
            "--fbatch-size",
            "5",
            "--num-epochs",
            "1",
            "--cpu",
            "--no-input",
        ],
        [
            "--cd",
            dataset_path,
            "fit",
            "--model",
            "cosmos",
            "--learning-rate",
            "0.005",
            "--nbatch-size",
            "2",
            "--fbatch-size",
            "5",
            "--num-epochs",
            "1",
            "--cpu",
            "--no-input",
        ],
        [
            "--cd",
            dataset_path,
            "stats",
            "--model",
            "cosmos",
            "--nbatch-size",
            "2",
            "--fbatch-size",
            "5",
            "--cpu",
            "--no-input",
        ],
    ]

    for command in commands:
        result = runner.invoke(app, command)
        assert result.exit_code == 0

    #  model = Cosmos()
    #  window = MainWindow(model, dataset_path)
    #  qtbot.addWidget(window)
    #  qtbot.mouseClick(window.aoiIncr, Qt.LeftButton)
    #  qtbot.mouseClick(window.aoiDecr, Qt.LeftButton)
    #  qtbot.mouseClick(window.aoiIncrLarge, Qt.LeftButton)
    #  qtbot.mouseClick(window.aoiDecrLarge, Qt.LeftButton)
    #  qtbot.keyClicks(window.aoiNumber, "1")
    #  qtbot.mouseClick(window.refresh, Qt.LeftButton)
    #  qtbot.mouseClick(window.images, Qt.LeftButton)


@requires_cuda
def test_commands_cuda(dataset_path):
    commands = [
        ["--cd", dataset_path, "init"],
        [
            "--cd",
            dataset_path,
            "fit",
            "--model",
            "cosmos",
            "--marginal",
            "--learning-rate",
            "0.005",
            "--cuda",
            "--nbatch-size",
            "2",
            "--fbatch-size",
            "5",
            "--num-epochs",
            "1",
            "--no-input",
        ],
        [
            "--cd",
            dataset_path,
            "fit",
            "--model",
            "cosmos",
            "--learning-rate",
            "0.005",
            "--cuda",
            "--nbatch-size",
            "2",
            "--fbatch-size",
            "5",
            "--num-epochs",
            "1",
            "--no-input",
        ],
    ]

    for command in commands:
        result = runner.invoke(app, command)
        assert result.exit_code == 0
