# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

from subprocess import check_call

import pytest
import torch

from tapqir.models import Cosmos
from tapqir.utils.dataset import save
from tapqir.utils.simulate import simulate

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cuda is not available"
)


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
    P = 14

    model = Cosmos()
    simulate(model, N, F, P, params=params)

    # save data
    save(model.data, tmp_path)
    return tmp_path


def test_commands_cpu(dataset_path, qtbot):

    commands = [
        ["tapqir", "config", dataset_path],
        [
            "tapqir",
            "fit",
            "multispot",
            dataset_path,
            "-it",
            "100",
            "-dev",
            "cpu",
        ],
        [
            "tapqir",
            "fit",
            "marginal",
            dataset_path,
            "-it",
            "100",
            "-dev",
            "cpu",
        ],
        [
            "tapqir",
            "fit",
            "cosmos",
            dataset_path,
            "-it",
            "100",
            "-dev",
            "cpu",
        ],
        [
            "tapqir",
            "save",
            "cosmos",
            dataset_path,
            "-dev",
            "cpu",
        ],
    ]

    for command in commands:
        check_call(command)

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
        ["tapqir", "config", dataset_path],
        [
            "tapqir",
            "fit",
            "multispot",
            dataset_path,
            "-it",
            "100",
        ],
        [
            "tapqir",
            "fit",
            "marginal",
            dataset_path,
            "-it",
            "100",
        ],
        [
            "tapqir",
            "fit",
            "cosmos",
            dataset_path,
            "-it",
            "100",
        ],
    ]

    for command in commands:
        check_call(command)
