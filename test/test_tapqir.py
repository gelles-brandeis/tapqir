# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

from subprocess import check_call

import pytest
import torch
from PySide2.QtCore import Qt

from tapqir import __version__ as tapqir_version
from tapqir.commands.qtgui import MainWindow
from tapqir.models import Cosmos
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
    D = 14

    model = Cosmos(1, 2)
    simulate(model, N, F, D, params=params)

    # save data
    model.data.save(tmp_path)
    model.control.save(tmp_path)
    return tmp_path


def test_commands_cpu(dataset_path, qtbot):
    commands = [
        ["tapqir", "config", dataset_path],
        [
            "tapqir",
            "fit",
            "cosmos",
            dataset_path,
            "-it",
            "100",
            "-nsamples",
            "2",
            "-dev",
            "cpu",
        ],
        #  [
        #      "tapqir",
        #      "fit",
        #      "cosmos",
        #      dataset_path,
        #      "-it",
        #      "100",
        #      "-nsamples",
        #      "2",
        #      "-dev",
        #      "cpu",
        #      "-backend",
        #      "funsor",
        #  ],
    ]

    for command in commands:
        check_call(command)

    parameters_path = (
        dataset_path
        / "runs"
        / "cosmos"
        / tapqir_version.split("+")[0]
        / "S1"
        / "control"
        / "lr0.005"
        / "bs1"
    )
    model = Cosmos(1, 2)
    window = MainWindow(model, dataset_path, parameters_path)
    qtbot.addWidget(window)
    qtbot.mouseClick(window.aoiIncr, Qt.LeftButton)
    qtbot.mouseClick(window.aoiDecr, Qt.LeftButton)
    qtbot.mouseClick(window.aoiIncrLarge, Qt.LeftButton)
    qtbot.mouseClick(window.aoiDecrLarge, Qt.LeftButton)
    qtbot.keyClicks(window.aoiNumber, "1")
    qtbot.mouseClick(window.refresh, Qt.LeftButton)
    qtbot.mouseClick(window.images, Qt.LeftButton)


@requires_cuda
def test_commands_cuda(dataset_path):
    commands = [
        ["tapqir", "config", dataset_path],
        [
            "tapqir",
            "fit",
            "cosmos",
            dataset_path,
            "-it",
            "100",
            "-nsamples",
            "2",
        ],
        #  [
        #      "tapqir",
        #      "fit",
        #      "cosmos",
        #      dataset_path,
        #      "-it",
        #      "100",
        #      "-nsamples",
        #      "2",
        #      "-backend",
        #      "funsor",
        #  ],
    ]

    for command in commands:
        check_call(command)
