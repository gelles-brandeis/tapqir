import os
from subprocess import check_call

import pytest
import torch
from tapqir import __version__ as tapqir_version
from tapqir.commands.qtgui import MainWindow
from tapqir.models import Cosmos
from PySide2.QtCore import Qt

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cuda is not available"
)

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(TESTS_DIR, "seed0")
PARAMETERS_PATH = os.path.join(
    DATASET_PATH,
    "runs",
    "cosmos",
    tapqir_version.split("+")[0],
    "S1",
    "control",
    "lr0.005",
    "bs4",
)

CPU_TAPQIR_COMMANDS = [
    ["tapqir", "--version"],
    ["tapqir", "config", DATASET_PATH],
    [
        "tapqir",
        "fit",
        "cosmos",
        DATASET_PATH,
        "-it",
        "100",
        "-infer",
        "1",
        "-dev",
        "cpu",
    ],
]

CUDA_TAPQIR_COMMANDS = [
    ["tapqir", "config", DATASET_PATH],
    ["tapqir", "fit", "cosmos", DATASET_PATH, "-it", "100", "-infer", "1"],
]


@pytest.mark.parametrize("command", CPU_TAPQIR_COMMANDS)
def test_commands_cpu(command):
    check_call(command)


@requires_cuda
@pytest.mark.parametrize("command", CUDA_TAPQIR_COMMANDS)
def test_commands_cuda(command):
    check_call(command)


def test_qtgui(qtbot):
    model = Cosmos(1, 2)
    window = MainWindow(model, DATASET_PATH, PARAMETERS_PATH)
    qtbot.addWidget(window)
    qtbot.mouseClick(window.aoiIncr, Qt.LeftButton)
    qtbot.mouseClick(window.aoiDecr, Qt.LeftButton)
    qtbot.mouseClick(window.aoiIncrLarge, Qt.LeftButton)
    qtbot.mouseClick(window.aoiDecrLarge, Qt.LeftButton)
    qtbot.keyClicks(window.aoiNumber, "1")
    qtbot.mouseClick(window.refresh, Qt.LeftButton)
    qtbot.mouseClick(window.images, Qt.LeftButton)
