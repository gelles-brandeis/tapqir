# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

import sys
from functools import partial
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
import torch
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from pyqtgraph import HistogramLUTItem
from PySide6.QtCore import Qt
from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from tapqir.main import DEFAULTS, glimpse, init, main


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        self.tabs = QTabWidget()

        self.tabs.addTab(self.cdUI(), "Working Directory")
        self.tabs.addTab(self.glimpseUI(), "Extract AOIs")
        self.tabs.addTab(self.fitUI(), "Fit the data")

        self.setCentralWidget(self.tabs)

        self.resize(1200, 1000)
        self.setWindowTitle("Tapqir")
        self.default_dir = Path(".")
        self.show()

    def cdUI(self):
        cdTab = QWidget()
        layout = QVBoxLayout()
        # cd input
        cdLayout = QHBoxLayout()
        cdEdit = QLineEdit()
        cdBrowse = QPushButton("Browse")
        cdBrowse.clicked.connect(partial(self.getFolder, cdEdit))
        cdLayout.addWidget(cdEdit)
        cdLayout.addWidget(cdBrowse)
        # cd button
        cdChange = QPushButton("Set working directory")
        cdChange.clicked.connect(partial(main, Path(cdEdit.text())))
        # layout
        layout.addLayout(cdLayout)
        layout.addWidget(cdChange)
        layout.addStretch(0)
        cdTab.setLayout(layout)
        return cdTab

    def glimpseUI(self):
        glimpseTab = QWidget()
        layout = QVBoxLayout()
        formLayout = QFormLayout()
        # Dataset name
        datasetName = QLineEdit()
        formLayout.addRow("Dataset name:", datasetName)
        # AOI image size
        aoiSize = QSpinBox()
        aoiSize.setValue(14)
        aoiSize.setMinimum(6)
        aoiSize.setMaximum(20)
        formLayout.addRow("AOI image size:", aoiSize)
        self.aoiSize = aoiSize
        # Number of channels
        channelNumber = QSpinBox()
        channelNumber.setValue(1)
        channelNumber.setMinimum(1)
        channelNumber.setMaximum(4)
        channelNumber.valueChanged.connect(self.channelUI)
        formLayout.addRow("Number of color channels:", channelNumber)
        # Specify frame range?
        specifyFrame = QCheckBox()
        specifyFrame.setChecked(False)
        formLayout.addRow("Specify frame range?:", specifyFrame)
        # First frame
        firstFrame = QSpinBox()
        firstFrame.setValue(1)
        firstFrame.setEnabled(False)
        formLayout.addRow("First frame:", firstFrame)
        # Last frame
        lastFrame = QSpinBox()
        lastFrame.setValue(2)
        lastFrame.setEnabled(False)
        formLayout.addRow("Last frame:", lastFrame)
        specifyFrame.toggled.connect(partial(self.toggleWidgets, (firstFrame, lastFrame)))
        # channel tabs
        channelTabs = QTabWidget()
        self.channelTabs = channelTabs
        self.channelUI(channelNumber.value())
        # extract AOIs
        extractAOIs = QPushButton("tapqir glimpse")
        extractAOIs.clicked.connect(self.glimpseCmd)
        # Layout
        layout.addLayout(formLayout)
        layout.addWidget(channelTabs)
        layout.addStretch(0)
        glimpseTab.setLayout(layout)
        return glimpseTab

    def channelUI(self, C):
        currentC = self.channelTabs.count()
        for i in range(max(currentC, C)):
            if i < C and i < currentC:
                continue
            elif i < C and i >= currentC:
                widget = QWidget()
                vbox = QVBoxLayout()
                formLayout = QFormLayout()
                # channel name
                nameEdit = QLineEdit()
                formLayout.addRow("Channel name:", nameEdit)
                # header/glimpse
                headerLayout = QHBoxLayout()
                headerEdit = QLineEdit()
                headerBrowse = QPushButton("Browse")
                headerBrowse.clicked.connect(partial(self.getFolder, headerEdit))
                headerLayout.addWidget(headerEdit)
                headerLayout.addWidget(headerBrowse)
                formLayout.addRow("Header/glimpse folder:", headerLayout)
                # on-target aoiinfo
                ontargetLayout = QHBoxLayout()
                ontargetEdit = QLineEdit()
                ontargetBrowse = QPushButton("Browse")
                ontargetBrowse.clicked.connect(partial(self.getFile, ontargetEdit))
                ontargetLayout.addWidget(ontargetEdit)
                ontargetLayout.addWidget(ontargetBrowse)
                formLayout.addRow("Target molecule locations file:", ontargetLayout)
                # Add off-target AOI locations?
                addOfftarget = QCheckBox()
                addOfftarget.setChecked(True)
                formLayout.addRow("Add off-target AOI locations?", addOfftarget)
                # off-target aoiinfo
                offtargetLayout = QHBoxLayout()
                offtargetEdit = QLineEdit()
                offtargetBrowse = QPushButton("Browse")
                offtargetBrowse.clicked.connect(partial(self.getFile, offtargetEdit))
                offtargetLayout.addWidget(offtargetEdit)
                offtargetLayout.addWidget(offtargetBrowse)
                formLayout.addRow("Off-target control locations file:", offtargetLayout)
                addOfftarget.toggled.connect(partial(self.toggleWidgets, (offtargetEdit, offtargetBrowse)))
                # driftlist
                driftlistLayout = QHBoxLayout()
                driftlistEdit = QLineEdit()
                driftlistBrowse = QPushButton("Browse")
                driftlistBrowse.clicked.connect(partial(self.getFile, driftlistEdit))
                driftlistLayout.addWidget(driftlistEdit)
                driftlistLayout.addWidget(driftlistBrowse)
                formLayout.addRow("Driftlist file:", driftlistLayout)
                # layout
                vbox.addLayout(formLayout)
                vbox.addStretch(0)
                widget.setLayout(vbox)
                self.channelTabs.addTab(widget, f"Channel #{i}")
            else:
                self.channelTabs.removeTab(C)

    def fitUI(self):
        fitTab = QWidget()
        layout = QVBoxLayout()
        formLayout = QFormLayout()
        # model
        modelComboBox = QComboBox()
        formLayout.addRow("Tapqir model:", modelComboBox)
        # channel numbers
        channelEdit = QLineEdit()
        formLayout.addRow("Channel numbers:", channelEdit)
        # device
        useGpu = QCheckBox()
        useGpu.setChecked(False)
        formLayout.addRow("Run computations on GPU?", useGpu)
        # AOI batch size
        aoiBatch = QLineEdit("10")
        formLayout.addRow("AOI batch size:", aoiBatch)
        # Frame batch size
        frameBatch = QLineEdit("512")
        formLayout.addRow("Frame batch size:", frameBatch)
        # Learning rate
        learningRate = QLineEdit("0.005")
        formLayout.addRow("Learning rate:", learningRate)
        # Number of iterations
        iterationNumber = QLineEdit("0")
        formLayout.addRow("Number of iterations:", iterationNumber)
        # Save parameters in matlab format?
        saveMatlab = QCheckBox()
        saveMatlab.setChecked(False)
        formLayout.addRow("Save parameters in matlab format?", saveMatlab)
        # layout
        layout.addLayout(formLayout)
        layout.addStretch(0)
        fitTab.setLayout(layout)
        return fitTab

    def getFolder(self, widget):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.Directory)
        fname = dlg.getExistingDirectory(
            self,
            "Select folder",
            str(self.default_dir),
        )
        widget.setText(fname)
        self.default_dir = Path(fname).parent
        return fname

    def getFile(self, widget):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.Directory)
        fname, _ = dlg.getOpenFileName(self, "Select file", str(self.default_dir))
        widget.setText(fname)
        self.default_dir = Path(fname).parent
        return fname

    def toggleWidgets(self, widgets, checked):
        if checked:
            for widget in widgets:
                widget.setEnabled(True)
        else:
            for widget in widgets:
                widget.setEnabled(False)

    def glimpseCmd(self):
        global DEFAULTS
        DEFAULTS["dataset"] = dataset
        DEFAULTS["P"] = P
        DEFAULTS["num-channels"] = num_channels
        glimpse(no_input=True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
