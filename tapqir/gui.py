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
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from tapqir.main import DEFAULTS, fit, glimpse, main


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        tabs = QTabWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        # widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.cdUI())
        layout.addWidget(tabs)
        # widget.setLayout(layout)
        scroll.setLayout(layout)

        # tabs.addTab(self.cdUI(), "Working Directory")
        tabs.addTab(self.glimpseUI(), "Extract AOIs")
        tabs.addTab(self.fitUI(), "Fit the data")

        self.setCentralWidget(scroll)

        self.resize(1200, 1000)
        self.setWindowTitle("Tapqir")
        self.default_dir = Path(".")
        self.show()

    def cdUI(self):
        cdTab = QWidget()
        layout = QVBoxLayout()
        # cd input
        cdLayout = QHBoxLayout()
        self.cdEdit = QLineEdit()
        cdBrowse = QPushButton("Browse")
        cdBrowse.clicked.connect(partial(self.getFolder, self.cdEdit))
        cdLayout.addWidget(self.cdEdit)
        cdLayout.addWidget(cdBrowse)
        # cd button
        cdChange = QPushButton("Set working directory")
        cdChange.clicked.connect(self.cdCmd)
        # layout
        layout.addLayout(cdLayout)
        layout.addWidget(cdChange)
        layout.addStretch(0)
        cdTab.setLayout(layout)
        return cdTab

    def cdCmd(self):
        main(cd=Path(self.cdEdit.text()))
        # glimpse config
        if "dataset" in DEFAULTS:
            self.datasetName.setText(DEFAULTS["dataset"])
        if "P" in DEFAULTS:
            self.aoiSize.setValue(DEFAULTS["P"])
        if "num-channels" in DEFAULTS:
            self.numChannels.setValue(DEFAULTS["num-channels"])
        if "frame-start" in DEFAULTS:
            self.firstFrame.setValue(DEFAULTS["frame-start"])
        if "frame-end" in DEFAULTS:
            self.lastFrame.setValue(DEFAULTS["frame-end"])
        if "channels" not in DEFAULTS:
            DEFAULTS["channels"] = []
        for c in range(self.numChannels.value()):
            if len(DEFAULTS["channels"]) < c + 1:
                DEFAULTS["channels"].append({})
            if "name" in DEFAULTS["channels"][c]:
                self.name[c].setText(DEFAULTS["channels"][c]["name"])
            if "glimpse-folder" in DEFAULTS["channels"][c]:
                self.header[c].setText(DEFAULTS["channels"][c]["glimpse-folder"])
            if "ontarget-aoiinfo" in DEFAULTS["channels"][c]:
                self.ontarget[c].setText(DEFAULTS["channels"][c]["ontarget-aoiinfo"])
            if "offtarget-aoiinfo" in DEFAULTS["channels"][c]:
                self.offtarget[c].setText(DEFAULTS["channels"][c]["offtarget-aoiinfo"])
            if "driftlist" in DEFAULTS["channels"][c]:
                self.driftlist[c].setText(DEFAULTS["channels"][c]["driftlist"])
        # fit config
        self.channelNumber.clear()
        self.channelNumber.addItems([str(c) for c in range(self.numChannels.value())])
        if DEFAULTS["cuda"] is not None:
            self.useGpu.setChecked(DEFAULTS["cuda"])
        if DEFAULTS["nbatch-size"] is not None:
            self.aoiBatch.setText(str(DEFAULTS["nbatch-size"]))
        if DEFAULTS["fbatch-size"] is not None:
            self.frameBatch.setText(str(DEFAULTS["fbatch-size"]))
        if DEFAULTS["learning-rate"] is not None:
            self.learningRate.setText(str(DEFAULTS["learning-rate"]))

    def glimpseUI(self):
        glimpseTab = QWidget()
        layout = QVBoxLayout()
        formLayout = QFormLayout()
        # Dataset name
        datasetName = QLineEdit()
        formLayout.addRow("Dataset name:", datasetName)
        self.datasetName = datasetName
        # AOI image size
        aoiSize = QSpinBox()
        aoiSize.setValue(14)
        aoiSize.setMinimum(6)
        aoiSize.setMaximum(20)
        formLayout.addRow("AOI image size:", aoiSize)
        self.aoiSize = aoiSize
        # Number of channels
        numChannels = QSpinBox()
        numChannels.setValue(1)
        numChannels.setMinimum(1)
        numChannels.setMaximum(4)
        numChannels.valueChanged.connect(self.channelUI)
        formLayout.addRow("Number of color channels:", numChannels)
        self.numChannels = numChannels
        # Specify frame range?
        specifyFrame = QCheckBox()
        specifyFrame.setChecked(False)
        formLayout.addRow("Specify frame range?:", specifyFrame)
        # First frame
        firstFrame = QSpinBox()
        firstFrame.setValue(1)
        firstFrame.setEnabled(False)
        formLayout.addRow("First frame:", firstFrame)
        self.firstFrame = firstFrame
        # Last frame
        lastFrame = QSpinBox()
        lastFrame.setValue(2)
        lastFrame.setMaximum(50000)
        lastFrame.setEnabled(False)
        formLayout.addRow("Last frame:", lastFrame)
        self.lastFrame = lastFrame
        specifyFrame.toggled.connect(
            partial(self.toggleWidgets, (firstFrame, lastFrame))
        )
        # channel tabs
        channelTabs = QTabWidget()
        self.channelTabs = channelTabs
        self.name, self.header, self.ontarget, self.offtarget, self.driftlist = (
            {},
            {},
            {},
            {},
            {},
        )
        self.channelUI(numChannels.value())
        # progress bar
        progressBar = QProgressBar()
        self.progressBar = progressBar
        # extract AOIs
        extractAOIs = QPushButton("tapqir glimpse")
        extractAOIs.clicked.connect(self.glimpseCmd)
        # Layout
        layout.addLayout(formLayout)
        layout.addWidget(channelTabs)
        layout.addWidget(progressBar)
        layout.addWidget(extractAOIs)
        layout.addStretch(0)
        glimpseTab.setLayout(layout)
        return glimpseTab

    def glimpseCmd(self):
        glimpse(
            dataset=self.datasetName.text(),
            P=self.aoiSize.value(),
            num_channels=self.numChannels.value(),
            name=[Edit.text() for Edit in self.name.values()],
            glimpse_folder=[Edit.text() for Edit in self.header.values()],
            ontarget_aoiinfo=[Edit.text() for Edit in self.ontarget.values()],
            offtarget_aoiinfo=[Edit.text() for Edit in self.offtarget.values()],
            driftlist=[Edit.text() for Edit in self.driftlist.values()],
            frame_start=self.firstFrame.value(),
            frame_end=self.lastFrame.value(),
            no_input=True,
            progress_bar=self.progressBar,
        )

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
                self.name[i] = nameEdit = QLineEdit()
                formLayout.addRow("Channel name:", nameEdit)
                # header/glimpse
                headerLayout = QHBoxLayout()
                self.header[i] = headerEdit = QLineEdit()
                headerBrowse = QPushButton("Browse")
                headerBrowse.clicked.connect(partial(self.getFolder, headerEdit))
                headerLayout.addWidget(headerEdit)
                headerLayout.addWidget(headerBrowse)
                formLayout.addRow("Header/glimpse folder:", headerLayout)
                # on-target aoiinfo
                ontargetLayout = QHBoxLayout()
                self.ontarget[i] = ontargetEdit = QLineEdit()
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
                self.offtarget[i] = offtargetEdit = QLineEdit()
                offtargetBrowse = QPushButton("Browse")
                offtargetBrowse.clicked.connect(partial(self.getFile, offtargetEdit))
                offtargetLayout.addWidget(offtargetEdit)
                offtargetLayout.addWidget(offtargetBrowse)
                formLayout.addRow("Off-target control locations file:", offtargetLayout)
                addOfftarget.toggled.connect(
                    partial(self.toggleWidgets, (offtargetEdit, offtargetBrowse))
                )
                # driftlist
                driftlistLayout = QHBoxLayout()
                self.driftlist[i] = driftlistEdit = QLineEdit()
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
                del self.name[i]
                del self.header[i]
                del self.ontarget[i]
                del self.offtarget[i]
                del self.driftlist[i]
                self.channelTabs.removeTab(C)

    def fitUI(self):
        fitTab = QWidget()
        layout = QVBoxLayout()
        formLayout = QFormLayout()
        # model
        modelComboBox = QComboBox()
        modelComboBox.addItem("cosmos")
        formLayout.addRow("Tapqir model:", modelComboBox)
        self.modelComboBox = modelComboBox
        # channel numbers
        channelNumber = QComboBox()
        formLayout.addRow("Channel numbers:", channelNumber)
        self.channelNumber = channelNumber
        # device
        useGpu = QCheckBox()
        useGpu.setChecked(False)
        formLayout.addRow("Run computations on GPU?", useGpu)
        self.useGpu = useGpu
        # AOI batch size
        aoiBatch = QLineEdit()
        formLayout.addRow("AOI batch size:", aoiBatch)
        self.aoiBatch = aoiBatch
        # Frame batch size
        frameBatch = QLineEdit()
        formLayout.addRow("Frame batch size:", frameBatch)
        self.frameBatch = frameBatch
        # Learning rate
        learningRate = QLineEdit()
        formLayout.addRow("Learning rate:", learningRate)
        self.learningRate = learningRate
        # Number of iterations
        iterationNumber = QLineEdit("0")
        formLayout.addRow("Number of iterations:", iterationNumber)
        self.iterationNumber = iterationNumber
        # Save parameters in matlab format?
        saveMatlab = QCheckBox()
        saveMatlab.setChecked(False)
        formLayout.addRow("Save parameters in matlab format?", saveMatlab)
        self.saveMatlab = saveMatlab
        # progress bar
        fitBar = QProgressBar()
        self.fitBar = fitBar
        # fit the data
        fitData = QPushButton("Fit the data")
        fitData.clicked.connect(self.fitCmd)
        # layout
        layout.addLayout(formLayout)
        layout.addWidget(fitBar)
        layout.addWidget(fitData)
        layout.addStretch(0)
        fitTab.setLayout(layout)
        return fitTab

    def fitCmd(self):
        fit(
            model=self.modelComboBox.currentText(),
            channels=[int(self.channelNumber.currentText())],
            cuda=self.useGpu.isChecked(),
            nbatch_size=int(self.aoiBatch.text()),
            fbatch_size=int(self.frameBatch.text()),
            learning_rate=float(self.learningRate.text()),
            num_iter=int(self.iterationNumber.text()),
            matlab=self.saveMatlab.isChecked(),
            no_input=True,
            progress_bar=self.progressBar,
        )

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
        for widget in widgets:
            widget.setEnabled(checked)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
