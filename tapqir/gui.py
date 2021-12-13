
import sys
from functools import partial

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
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QTabWidget,
)
from tapqir.main import DEFAULTS, main, init, glimpse

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        #  self.Model = model
        #  self.path = path
        #  self.Model.load(path, data_only=False)

        self.initUI()

    def initUI(self):
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        self.tabs = QTabWidget()
        self.cd = QWidget()
        self.glimpse = QWidget()
        self.fit = QWidget()

        self.tabs.addTab(self.cd, "Working Directory")
        self.tabs.addTab(self.glimpse, "Extract AOIs")
        self.tabs.addTab(self.fit, "Fit the data")

        self.cdUI()
        self.glimpseUI()
        #  self.scroll = QScrollArea()
        #  self.vbox = QVBoxLayout()

        #  self.widget.setLayout(self.vbox)

        # Scroll Area Properties
        #  self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        #  self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        #  self.scroll.setWidgetResizable(True)
        #  self.scroll.setWidget(self.widget)

        self.setCentralWidget(self.tabs)

        self.resize(1200, 1000)
        self.setWindowTitle("Tapqir")
        self.show()

    def cdUI(self):
        vbox = QVBoxLayout()
        # cd select
        cdSelect = QWidget()
        hbox = QHBoxLayout()
        cdEdit = QLineEdit()
        cdOpen = QPushButton("Open")
        hbox.addWidget(cdEdit)
        hbox.addWidget(cdOpen)
        cdSelect.setLayout(hbox)
        # cd slot
        cdChange = QPushButton("Change working directory")
        # spacer
        spacer = QSpacerItem(100, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        # Layout
        vbox.addWidget(cdSelect)
        vbox.addWidget(cdChange)
        vbox.addItem(spacer)
        self.cd.setLayout(vbox)

    def glimpseUI(self):
        vbox = QVBoxLayout()
        # Dataset name
        nameWidget = QWidget()
        nameLayout = QVBoxLayout()
        nameEdit = QLineEdit()
        nameLabel = QLabel("Dataset name")
        nameLayout.addWidget(nameEdit)
        nameLayout.addWidget(nameLabel)
        nameWidget.setLayout(nameLayout)
        # Number of channels
        numberWidget = QWidget()
        numberLayout = QVBoxLayout()
        numberSpin = QSpinBox()
        numberSpin.setValue(1)
        numberSpin.setMinimum(1)
        numberSpin.setMaximum(4)
        numberSpin.valueChanged.connect(self.numberSlot)
        numberLabel = QLabel("Number of color channel")
        numberLayout.addWidget(numberSpin)
        numberLayout.addWidget(numberLabel)
        numberWidget.setLayout(numberLayout)
        # spacer
        spacer = QSpacerItem(100, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        # Layout
        vbox.addWidget(nameWidget)
        vbox.addWidget(numberWidget)
        vbox.addItem(spacer)
        self.glimpse.setLayout(vbox)

    def numberSlot(self):
        f

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
