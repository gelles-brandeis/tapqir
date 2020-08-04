from PySide2.QtWidgets import (QWidget, QSlider, QLineEdit, QLabel, QPushButton, QScrollArea, QApplication,
                             QHBoxLayout, QVBoxLayout, QMainWindow, QSizePolicy, QSpacerItem)
from PySide2.QtCore import Qt, QSize
from PySide2.QtGui import QIntValidator
from PySide2 import QtWidgets
import sys
from functools import partial

import pyqtgraph as pg
import torch
import numpy as np
import pyro
from pyro import param
from pyro.ops.stats import pi
from cosmos.models.tracker import Tracker
import pyro.distributions as dist
from cosmos.ui.utils import plot_graph, \
    HistogramLUTGraph
from collections import defaultdict

C = {}
C[0] = (31, 119, 180)
C[1] = (255, 127, 14)
C[2] = (44, 160, 44)
C[3] = (214, 39, 40)

from pyqtgraph import HistogramLUTItem

class HistogramLUTGraph(HistogramLUTItem):

    def __init__(self, graph, **kwargs):
        self.graph = graph
        super().__init__(**kwargs)

    def regionChanged(self):
        for image in self.graph.values():
            image.setLevels(self.getLevels())
        self.sigLevelChangeFinished.emit(self)

    def regionChanging(self):
        pass

class AnotherWindow(QScrollArea):
    """
    This "window" is a QWidget. If it has no parent, it 
    will appear as a free-floating window as we want.
    """
    def __init__(self):
        super().__init__()
        #layout = QVBoxLayout()
        #self.label = QLabel("Another Window")
        #layout.addWidget(self.label)
        #self.setLayout(layout)
        #self.setWindowTitle("CoSMoS Images")
        self.initUI()

    def initUI(self):
        self.widget = QWidget()                 # Widget that contains the collection of Vertical Box
        self.vbox = QVBoxLayout()               # The Vertical Box that contains the Horizontal Boxes of  labels and buttons


        self.widget.setLayout(self.vbox)

        #Scroll Area Properties
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setWidgetResizable(True)
        self.setWidget(self.widget)

        #self.setGeometry(600, 100, 1000, 1000)
        self.resize(1200, 600)
        self.setWindowTitle("CoSMoS Images")
        self.show()

        return

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__(dataset=None, parameters=None, control, device)
        if dataset is None:
            dataset = "/home/ordabayev/Documents/cosmos_test/test_data/GraceArticlePol2",
        if parameters is None:
            parameters = "/home/ordabayev/Documents/cosmos_test/test_data/GraceArticlePol2/runs/trackerv1.1.2+14.g2c7940f.dirty/control/lr0.005/bs8"

        self.Model = Tracker()
        self.Model.load(dataset, control, device)
        self.Model.load_parameters(parameters)

        self.initUI()

    def initUI(self):
        self.scroll = QScrollArea()             # Scroll Area which contains the widgets, set as the centralWidget
        self.widget = QWidget()                 # Widget that contains the collection of Vertical Box
        self.vbox = QVBoxLayout()               # The Vertical Box that contains the Horizontal Boxes of  labels and buttons

        self.controlPanel()
        self.initParams()
        self.updateParams(0)

        self.widget.setLayout(self.vbox)


        #Scroll Area Properties
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.widget)

        self.setCentralWidget(self.scroll)

        #self.setGeometry(600, 100, 1000, 1000)
        self.resize(1000, 1000)
        self.setWindowTitle("CoSMoS Params")
        self.show()

        return

    def controlPanel(self):
        layout = QHBoxLayout()
        #layout.setMinimumSize(900, 25)

        self.aoiIncr = QPushButton(">")
        self.aoiIncr.clicked.connect(partial(self.updateParams, 1))
        self.aoiIncrLarge = QPushButton(">>")
        self.aoiIncrLarge.clicked.connect(partial(self.updateParams, 10))

        self.aoiDecr = QPushButton("<")
        self.aoiDecr.clicked.connect(partial(self.updateParams, -1))
        self.aoiDecrLarge = QPushButton("<<")
        self.aoiDecrLarge.clicked.connect(partial(self.updateParams, -10))

        self.aoiNumber = QLineEdit("0")
        self.aoiNumber.setValidator(QIntValidator(0, self.Model.data.N-1))
        self.aoiNumber.setMaximumWidth(50)
        self.aoiNumber.setAlignment(Qt.AlignRight)
        self.aoiNumber.returnPressed.connect(partial(self.updateParams, 0))

        self.aoiLabel = QLabel("Aoi")
        self.aoiMax = QLabel(f"/{self.Model.data.N}")
        self.hspacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.w = None
        self.images = QPushButton("toggle images")
        self.images.clicked.connect(self.show_new_window)

        layout.addWidget(self.aoiDecrLarge)
        layout.addWidget(self.aoiDecr)
        layout.addWidget(self.aoiLabel)
        layout.addWidget(self.aoiNumber)
        layout.addWidget(self.aoiMax)
        layout.addWidget(self.aoiIncr)
        layout.addWidget(self.aoiIncrLarge)
        layout.addItem(self.hspacer)
        layout.addWidget(self.images)

        self.vbox.addLayout(layout)

    def show_new_window(self, checked):
        if self.w is None:
            self.lr = pg.LinearRegionItem([400, 500])
            self.plot["z_probs"].addItem(self.lr)
            self.w = AnotherWindow()
            self.initImages()
            self.updateImages()
            self.hist.regionChanged()
            self.lr.sigRegionChangeFinished.connect(self.updateImages)
            #self.w.show()
        
        else:
            self.plot["z_probs"].removeItem(self.lr)
            self.w.close()  # Close window.
            self.w = None  # Discard reference.

    def initImages(self):
        widget = pg.GraphicsLayoutWidget()
        widget.setMinimumSize(900, 500)

        # create plots and items
        self.box = {}
        self.img = {}
        self.label = {}
        self.prob = {}
        for i in range(100):
            r, c = divmod(i, 20)
            self.label[i] = widget.addLabel(text=str(i), row=2*r, col=c)
            self.box[i] = widget.addViewBox(lockAspect=True, row=2*r+1, col=c)
            self.img[i] = pg.ImageItem()
            self.box[i].addItem(self.img[i])
            self.prob[i] = pg.BarGraphItem(x=(-1.,), height=(14.), width=1, pen=C[2], brush=C[2])
            self.box[i].addItem(self.prob[i])

        img = pg.ImageItem(self.Model.data.data.cpu().numpy())
        range_min = np.percentile(self.Model.data.data.cpu().numpy(), 0.5)
        range_max = np.percentile(self.Model.data.data.cpu().numpy(), 99.5)
        self.hist = HistogramLUTGraph(self.img, image=img)
        self.hist.setLevels(
            min=self.Model.data.vmin,
            max=self.Model.data.vmax
        )
        self.hist.setHistogramRange(range_min, range_max)
        widget.addItem(self.hist, col=20, rowspan=10)

        self.w.vbox.addWidget(widget)

    def updateImages(self):
        f1, f2 = self.lr.getRegion()
        f1 = int(f1)
        f2 = int(f2)
        for f in range(f1, f2):
            self.label[(f - f1) % 100].setText(text=str(f))
            self.img[(f - f1) % 100].setImage(
                self.Model.data[int(self.aoiNumber.text()), f].cpu().numpy()
            )
            self.prob[(f - f1) % 100].setOpts(
                height=(self.Model.predictions["z_prob"][int(self.aoiNumber.text()), f]*self.Model.data.D,)
            )

    def initParams(self):
        widget = pg.GraphicsLayoutWidget()
        widget.setMinimumSize(900, 150 * 6)

        # create plots and items
        self.params = ["z_probs", "d/height", "d/width", "d/x", "d/y", "d/background"]
        self.plot = {}
        self.item = {}
        for i, p in enumerate(self.params):
            self.plot[p] = widget.addPlot(row=i, col=0)
            self.plot[p].setLabel("left", p)
            self.plot[p].setXRange(0, self.Model.data.F, padding=0.01)
            self.plot[p].getViewBox().setMouseMode(pg.ViewBox.RectMode)
            if p == "z_probs":
                self.item[p] = self.plot[p].plot(
                    pen=C[2], symbol="o", symbolBrush=C[2],
                    symbolPen=None, symbolSize=5, name="z_probs"
                )
            else:
                k_max = 1 if p.endswith("background") else 2
                for k in range(k_max):
                    self.item[f"{p}.{k}_mean"] = pg.PlotDataItem(
                        pen=C[k], symbol="o", symbolBrush=C[k],
                        symbolPen=None, symbolSize=5, name=k
                        )
                    self.item[f"{p}.{k}_high"] = pg.PlotDataItem(pen=(*C[k], 70))
                    self.item[f"{p}.{k}_low"] = pg.PlotDataItem(pen=(*C[k], 70))
                    self.item[f"{p}.{k}_fill"] = pg.FillBetweenItem(
                        self.item[f"{p}.{k}_high"], self.item[f"{p}.{k}_low"],
                        brush=(*C[k], 70)
                    )

        # add items to plots
        for key, value in self.item.items():
            self.plot[key.split(".")[0]].addItem(value)
                
        # set plot ranges
        self.plot["z_probs"].setYRange(0, 1, padding=0.01)
        self.plot["d/height"].setYRange(0, 7000, padding=0.01)
        self.plot["d/x"].setYRange(-(self.Model.data.D+1)/2, (self.Model.data.D+1)/2, padding=0.01)
        self.plot["d/y"].setYRange(-(self.Model.data.D+1)/2, (self.Model.data.D+1)/2, padding=0.01)
        self.plot["d/background"].setYRange(0, 300, padding=0.01)

        self.vbox.addWidget(widget)

    def updateParams(self, inc):
        n = (int(self.aoiNumber.text()) + inc) % self.Model.data.N
        self.aoiNumber.setText(str(n))
        
        self.Model.n = torch.tensor([n])
        self.Model.frames = torch.arange(self.Model.data.F)
        trace = pyro.poutine.trace(self.Model.guide).get_trace()
        self.Model.n = None
        self.Model.frames = None
        for i, p in enumerate(self.params):
            if p == "z_probs":
                self.item[p].setData(self.Model.predictions["z_prob"][n])
            else:
                hpd = pi(trace.nodes[p]["fn"].sample((500,)).data.squeeze().cpu(), 0.95, dim=0)
                #std = trace.nodes[p]["fn"].variance.data.squeeze().cpu().sqrt()
                mean = trace.nodes[p]["fn"].mean.data.squeeze().cpu()
                k_max = 2
                if p.endswith("background"):
                    k_max = 1
                    #mean, std = mean[None], std[None]
                    mean, hpd = mean[None], hpd[:, None]
                elif p.endswith("x") or p.endswith("y"):
                    mean = mean * (self.Model.data.D+1) - (self.Model.data.D+1)/2
                    hpd = hpd * (self.Model.data.D+1) - (self.Model.data.D+1)/2
                    #std = std * (self.Model.data.D+1) - (self.Model.data.D+1)/2
                elif p.endswith("width"):
                    mean = mean * 2.5 + 0.5
                    hpd = hpd * 2.5 + 0.5
                    #std = std * 2.5 + 0.5
                for k in range(k_max):
                    self.item[f"{p}.{k}_high"].setData(hpd[0, k])
                    self.item[f"{p}.{k}_low"].setData(hpd[1, k])
                    #self.item[f"{p}.{k}_high"].setData(mean[k] + 2 * std[k])
                    #self.item[f"{p}.{k}_low"].setData(mean[k] - 2 * std[k])
                    self.item[f"{p}.{k}_mean"].setData(mean[k])

        if self.w is not None:
            self.updateImages()
