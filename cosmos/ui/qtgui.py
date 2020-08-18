from pyqtgraph import HistogramLUTItem
from PySide2.QtWidgets import (QWidget, QLineEdit, QLabel, QPushButton, QScrollArea, QApplication,
                               QHBoxLayout, QVBoxLayout, QMainWindow, QSizePolicy, QSpacerItem)
from PySide2.QtCore import Qt
from PySide2.QtGui import QIntValidator
import sys
from functools import partial

import pyqtgraph as pg
import torch
import numpy as np
import pyro
from pyro.ops.stats import pi

C = {}
C[0] = (31, 119, 180)
C[1] = (255, 127, 14)
C[2] = (44, 160, 44)
C[3] = (214, 39, 40)


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


class ImagesWindow(QScrollArea):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.widget = QWidget()
        self.vbox = QVBoxLayout()

        self.widget.setLayout(self.vbox)

        # Scroll Area Properties
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setWidgetResizable(True)
        self.setWidget(self.widget)

        self.resize(1200, 600)
        self.setWindowTitle("CoSMoS Images")
        self.show()

        return


class ZoomWindow(QScrollArea):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.widget = QWidget()
        self.vbox = QVBoxLayout()

        self.widget.setLayout(self.vbox)

        # Scroll Area Properties
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setWidgetResizable(True)
        self.setWidget(self.widget)

        self.resize(1000, 1000)
        self.setWindowTitle("Zoom Parameters")
        self.show()

        return


class MainWindow(QMainWindow):

    def __init__(self, model, dataset, parameters, control=False):
        super().__init__()

        self.Model = model
        self.parameters = parameters
        self.Model.load(dataset, control, "cpu")
        self.Model.load_parameters(self.parameters)

        self.initUI()

    def initUI(self):
        self.scroll = QScrollArea()
        self.widget = QWidget()
        self.vbox = QVBoxLayout()

        self.controlPanel()
        self.initParams()
        self.updateParams(0)

        self.widget.setLayout(self.vbox)

        # Scroll Area Properties
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.widget)

        self.setCentralWidget(self.scroll)

        self.resize(1000, 1000)
        self.setWindowTitle("CoSMoS Params")
        self.show()

        return

    def controlPanel(self):
        layout = QHBoxLayout()

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

        self.refresh = QPushButton("refresh")
        self.refresh.clicked.connect(self.refreshParams)

        layout.addWidget(self.aoiDecrLarge)
        layout.addWidget(self.aoiDecr)
        layout.addWidget(self.aoiLabel)
        layout.addWidget(self.aoiNumber)
        layout.addWidget(self.aoiMax)
        layout.addWidget(self.aoiIncr)
        layout.addWidget(self.aoiIncrLarge)
        layout.addItem(self.hspacer)
        layout.addWidget(self.images)
        layout.addWidget(self.refresh)

        self.vbox.addLayout(layout)

    def refreshParams(self):
        self.Model.load_parameters(self.parameters)
        self.updateParams(0)

    def show_new_window(self, checked):
        if self.w is None:
            self.lr = pg.LinearRegionItem([400, 500])
            self.plot["z"].addItem(self.lr)
            self.w = ImagesWindow()
            self.initImages()
            self.updateImages()
            self.hist.regionChanged()
            self.lr.sigRegionChangeFinished.connect(self.updateImages)
            # self.w.show()

        else:
            self.plot["z"].removeItem(self.lr)
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

            self.img[(f - f1) % 100].setLevels(self.hist.getLevels())

    def initParams(self):
        widget = pg.GraphicsLayoutWidget()
        widget.setMinimumSize(900, 150 * 6)

        # create plots and items
        self.params = ["z", "d/height", "d/width", "d/x", "d/y", "d/background"]
        self.items = ["z", "d/height_0", "d/height_1", "d/width_0", "d/width_1",
                      "d/x_0", "d/x_1", "d/y_0", "d/y_1", "d/background"]
        self.plot = {}
        self.item = {}
        for i, p in enumerate(self.params):
            self.plot[p] = widget.addPlot(row=i, col=0)
            self.plot[p].addLegend()
            self.plot[p].setLabel("left", p)
            self.plot[p].setXRange(0, self.Model.data.F, padding=0.01)
            self.plot[p].getViewBox().setMouseMode(pg.ViewBox.RectMode)
        for i, p in enumerate(self.items):
            if p == "z":
                self.item[f"{p}_probs"] = pg.PlotDataItem(
                    pen=C[2], symbol="o", symbolBrush=C[2],
                    symbolPen=None, symbolSize=5, name="z_probs"
                )
                self.item[f"{p}_binary"] = pg.PlotDataItem(
                    pen=C[3], symbol=None,
                    symbolPen=None, name="z_binary"
                )
            else:
                k = 0 if p.endswith("background") else int(p.split("_")[-1])

                self.item[f"{p}_mean"] = pg.PlotDataItem(
                    pen=C[k], symbol="o", symbolBrush=C[k],
                    symbolPen=None, symbolSize=5, name=k
                )
                self.item[f"{p}_high"] = pg.PlotDataItem(pen=(*C[k], 70))
                self.item[f"{p}_low"] = pg.PlotDataItem(pen=(*C[k], 70))
                self.item[f"{p}_fill"] = pg.FillBetweenItem(
                    self.item[f"{p}_high"], self.item[f"{p}_low"],
                    brush=(*C[k], 70)
                )

        # add items to plots
        for key, value in self.item.items():
            self.plot[key.split("_")[0]].addItem(value)

        # set plot ranges
        self.plot["z"].setYRange(0, 1, padding=0.01)
        # self.plot["d/height"].setYRange(0, 7000, padding=0.01)
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
        for i, p in enumerate(self.items):
            if p == "z":
                self.item[f"{p}_probs"].setData(self.Model.predictions["z_prob"][n])
                self.item[f"{p}_binary"].setData(self.Model.predictions["z"][n])
            else:
                hpd = pi(trace.nodes[p]["fn"].sample((500,)).data.squeeze().cpu(), 0.95, dim=0)
                mean = trace.nodes[p]["fn"].mean.data.squeeze().cpu()
                self.item[f"{p}_high"].setData(hpd[0])
                self.item[f"{p}_low"].setData(hpd[1])
                self.item[f"{p}_mean"].setData(mean)

        if self.w is not None:
            self.updateImages()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
