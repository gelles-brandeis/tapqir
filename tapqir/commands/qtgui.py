# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

import sys
from functools import partial

import numpy as np
import pyqtgraph as pg
import torch
from pyqtgraph import HistogramLUTItem
from pyro.ops.stats import quantile
from pyroapi import pyro
from PySide2.QtCore import Qt
from PySide2.QtGui import QIntValidator
from PySide2.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

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
        self.Model.load(dataset, control)
        self.Model.load_parameters(self.parameters)

        self.initUI()

    def initUI(self):
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
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

        self.resize(1200, 1000)
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
        self.aoiNumber.setValidator(QIntValidator(0, self.Model.data.N - 1))
        self.aoiNumber.setMaximumWidth(50)
        self.aoiNumber.setAlignment(Qt.AlignRight)
        self.aoiNumber.returnPressed.connect(partial(self.updateParams, 0))

        self.aoiLabel = QLabel("Aoi")
        self.aoiMax = QLabel(f"/{self.Model.data.N}")
        self.hspacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.w = None
        self.images = QPushButton("Zoom/Images")
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

    def updateRange(self):
        for p in self.params:
            self.plot[p].setXRange(*self.lr.getRegion(), padding=0.01)

    def show_new_window(self, checked):
        if self.w is None:
            self.lr = pg.LinearRegionItem([0, min(self.Model.data.F, 100)])
            self.plot["zoom"].addItem(self.lr)
            self.w = ImagesWindow()
            self.initImages()
            self.updateImages()
            self.updateRange()
            self.hist.regionChanged()
            self.lr.sigRegionChangeFinished.connect(self.updateImages)
            self.lr.sigRegionChanged.connect(self.updateRange)
            # self.w.show()

        else:
            self.plot["zoom"].removeItem(self.lr)
            for p in self.params:
                self.plot[p].setXRange(0, self.Model.data.F, padding=0.01)
            self.w.close()  # Close window.
            self.w = None  # Discard reference.

    def initImages(self):
        widget = pg.GraphicsLayoutWidget()
        widget.setMinimumSize(900, 500)

        # create plots and items
        self.box = {}
        self.box_ideal = {}
        self.img = {}
        self.img_ideal = {}
        self.label = {}
        self.prob = {}
        for i in range(100):
            r, c = divmod(i, 20)
            self.label[i] = widget.addLabel(text=str(i), row=3 * r, col=c)
            self.box[i] = widget.addViewBox(lockAspect=True, row=3 * r + 1, col=c)
            self.img[i] = pg.ImageItem()
            self.box[i].addItem(self.img[i])
            self.prob[i] = pg.BarGraphItem(
                x=(-1.0,), height=(14.0), width=1, pen=C[2], brush=C[2]
            )
            self.box[i].addItem(self.prob[i])
            # plot ideal image
            self.box_ideal[i] = widget.addViewBox(lockAspect=True, row=3 * r + 2, col=c)
            self.img_ideal[i] = pg.ImageItem()
            self.box_ideal[i].addItem(self.img_ideal[i])

        img = pg.ImageItem(self.Model.data.data.cpu().numpy())
        range_min = np.percentile(self.Model.data.data.cpu().numpy(), 0.5)
        range_max = np.percentile(self.Model.data.data.cpu().numpy(), 99.5)
        self.hist = HistogramLUTGraph(self.img, image=img)
        self.hist.setLevels(min=self.Model.data.vmin, max=self.Model.data.vmax)
        self.hist.setHistogramRange(range_min, range_max)
        widget.addItem(self.hist, col=20, rowspan=10)

        self.w.vbox.addWidget(widget)

    def updateImages(self):
        n = int(self.aoiNumber.text())
        f1, f2 = self.lr.getRegion()
        f1 = int(f1)
        f2 = int(f2)
        frames = torch.arange(f1, f2)
        img_ideal = (
            self.Model.data.offset_mean
            + pyro.param("d/b_loc").data[n, frames, None, None]
        )
        gaussian = self.Model.data_loc(
            pyro.param("d/h_loc")
            .data[:, n, frames]
            .masked_fill(self.Model.m_probs[:, n, frames] < 0.5, 0.0),
            pyro.param("d/w_mean").data[:, n, frames],
            pyro.param("d/x_mean").data[:, n, frames],
            pyro.param("d/y_mean").data[:, n, frames],
            n,
            frames,
        )
        img_ideal = img_ideal + gaussian.sum(-4)
        for f in range(f1, f2):
            self.label[(f - f1) % 100].setText(text=str(f))
            self.img[(f - f1) % 100].setImage(
                self.Model.data[int(self.aoiNumber.text()), f].cpu().numpy()
            )
            self.prob[(f - f1) % 100].setOpts(
                height=(
                    self.Model.z_probs[:, int(self.aoiNumber.text()), f].sum()
                    * self.Model.data.D,
                )
            )
            # ideal image
            self.img_ideal[(f - f1) % 100].setImage(img_ideal[f - f1].cpu().numpy())

            self.img[(f - f1) % 100].setLevels(self.hist.getLevels())
            self.img_ideal[(f - f1) % 100].setLevels(self.hist.getLevels())

    def initParams(self):
        widget = pg.GraphicsLayoutWidget()
        widget.setMinimumSize(1080, 150 * 6)
        widget.addLabel("", row=0, col=1)
        widget.addLabel("", row=0, col=2)
        widget.addLabel("Parameters", row=0, col=0)
        widget.addLabel("", row=0, col=3)
        widget.addLabel("", row=0, col=4)
        widget.addLabel("Histogram", row=0, col=5)

        # create plots and items
        self.params = ["z", "d/height", "d/width", "d/x", "d/y", "d/background"]
        self.sites = [
            "d/height_0",
            "d/height_1",
            "d/width_0",
            "d/width_1",
            "d/x_0",
            "d/x_1",
            "d/y_0",
            "d/y_1",
            "d/background",
        ]
        self.plot = {}
        self.item = {}
        self.plot["zoom"] = widget.addPlot(row=1, col=0, colspan=5)
        self.plot["zoom"].setLabel("left", "zoom")
        self.plot["zoom"].setXRange(0, self.Model.data.F, padding=0.01)
        self.item["zoom"] = pg.PlotDataItem(
            pen=C[2], symbol="o", symbolBrush=C[2], symbolPen=None, symbolSize=5
        )
        offset_max = np.percentile(self.Model.data.offset.cpu().numpy(), 99.5)
        offset_min = np.percentile(self.Model.data.offset.cpu().numpy(), 0.5)
        y, x = np.histogram(
            self.Model.data.offset.cpu().numpy(),
            range=(offset_min, offset_max),
            bins=max(1, int(offset_max - offset_min)),
            density=True,
        )
        plt = widget.addPlot(row=1, col=5)
        plt.plot(
            x,
            y,
            stepMode="center",
            fillLevel=0,
            fillOutline=True,
            brush=(0, 0, 255, 70),
        )
        for i, p in enumerate(self.params):
            self.plot[p] = widget.addPlot(row=i + 2, col=0, colspan=5)
            self.plot[p].addLegend()
            self.plot[p].setLabel("left", p)
            self.plot[p].setXRange(0, self.Model.data.F, padding=0.01)
            self.plot[p].getViewBox().setMouseMode(pg.ViewBox.RectMode)

            self.plot[f"{p}Hist"] = widget.addPlot(row=i + 2, col=5)
            if p == "z":
                y, x = np.histogram(self.Model.z_probs.numpy(), bins=50)
            elif p == "d/height":
                y, x = np.histogram(
                    pyro.param("d/h_loc").data.reshape(-1).numpy(),
                    bins=50,
                    weights=self.Model.m_probs.reshape(-1).numpy(),
                )
                self.plot[f"{p}Hist"].setXRange(
                    0,
                    quantile(pyro.param("d/h_loc").data.flatten(), 0.99).item() * 1.3,
                    padding=0.01,
                )

                yz, xz = np.histogram(
                    pyro.param("d/h_loc").data.reshape(-1).numpy(),
                    bins=50,
                    weights=self.Model.z_probs.reshape(-1).numpy(),
                )

                yj, xj = np.histogram(
                    pyro.param("d/h_loc").data.reshape(-1).numpy(),
                    bins=50,
                    weights=self.Model.j_probs.reshape(-1).numpy(),
                )

            elif p == "d/width":
                y, x = np.histogram(
                    pyro.param("d/w_mean").data.reshape(-1).numpy(),
                    bins=50,
                    weights=self.Model.m_probs.reshape(-1).numpy(),
                )

                yz, xz = np.histogram(
                    pyro.param("d/w_mean").data.reshape(-1).numpy(),
                    bins=50,
                    weights=self.Model.z_probs.reshape(-1).numpy(),
                )

                yj, xj = np.histogram(
                    pyro.param("d/w_mean").data.reshape(-1).numpy(),
                    bins=50,
                    weights=self.Model.j_probs.reshape(-1).numpy(),
                )

            elif p == "d/x":
                y, x = np.histogram(
                    pyro.param("d/x_mean").data.reshape(-1).numpy(),
                    bins=50,
                    weights=self.Model.m_probs.reshape(-1).numpy(),
                )

                yz, xz = np.histogram(
                    pyro.param("d/x_mean").data.reshape(-1).numpy(),
                    bins=50,
                    weights=self.Model.z_probs.reshape(-1).numpy(),
                )

                yj, xj = np.histogram(
                    pyro.param("d/x_mean").data.reshape(-1).numpy(),
                    bins=50,
                    weights=self.Model.j_probs.reshape(-1).numpy(),
                )

            elif p == "d/y":
                y, x = np.histogram(
                    pyro.param("d/y_mean").data.reshape(-1).numpy(),
                    bins=50,
                    weights=self.Model.m_probs.reshape(-1).numpy(),
                )

                yz, xz = np.histogram(
                    pyro.param("d/y_mean").data.reshape(-1).numpy(),
                    bins=50,
                    weights=self.Model.z_probs.reshape(-1).numpy(),
                )

                yj, xj = np.histogram(
                    pyro.param("d/y_mean").data.reshape(-1).numpy(),
                    bins=50,
                    weights=self.Model.j_probs.reshape(-1).numpy(),
                )

            elif p == "d/background":
                y, x = np.histogram(
                    pyro.param("d/b_loc").data.reshape(-1).numpy(), bins=50
                )

            self.item[f"{p}Hist_m"] = pg.PlotDataItem(
                x,
                y,
                stepMode="center",
                fillLevel=0,
                fillOutline=True,
                brush=(0, 0, 255, 30),
            )
            if p in ["d/height", "d/width", "d/x", "d/y"]:
                self.item[f"{p}Hist_z"] = pg.PlotDataItem(
                    xz, yz, stepMode="center", fillLevel=0, fillOutline=False, pen=C[2]
                )
                self.item[f"{p}Hist_j"] = pg.PlotDataItem(
                    xj, yj, stepMode="center", fillLevel=0, fillOutline=False, pen=C[3]
                )

        for p in self.params:
            if p == "z":
                self.item["z_label"] = pg.PlotDataItem(
                    pen=C[3],
                    symbol="o",
                    symbolBrush=C[3],
                    symbolPen=None,
                    symbolSize=5,
                    name="z_label",
                )
                self.item[f"{p}_probs"] = pg.PlotDataItem(
                    pen=C[2],
                    symbol="o",
                    symbolBrush=C[2],
                    symbolPen=None,
                    symbolSize=5,
                    name="z_probs",
                )
            elif p.endswith("background"):
                k = 0
                self.item[f"{p}_mean"] = pg.PlotDataItem(
                    pen=C[k],
                    symbol="o",
                    symbolBrush=C[k],
                    symbolPen=None,
                    symbolSize=5,
                    name=k,
                )
                self.item[f"{p}_ul"] = pg.PlotDataItem(pen=(*C[k], 70))
                self.item[f"{p}_ll"] = pg.PlotDataItem(pen=(*C[k], 70))
                self.item[f"{p}_fill"] = pg.FillBetweenItem(
                    self.item[f"{p}_ul"], self.item[f"{p}_ll"], brush=(*C[k], 70)
                )
            else:

                for k in range(self.Model.K):
                    self.item[f"{p}_{k}_mean"] = pg.PlotDataItem(
                        pen=C[k],
                        symbol="o",
                        symbolBrush=C[k],
                        symbolPen=None,
                        symbolSize=5,
                        name=k,
                    )
                    self.item[f"{p}_{k}_ul"] = pg.PlotDataItem(pen=(*C[k], 70))
                    self.item[f"{p}_{k}_ll"] = pg.PlotDataItem(pen=(*C[k], 70))
                    self.item[f"{p}_{k}_fill"] = pg.FillBetweenItem(
                        self.item[f"{p}_{k}_ul"],
                        self.item[f"{p}_{k}_ll"],
                        brush=(*C[k], 70),
                    )

        # add items to plots
        for key, value in self.item.items():
            self.plot[key.split("_")[0]].addItem(value)

        # set plot ranges
        self.plot["z"].setYRange(0, 1, padding=0.01)
        self.plot["d/x"].setYRange(
            -(self.Model.data.D + 1) / 2, (self.Model.data.D + 1) / 2, padding=0.01
        )
        self.plot["d/y"].setYRange(
            -(self.Model.data.D + 1) / 2, (self.Model.data.D + 1) / 2, padding=0.01
        )
        self.plot["d/background"].setYRange(
            0,
            quantile(pyro.param("d/b_loc").data.flatten(), 0.99).item() * 1.1,
            padding=0.01,
        )

        self.vbox.addWidget(widget)

    def updateParams(self, inc):
        n = (int(self.aoiNumber.text()) + inc) % self.Model.data.N
        self.aoiNumber.setText(str(n))

        self.item["zoom"].setData(self.Model.z_marginal[n])
        for p in self.params:
            if p == "z":
                self.item[f"{p}_probs"].setData(self.Model.z_marginal[n])
                self.item["z_label"].setData(self.Model.data.labels["z"][n])
            elif p == "d/background":
                k = 0
                self.item[f"{p}_ul"].setData(self.Model.local_params[f"{p}_ul"][n])
                self.item[f"{p}_ll"].setData(self.Model.local_params[f"{p}_ll"][n])
                self.item[f"{p}_mean"].setData(self.Model.local_params[f"{p}_mean"][n])
            else:
                for k in range(self.Model.K):
                    self.item[f"{p}_{k}_ul"].setData(
                        self.Model.local_params[f"{p}_{k}_ul"][n]
                    )
                    self.item[f"{p}_{k}_ll"].setData(
                        self.Model.local_params[f"{p}_{k}_ll"][n]
                    )
                    self.item[f"{p}_{k}_mean"].setData(
                        self.Model.local_params[f"{p}_{k}_mean"][n]
                    )

        if self.w is not None:
            self.updateImages()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
