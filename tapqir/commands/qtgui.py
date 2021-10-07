# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

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
from qtrangeslider import QRangeSlider

from tapqir.distributions.kspotgammanoise import _gaussian_spots

mpl.use("Qt5Agg")
mpl.rc("text", usetex=True)
mpl.rcParams.update({"font.size": 12})


def config_axis(ax, label, f1, f2, ymin, ymax, xticklabels=False):
    plt.minorticks_on()
    ax.tick_params(
        direction="in",
        which="minor",
        length=1,
        bottom=True,
        top=True,
        left=True,
        right=True,
    )
    ax.tick_params(
        direction="in",
        which="major",
        length=2,
        bottom=True,
        top=True,
        left=True,
        right=True,
    )
    ax.set_ylabel(label)
    ax.set_xlim(f1 - 2, f2 + 1)
    ax.set_ylim(ymin, ymax)
    if not xticklabels:
        ax.set_xticklabels([])


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, constrained_layout=False)
        super(MplCanvas, self).__init__(fig)


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
        self.setWindowTitle("AOI Images")
        self.show()

        return


class MainWindow(QMainWindow):
    def __init__(self, model, path):
        super().__init__()

        self.Model = model
        self.path = path
        self.Model.load(path, data_only=False)
        self.prefix = "d"

        self.initUI()

    def initUI(self):
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        self.scroll = QScrollArea()
        self.widget = QWidget()
        self.vbox = QVBoxLayout()

        self.controlPanel()
        self.initParams()

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

        self.aoiIncr = QPushButton("+1")
        self.aoiIncr.clicked.connect(partial(self.updateParams, 1))
        self.aoiIncrLarge = QPushButton("+10")
        self.aoiIncrLarge.clicked.connect(partial(self.updateParams, 10))

        self.aoiDecr = QPushButton("-1")
        self.aoiDecr.clicked.connect(partial(self.updateParams, -1))
        self.aoiDecrLarge = QPushButton("-10")
        self.aoiDecrLarge.clicked.connect(partial(self.updateParams, -10))

        self.aoiNumber = QLineEdit("0")
        self.aoiNumber.setValidator(QIntValidator(0, self.Model.data.ontarget.N - 1))
        self.aoiNumber.setMaximumWidth(50)
        self.aoiNumber.setAlignment(Qt.AlignRight)
        self.aoiNumber.returnPressed.connect(partial(self.updateParams, 0))

        self.aoiLabel = QLabel("AOI")
        self.aoiMax = QLabel(f"/{self.Model.data.ontarget.N}")
        self.hspacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.frame1Label = QLabel("FrameMin")
        self.frame1 = QLineEdit("0")
        self.frame1.setValidator(QIntValidator(0, self.Model.data.ontarget.F - 2))
        self.frame1.setMaximumWidth(50)
        self.frame1.setAlignment(Qt.AlignRight)
        self.frame1.returnPressed.connect(self.frameChanged)

        self.frame2Label = QLabel("FrameMax")
        self.frame2 = QLineEdit("2")
        self.frame2.setValidator(QIntValidator(1, self.Model.data.ontarget.F - 1))
        self.frame2.setMaximumWidth(50)
        self.frame2.setAlignment(Qt.AlignRight)
        self.frame2.returnPressed.connect(self.frameChanged)

        self.w = None
        self.images = QPushButton("Images")
        self.images.clicked.connect(self.show_new_window)

        layout.addWidget(self.aoiDecrLarge)
        layout.addWidget(self.aoiDecr)
        layout.addWidget(self.aoiLabel)
        layout.addWidget(self.aoiNumber)
        layout.addWidget(self.aoiMax)
        layout.addWidget(self.aoiIncr)
        layout.addWidget(self.aoiIncrLarge)
        layout.addWidget(self.frame1Label)
        layout.addWidget(self.frame1)
        layout.addWidget(self.frame2Label)
        layout.addWidget(self.frame2)
        layout.addItem(self.hspacer)
        layout.addWidget(self.images)

        self.vbox.addLayout(layout)

    def frameChanged(self):
        f1 = int(self.frame1.text())
        f2 = int(self.frame2.text())
        self.range_slider.setValue((f1, f2))

    def updateRange(self):
        f1, f2 = self.range_slider.value()
        self.frame1.setText(str(f1))
        self.frame2.setText(str(f2))
        for ax in self.ax.values():
            ax.set_xlim(f1 - 1, f2 + 1)
        self.sc.draw()

    def show_new_window(self, checked):
        if self.w is None:
            self.w = ImagesWindow()
            self.initImages()
            self.updateImages()
            self.hist.regionChanged()
            self.range_slider.valueChanged.connect(self.updateImages)

        else:
            self.range_slider.valueChanged.disconnect(self.updateImages)
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

        img = pg.ImageItem(self.Model.data.ontarget.images.numpy())
        range_min = np.percentile(self.Model.data.ontarget.images.numpy(), 0.5)
        range_max = np.percentile(self.Model.data.ontarget.images.numpy(), 99.5)
        self.hist = HistogramLUTGraph(self.img, image=img)
        self.hist.setLevels(min=self.Model.data.vmin, max=self.Model.data.vmax)
        self.hist.setHistogramRange(range_min, range_max)
        widget.addItem(self.hist, col=20, rowspan=10)

        self.w.vbox.addWidget(widget)

    def updateImages(self):
        n = int(self.aoiNumber.text())
        f1, _ = self.range_slider.value()
        frames = torch.arange(f1, f1 + 100)
        img_ideal = (
            self.Model.data.offset.mean
            + self.Model.params[f"{self.prefix}/background"]["Mean"][
                n, frames, None, None
            ]
        )
        gaussian = _gaussian_spots(
            self.Model.params[f"{self.prefix}/height"]["Mean"][:, n, frames],
            #  self.Model.params["d/height"]["Mean"][:, n, frames].masked_fill(
            #      self.Model.params["d/m_probs"][:, n, frames] < 0.5, 0.0
            #  ),
            self.Model.params[f"{self.prefix}/width"]["Mean"][:, n, frames],
            self.Model.params[f"{self.prefix}/x"]["Mean"][:, n, frames],
            self.Model.params[f"{self.prefix}/y"]["Mean"][:, n, frames],
            self.Model.data.ontarget.xy[n, frames],
            self.Model.data.ontarget.P,
        )
        img_ideal = img_ideal + gaussian.sum(-4)
        for f in range(f1, f1 + 100):
            self.label[(f - f1) % 100].setText(text=str(f))
            self.img[(f - f1) % 100].setImage(
                self.Model.data.ontarget.images[int(self.aoiNumber.text()), f].numpy()
            )
            # if self.Model._classifier:
            self.prob[(f - f1) % 100].setOpts(
                height=(
                    self.Model.params["p(specific)"][int(self.aoiNumber.text()), f]
                    * self.Model.data.P,
                )
            )
            # ideal image
            self.img_ideal[(f - f1) % 100].setImage(img_ideal[f - f1].cpu().numpy())

            self.img[(f - f1) % 100].setLevels(self.hist.getLevels())
            self.img_ideal[(f - f1) % 100].setLevels(self.hist.getLevels())

    def initParams(self):
        f1, f2 = 0, self.Model.data.ontarget.F

        sc = MplCanvas(self, width=5, height=4, dpi=100)
        toolbar = NavigationToolbar(sc, self)
        self.vbox.addWidget(toolbar)

        self.range_slider = QRangeSlider(Qt.Horizontal)
        self.range_slider.setMinimum(f1)
        self.range_slider.setMaximum(f2 - 1)
        self.range_slider.setValue((f1, f2 - 1))
        self.range_slider.valueChanged.connect(self.updateRange)
        self.vbox.addWidget(self.range_slider)

        self.vbox.addWidget(sc)
        self.sc = sc
        fig = sc.figure
        gs = fig.add_gridspec(
            nrows=6,
            ncols=1,
            top=0.99,
            bottom=0.05,
            left=0.07,
            right=0.98,
            hspace=0.1,
        )
        self.params = ["z", "height", "width", "x", "y", "background"]
        self.ax = {}
        self.item = {}

        self.ax["pspecific"] = fig.add_subplot(gs[0])
        config_axis(self.ax["pspecific"], r"$p(\mathsf{specific})$", f1, f2, -0.1, 1.1)
        (self.item["pspecific"],) = self.ax["pspecific"].plot(
            torch.arange(0, self.Model.data.ontarget.F),
            self.Model.params["p(specific)"][0],
            "o-",
            ms=3,
            lw=1,
            color="C2",
        )

        self.ax["height"] = fig.add_subplot(gs[1])
        config_axis(self.ax["height"], r"$h$", f1, f2, -100, 6000)

        self.ax["width"] = fig.add_subplot(gs[2])
        config_axis(self.ax["width"], r"$w$", f1, f2, 0.5, 2.5)

        self.ax["x"] = fig.add_subplot(gs[3])
        config_axis(self.ax["x"], r"$x$", f1, f2, -9, 9)

        self.ax["y"] = fig.add_subplot(gs[4])
        config_axis(self.ax["y"], r"$y$", f1, f2, -9, 9)

        self.ax["background"] = fig.add_subplot(gs[5])
        config_axis(self.ax["background"], r"$b$", f1, f2, 0, 500, True)
        self.ax["background"].set_xlabel("Time (frame)")

        for p in ["height", "width", "x", "y"]:
            for k in range(self.Model.K):
                (self.item[f"{p}_{k}_mean"],) = self.ax[p].plot(
                    torch.arange(0, self.Model.data.ontarget.F),
                    self.Model.params[f"d/{p}"]["Mean"][k, 0],
                    "o-",
                    ms=3,
                    lw=1,
                    color=f"C{k}",
                )
                self.item[f"{p}_{k}_fill"] = self.ax[p].fill_between(
                    torch.arange(0, self.Model.data.ontarget.F),
                    self.Model.params[f"d/{p}"]["LL"][k, 0],
                    self.Model.params[f"d/{p}"]["UL"][k, 0],
                    alpha=0.3,
                    color=f"C{k}",
                )
        (self.item["background_mean"],) = self.ax["background"].plot(
            torch.arange(0, self.Model.data.ontarget.F),
            self.Model.params["d/background"]["Mean"][0],
            "o-",
            ms=3,
            lw=1,
            color="k",
        )
        self.item["background_fill"] = self.ax["background"].fill_between(
            torch.arange(0, self.Model.data.ontarget.F),
            self.Model.params["d/background"]["LL"][0],
            self.Model.params["d/background"]["UL"][0],
            alpha=0.3,
            color="k",
        )

    def updateParams(self, inc):
        self.n = (int(self.aoiNumber.text()) + inc) % self.Model.data.ontarget.N
        self.aoiNumber.setText(str(self.n))

        for p in self.params:
            if p == "z":
                if "p(specific)" in self.Model.params:
                    self.item["pspecific"].set_ydata(
                        self.Model.params["p(specific)"][self.n]
                    )
            elif p == "background":
                self.item[f"{p}_fill"].remove()
                self.item[f"{p}_fill"] = self.ax[p].fill_between(
                    torch.arange(0, self.Model.data.ontarget.F),
                    self.Model.params[f"d/{p}"]["LL"][self.n],
                    self.Model.params[f"d/{p}"]["UL"][self.n],
                    alpha=0.3,
                    color="k",
                )
                self.item[f"{p}_mean"].set_ydata(
                    self.Model.params[f"{self.prefix}/{p}"]["Mean"][self.n]
                )
            else:
                for k in range(self.Model.K):
                    self.item[f"{p}_{k}_fill"].remove()
                    self.item[f"{p}_{k}_fill"] = self.ax[p].fill_between(
                        torch.arange(0, self.Model.data.ontarget.F),
                        self.Model.params[f"d/{p}"]["LL"][k, self.n],
                        self.Model.params[f"d/{p}"]["UL"][k, self.n],
                        alpha=0.3,
                        color=f"C{k}",
                    )
                    self.item[f"{p}_{k}_mean"].set_ydata(
                        self.Model.params[f"{self.prefix}/{p}"]["Mean"][k, self.n]
                    )
        self.sc.draw()

        if self.w is not None:
            self.updateImages()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
