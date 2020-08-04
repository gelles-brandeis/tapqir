import sys
import torch

#from PySide2.QtWidgets import \
#    QMainWindow, QApplication, QFileDialog
from PySide2.QtWidgets import *
from PySide2.QtCore import Qt

import pyqtgraph as pg
import numpy as np

from cosmos.ui.MainWindow import Ui_MainWindow

from cosmos.models.tracker import Tracker
from cosmos.utils.visualize import view_m_probs
import pyro
from pyro import param
import pyro.distributions as dist
from cosmos.ui.utils import plot_graph, construct_graph, construct_image, \
    HistogramLUTGraph
from collections import defaultdict

C = {}
C[0] = (31, 119, 180)
C[1] = (255, 127, 14)
C[2] = (44, 160, 44)
C[3] = (214, 39, 40)

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.setupUi(self)


    ### Data ###
    def browseSlot(self):
        options = QFileDialog.DontResolveSymlinks
        options |= QFileDialog.DontUseNativeDialog
        options |= QFileDialog.ShowDirsOnly
        pathName = QFileDialog.getExistingDirectory(
                    None,
                    "Select Data Folder",
                    "",
                    options=options)

        self.pathName.setText(pathName)

    def loadDataSlot(self):
        try:
            self.Model = Tracker()
            self.Model.load(self.pathName.text(), self.controlBox.isChecked(), self.deviceName.currentText())

            self.dataGraph = pg.GraphicsLayoutWidget()
            v = self.dataGraph.addViewBox(lockAspect=True, row=0, col=0)
            im = pg.ImageItem(self.Model.data.data.mean(axis=(0, 1)).cpu().numpy())
            v.addItem(im)

            p = self.dataGraph.addPlot(title="Intensity histogram", row=0, col=1)
            y, x = np.histogram(self.Model.data.data.cpu().reshape(-1), bins=50)
            p.plot(x, y, stepMode=True, fillLevel=0, fillOutline=True, brush=(0,0,255,150))
            self.dataLayout.addWidget(self.dataGraph)

            self.Analysis.setEnabled(True)
            self.Parameters.setEnabled(True)
        except:
            self.Analysis.setEnabled(False)
            self.Parameters.setEnabled(False)

    ### Parameters ###
    def browseParamSlot(self):
        options = QFileDialog.DontResolveSymlinks
        options |= QFileDialog.DontUseNativeDialog
        options |= QFileDialog.ShowDirsOnly
        pathName = QFileDialog.getExistingDirectory(
                    None,
                    "Select Parameter Folder",
                    "",
                    options=options)

        self.paramPath.setText(pathName)

    def loadParamSlot(self):
        self.Model.load_parameters(self.paramPath.text())
        self.aoiNumber.setRange(0, self.Model.data.N - 1)
        self.aoiMax.setText("/{}".format(self.Model.data.N - 1))

        self.paramGraph = pg.GraphicsLayoutWidget()

        self.pmean = {}
        self.phigh = {}
        self.plow = {}
        self.pfill = {}
        p1 = self.paramGraph.addPlot(title="Spot probability", row=0, col=0)
        p1.setXRange(-2, self.Model.data.F + 2)
        self.pmean["z_probs"] = p1.plot(
            pen=C[2], symbol="o", symbolBrush=C[2], symbolPen=None, symbolSize=5, name="z_probs")
        p1.setLabel("left", "Probability")
        self.lr = pg.LinearRegionItem([400, 500])
        p1.addItem(self.lr)
        self.lr.sigRegionChangeFinished.connect(self.updateImages)

        p2 = self.paramGraph.addPlot(title="Intensity", row=1, col=0)
        p2.setYRange(0, 7000)
        p2.enableAutoRange("xy", False)
        #p2.setXLink(p1)
        p2.setXRange(-2, self.Model.data.F + 2)
        construct_graph(self, p2, "d/height", 0, C)
        construct_graph(self, p2, "d/height", 1, C)

        p3 = self.paramGraph.addPlot(title="Width", row=2, col=0)
        p3.setYRange(0, 3)
        p3.enableAutoRange("xy", False)
        #p3.setXLink(p1)
        p3.setXRange(-2, self.Model.data.F + 2)
        construct_graph(self, p3, "d/width", 0, C)
        construct_graph(self, p3, "d/width", 1, C)

        p4 = self.paramGraph.addPlot(title="x-position", row=3, col=0)
        p4.setYRange(0, 1)
        #p4.setXRange(-2, self.Model.data.F + 2)
        p4.enableAutoRange("xy", False)
        #p4.enableAutoRange("y", False)
        p4.setXLink(p1)
        construct_graph(self, p4, "d/x", 0, C)
        construct_graph(self, p4, "d/x", 1, C)

        p5 = self.paramGraph.addPlot(title="y-position", row=4, col=0)
        p5.setYRange(0, 1)
        p5.setXRange(-2, self.Model.data.F + 2)
        p5.enableAutoRange("y", False)
        #p5.setXLink(p1)
        construct_graph(self, p5, "d/y", 0, C)
        construct_graph(self, p5, "d/y", 1, C)

        p6 = self.paramGraph.addPlot(title="Background", row=5, col=0)
        p6.setYRange(0, 300)
        p6.setXRange(-2, self.Model.data.F + 2)
        #p6.setXLink(p1)
        construct_graph(self, p6, "d/background", 0, C)
        p6.setLabel("bottom", "Time", units="frame")

        self.paramGraph.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.paramGraph.setFixedSize(900, 150 * 6)
        self.scrollContent.addWidget(self.paramGraph)
        self.scrollContent.setAlignment(self.paramGraph, Qt.AlignTop)
        #self.scrollLayout.setAlignment(self.paramGraph, Qt.AlignHCenter | Qt.AlignTop)

        ### images
        self.imagesGraph = pg.GraphicsLayoutWidget()
        self.imagesGraph.setFixedSize(1200, 900)
        #self.imagesGraph.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        #self.imagesGraph.updateGeometry()
        self.img = {}

        for i in range(100):
            r, c = divmod(i, 20)
            v = self.imagesGraph.addViewBox(lockAspect=True, row=r, col=c)
            construct_image(self, v, i)

        img = pg.ImageItem(self.Model.data.data.cpu().numpy())
        hist = HistogramLUTGraph(self.img, image=img)
        self.imagesGraph.addItem(hist, col=20, rowspan=5)

        self.scrollContent.addWidget(self.imagesGraph)
        self.scrollContent.setAlignment(self.imagesGraph, Qt.AlignTop)
        #self.scrollContent.setAlignment(self.imagesGraph, Qt.AlignHCenter | Qt.AlignTop)

        self.aoiNumber.setValue(1)

        print("Successful")

    def plotParamSlot(self):
        self.Model.n = torch.tensor([self.aoiNumber.value()])
        self.Model.frames = torch.arange(self.Model.data.F)
        trace = pyro.poutine.trace(self.Model.guide).get_trace()
        self.Model.n = None
        self.Model.frames = None
        params = ["z_probs", "d/height", "d/width", "d/x", "d/y", "d/background"]
            
        plot_graph(
            self,
            self.Model.predictions, self.aoiNumber.value(),
            self.Model.data.drift.index, trace, params
        )

        for f in range(100):
            #self.img[f].setImage(np.random.normal(size=(14,14)))
            self.img[f].setImage(self.Model.data[self.aoiNumber.value(), f].cpu().numpy())

    def updateImages(self):
        f1, f2 = self.lr.getRegion()
        f1 = int(f1)
        f2 = int(f2)
        print(f1, f2)
        for f in range(f1, f2):
            self.img[f % 100].setImage(self.Model.data[self.aoiNumber.value(), f].cpu().numpy())

    ### Analysis ###
    def runAnalysisSlot(self):
        print(self.batchSize.value(), self.learningRate.value(), self.numIter.value())
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(self.numIter.value())
        self.Model.settings(self.learningRate.value(), self.batchSize.value())
        for i in range(self.numIter.value()):
            self.Model.epoch_loss = self.Model.svi.step()
            if not self.Model.iter % 100:
                self.Model.infer()
                self.Model.save_checkpoint()
            self.Model.iter += 1
            self.progressBar.setValue(i)
        print("Successful")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
