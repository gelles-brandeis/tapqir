import sys
import torch
import matplotlib
matplotlib.use('Qt5Agg')

#from PySide2.QtWidgets import \
#    QMainWindow, QApplication, QFileDialog
from PySide2.QtWidgets import *
from PySide2.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pyqtgraph as pg
import numpy as np

from cosmos.ui.MainWindow import Ui_MainWindow

from cosmos.models.tracker import Tracker
from cosmos.utils.visualize import view_m_probs
import pyro
from pyro import param
import pyro.distributions as dist
from cosmos.ui.utils import plot_graph, construct_graph
from collections import defaultdict

C = {}
C[0] = (31, 119, 180)
C[1] = (255, 127, 14)
C[2] = (44, 160, 44)
C[3] = (214, 39, 40)

class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, params=None):
        fig, self.axes = plt.subplots(len(params), 1, sharex=True,
                               figsize=(9, 1.5*len(params))) 
        super(MplCanvas, self).__init__(fig)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Fixed,
                                   QSizePolicy.Fixed)
        FigureCanvas.updateGeometry(self)
        self._line = defaultdict(lambda: None)
        self._fill_line = {}

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.setupUi(self)
        self.Model = Tracker()


        self.paramGraph = pg.GraphicsLayoutWidget()
        #self.paramGraph.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        #self.paramGraph.resize(1900, 150*6) 
        #self.paramGraph.updateGeometry()

        self.pmean = {}
        self.phigh = {}
        self.plow = {}
        self.pfill = {}
        p1 = self.paramGraph.addPlot(title="Spot probability", row=0, col=0)
        self.pmean["z_probs"] = p1.plot(pen=C[2], name="z_probs")

        p2 = self.paramGraph.addPlot(title="Intensity", row=1, col=0)
        construct_graph(self, p2, "d/height", 0, C)
        construct_graph(self, p2, "d/height", 1, C)

        p3 = self.paramGraph.addPlot(title="Width", row=2, col=0)
        construct_graph(self, p3, "d/width", 0, C)
        construct_graph(self, p3, "d/width", 1, C)

        p4 = self.paramGraph.addPlot(title="x-position", row=3, col=0)
        construct_graph(self, p4, "d/x", 0, C)
        construct_graph(self, p4, "d/x", 1, C)

        p5 = self.paramGraph.addPlot(title="y-position", row=4, col=0)
        construct_graph(self, p5, "d/y", 0, C)
        construct_graph(self, p5, "d/y", 1, C)

        p6 = self.paramGraph.addPlot(title="Background", row=5, col=0)
        construct_graph(self, p6, "d/background", 0, C)

        self.scrollLayout.addWidget(self.paramGraph)
        self.scrollLayout.setAlignment(self.paramGraph, Qt.AlignHCenter)

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
        print(self.pathName.text(), self.controlBox.isChecked(), self.deviceName.currentText())
        try:
            self.Model.load(self.pathName.text(), self.controlBox.isChecked(), self.deviceName.currentText())
            self.Analysis.setEnabled(True)
            self.Parameters.setEnabled(True)
            print("Successful")
        except:
            print("Something went wrong")

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
            self.Model.data.drift.index, trace, params)

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
