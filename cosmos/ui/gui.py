import sys
import torch
import matplotlib
matplotlib.use('Qt5Agg')

#from PySide2.QtWidgets import \
#    QMainWindow, QApplication, QFileDialog
from PySide2.QtWidgets import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from MainWindow import Ui_MainWindow

from cosmos.models.tracker import Tracker
from cosmos.utils.visualize import view_m_probs
import pyro
from pyro import param
import pyro.distributions as dist
from cosmos.ui.utils import plot_dist

class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, params=None):
        fig, self.axes = plt.subplots(len(params), 1, sharex=True,
                               figsize=(15, 2.5*len(params))) 
        super(MplCanvas, self).__init__(fig)

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.Model = Tracker()

        #self.canvas = MplCanvas(self, width=1, height=1, dpi=100)
        #self.canvas.axes.plot([0,1,2,3,4], [10,1,20,3,40])

    def refreshAll(self):
        '''
        Updates the widgets whenever an interaction happens.
        '''
        #self.textEdit.setText( self.model.getFileContents() )
        #self.Model.load(self.dirName, True, "cuda")
        pass

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
            print("Successful")
        except:
            print("Something went wrong")

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
        #try:
        self.Model.load_parameters(self.paramPath.text())
        #import pdb; pdb.set_trace()
        self.aoiNumber.setMaximum(self.Model.data.N - 1)
        self.aoiNumber.setSingleStep(1)
        self.aoiNumber.setPageStep(10)
        #self.aoiNumber.setTickInterval(20)
        self.Model.data.predictions = self.Model.predictions
        
        params = ["d/height", "d/width", "d/x", "d/y", "d/background"]
        self.canvas = MplCanvas(self, params=params)
        toolbar = NavigationToolbar(self.canvas, self)
        self.paramPlot.addWidget(toolbar)
        self.paramPlot.addWidget(self.canvas)

        self.aoiNumber.setValue(1)

        print("Successful")
        #except:
        #    print("Something went wrong")

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

    def plotParamSlot(self):
        self.label_8.setText("Aoi: {}/{}".format(self.aoiNumber.value(), self.Model.data.N - 1))

        self.Model.n = torch.tensor([self.aoiNumber.value()])
        self.Model.frames = torch.arange(self.Model.data.F)
        trace = pyro.poutine.trace(self.Model.guide).get_trace()
        self.Model.n = None
        self.Model.frames = None
        params = ["d/height", "d/width", "d/x", "d/y", "d/background"]
        plot_dist(self.canvas.axes, self.Model.data.drift.index, trace, params, ci=0.95)
        self.canvas.draw()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
