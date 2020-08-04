# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainwindow.ui'
##
## Created by: Qt User Interface Compiler version 5.15.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import (QCoreApplication, QDate, QDateTime, QMetaObject,
    QObject, QPoint, QRect, QSize, QTime, QUrl, Qt)
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
    QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter,
    QPixmap, QRadialGradient)
from PySide2.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setTabShape(QTabWidget.Rounded)
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.horizontalLayout_13 = QHBoxLayout(self.tab)
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.verticalLayout_7 = QVBoxLayout()
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.formLayout = QFormLayout()
        self.formLayout.setObjectName(u"formLayout")
        self.label_3 = QLabel(self.tab)
        self.label_3.setObjectName(u"label_3")

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.label_3)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.deviceName = QComboBox(self.tab)
        self.deviceName.addItem("")
        self.deviceName.addItem("")
        self.deviceName.addItem("")
        self.deviceName.setObjectName(u"deviceName")

        self.horizontalLayout_2.addWidget(self.deviceName)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_2)


        self.formLayout.setLayout(0, QFormLayout.FieldRole, self.horizontalLayout_2)

        self.label_2 = QLabel(self.tab)
        self.label_2.setObjectName(u"label_2")

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.label_2)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.controlBox = QCheckBox(self.tab)
        self.controlBox.setObjectName(u"controlBox")
        self.controlBox.setLayoutDirection(Qt.RightToLeft)

        self.horizontalLayout_3.addWidget(self.controlBox)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer)


        self.formLayout.setLayout(1, QFormLayout.FieldRole, self.horizontalLayout_3)

        self.label = QLabel(self.tab)
        self.label.setObjectName(u"label")

        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.label)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.pathName = QLineEdit(self.tab)
        self.pathName.setObjectName(u"pathName")

        self.horizontalLayout.addWidget(self.pathName)

        self.pushButton = QPushButton(self.tab)
        self.pushButton.setObjectName(u"pushButton")

        self.horizontalLayout.addWidget(self.pushButton)


        self.formLayout.setLayout(2, QFormLayout.FieldRole, self.horizontalLayout)


        self.verticalLayout_7.addLayout(self.formLayout)

        self.pushButton_2 = QPushButton(self.tab)
        self.pushButton_2.setObjectName(u"pushButton_2")

        self.verticalLayout_7.addWidget(self.pushButton_2)

        self.line_2 = QFrame(self.tab)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.HLine)
        self.line_2.setFrameShadow(QFrame.Sunken)

        self.verticalLayout_7.addWidget(self.line_2)

        self.scrollArea_2 = QScrollArea(self.tab)
        self.scrollArea_2.setObjectName(u"scrollArea_2")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scrollArea_2.sizePolicy().hasHeightForWidth())
        self.scrollArea_2.setSizePolicy(sizePolicy)
        self.scrollArea_2.setWidgetResizable(True)
        self.scrollAreaWidgetContents_2 = QWidget()
        self.scrollAreaWidgetContents_2.setObjectName(u"scrollAreaWidgetContents_2")
        self.scrollAreaWidgetContents_2.setGeometry(QRect(0, 0, 744, 359))
        self.dataLayout = QHBoxLayout(self.scrollAreaWidgetContents_2)
        self.dataLayout.setObjectName(u"dataLayout")
        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_2)

        self.verticalLayout_7.addWidget(self.scrollArea_2)


        self.horizontalLayout_13.addLayout(self.verticalLayout_7)

        self.output = QLabel(self.tab)
        self.output.setObjectName(u"output")

        self.horizontalLayout_13.addWidget(self.output)

        self.tabWidget.addTab(self.tab, "")
        self.Analysis = QWidget()
        self.Analysis.setObjectName(u"Analysis")
        self.Analysis.setEnabled(False)
        self.horizontalLayout_8 = QHBoxLayout(self.Analysis)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.formLayout_2 = QFormLayout()
        self.formLayout_2.setObjectName(u"formLayout_2")
        self.label_4 = QLabel(self.Analysis)
        self.label_4.setObjectName(u"label_4")

        self.formLayout_2.setWidget(0, QFormLayout.LabelRole, self.label_4)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.batchSize = QSpinBox(self.Analysis)
        self.batchSize.setObjectName(u"batchSize")
        self.batchSize.setMaximum(999)
        self.batchSize.setValue(8)

        self.horizontalLayout_7.addWidget(self.batchSize)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_7.addItem(self.horizontalSpacer_3)


        self.formLayout_2.setLayout(0, QFormLayout.FieldRole, self.horizontalLayout_7)

        self.label_5 = QLabel(self.Analysis)
        self.label_5.setObjectName(u"label_5")

        self.formLayout_2.setWidget(1, QFormLayout.LabelRole, self.label_5)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.learningRate = QDoubleSpinBox(self.Analysis)
        self.learningRate.setObjectName(u"learningRate")
        self.learningRate.setDecimals(3)
        self.learningRate.setSingleStep(0.001000000000000)
        self.learningRate.setValue(0.005000000000000)

        self.horizontalLayout_6.addWidget(self.learningRate)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_6.addItem(self.horizontalSpacer_4)


        self.formLayout_2.setLayout(1, QFormLayout.FieldRole, self.horizontalLayout_6)

        self.label_6 = QLabel(self.Analysis)
        self.label_6.setObjectName(u"label_6")

        self.formLayout_2.setWidget(2, QFormLayout.LabelRole, self.label_6)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.numIter = QSpinBox(self.Analysis)
        self.numIter.setObjectName(u"numIter")
        self.numIter.setMaximum(999999)
        self.numIter.setSingleStep(5000)
        self.numIter.setValue(20000)

        self.horizontalLayout_5.addWidget(self.numIter)

        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_5)


        self.formLayout_2.setLayout(2, QFormLayout.FieldRole, self.horizontalLayout_5)


        self.verticalLayout_3.addLayout(self.formLayout_2)

        self.pushButton_3 = QPushButton(self.Analysis)
        self.pushButton_3.setObjectName(u"pushButton_3")

        self.verticalLayout_3.addWidget(self.pushButton_3)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_3.addItem(self.verticalSpacer_2)

        self.progressBar = QProgressBar(self.Analysis)
        self.progressBar.setObjectName(u"progressBar")
        self.progressBar.setValue(0)

        self.verticalLayout_3.addWidget(self.progressBar)


        self.horizontalLayout_8.addLayout(self.verticalLayout_3)

        self.tabWidget.addTab(self.Analysis, "")
        self.Parameters = QWidget()
        self.Parameters.setObjectName(u"Parameters")
        self.Parameters.setEnabled(True)
        self.horizontalLayout_11 = QHBoxLayout(self.Parameters)
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.verticalLayout_4 = QVBoxLayout()
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.horizontalLayout_10 = QHBoxLayout()
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.label_7 = QLabel(self.Parameters)
        self.label_7.setObjectName(u"label_7")

        self.horizontalLayout_10.addWidget(self.label_7)

        self.paramPath = QLineEdit(self.Parameters)
        self.paramPath.setObjectName(u"paramPath")

        self.horizontalLayout_10.addWidget(self.paramPath)

        self.pushButton_4 = QPushButton(self.Parameters)
        self.pushButton_4.setObjectName(u"pushButton_4")

        self.horizontalLayout_10.addWidget(self.pushButton_4)


        self.verticalLayout_4.addLayout(self.horizontalLayout_10)

        self.pushButton_5 = QPushButton(self.Parameters)
        self.pushButton_5.setObjectName(u"pushButton_5")

        self.verticalLayout_4.addWidget(self.pushButton_5)

        self.line = QFrame(self.Parameters)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.verticalLayout_4.addWidget(self.line)

        self.aoiSlider = QSlider(self.Parameters)
        self.aoiSlider.setObjectName(u"aoiSlider")
        self.aoiSlider.setTracking(False)
        self.aoiSlider.setOrientation(Qt.Horizontal)

        self.verticalLayout_4.addWidget(self.aoiSlider)

        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.horizontalSpacer_6 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_9.addItem(self.horizontalSpacer_6)

        self.label_9 = QLabel(self.Parameters)
        self.label_9.setObjectName(u"label_9")

        self.horizontalLayout_9.addWidget(self.label_9)

        self.aoiNumber = QSpinBox(self.Parameters)
        self.aoiNumber.setObjectName(u"aoiNumber")

        self.horizontalLayout_9.addWidget(self.aoiNumber)

        self.aoiMax = QLabel(self.Parameters)
        self.aoiMax.setObjectName(u"aoiMax")

        self.horizontalLayout_9.addWidget(self.aoiMax)

        self.horizontalSpacer_7 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_9.addItem(self.horizontalSpacer_7)


        self.verticalLayout_4.addLayout(self.horizontalLayout_9)

        self.scrollArea = QScrollArea(self.Parameters)
        self.scrollArea.setObjectName(u"scrollArea")
        sizePolicy.setHeightForWidth(self.scrollArea.sizePolicy().hasHeightForWidth())
        self.scrollArea.setSizePolicy(sizePolicy)
        self.scrollArea.setWidgetResizable(True)
        self.scrollWidget = QWidget()
        self.scrollWidget.setObjectName(u"scrollWidget")
        self.scrollWidget.setGeometry(QRect(0, 0, 756, 367))
        self.scrollContent = QHBoxLayout(self.scrollWidget)
        self.scrollContent.setObjectName(u"scrollContent")
        self.scrollArea.setWidget(self.scrollWidget)

        self.verticalLayout_4.addWidget(self.scrollArea)


        self.horizontalLayout_11.addLayout(self.verticalLayout_4)

        self.tabWidget.addTab(self.Parameters, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.horizontalLayout_4 = QHBoxLayout(self.tab_2)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.verticalLayout_5 = QVBoxLayout()
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.pushButton_6 = QPushButton(self.tab_2)
        self.pushButton_6.setObjectName(u"pushButton_6")

        self.verticalLayout_5.addWidget(self.pushButton_6)

        self.scrollArea_3 = QScrollArea(self.tab_2)
        self.scrollArea_3.setObjectName(u"scrollArea_3")
        self.scrollArea_3.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 756, 460))
        self.imagesLayout = QVBoxLayout(self.scrollAreaWidgetContents)
        self.imagesLayout.setObjectName(u"imagesLayout")
        self.scrollArea_3.setWidget(self.scrollAreaWidgetContents)

        self.verticalLayout_5.addWidget(self.scrollArea_3)


        self.horizontalLayout_4.addLayout(self.verticalLayout_5)

        self.tabWidget.addTab(self.tab_2, "")

        self.verticalLayout.addWidget(self.tabWidget)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 800, 20))
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName(u"menuFile")
        self.menuHelp = QMenu(self.menubar)
        self.menuHelp.setObjectName(u"menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)
        QWidget.setTabOrder(self.pathName, self.pushButton)
        QWidget.setTabOrder(self.pushButton, self.deviceName)
        QWidget.setTabOrder(self.deviceName, self.controlBox)
        QWidget.setTabOrder(self.controlBox, self.pushButton_2)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        self.pushButton.clicked.connect(MainWindow.browseSlot)
        self.pushButton_2.clicked.connect(MainWindow.loadDataSlot)
        self.pushButton_3.clicked.connect(MainWindow.runAnalysisSlot)
        self.pushButton_4.clicked.connect(MainWindow.browseParamSlot)
        self.pushButton_5.clicked.connect(MainWindow.loadParamSlot)
        self.aoiSlider.valueChanged.connect(MainWindow.plotParamSlot)
        self.aoiNumber.valueChanged.connect(MainWindow.plotParamSlot)

        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Device", None))
        self.deviceName.setItemText(0, QCoreApplication.translate("MainWindow", u"cpu", None))
        self.deviceName.setItemText(1, QCoreApplication.translate("MainWindow", u"cuda:0", None))
        self.deviceName.setItemText(2, QCoreApplication.translate("MainWindow", u"cuda:1", None))

        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Control", None))
        self.controlBox.setText("")
        self.label.setText(QCoreApplication.translate("MainWindow", u"Folder Name", None))
        self.pushButton.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.pushButton_2.setText(QCoreApplication.translate("MainWindow", u"Load Data", None))
        self.output.setText("")
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QCoreApplication.translate("MainWindow", u"Data", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Batch size", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Learning rate", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"Num. of iterations", None))
        self.pushButton_3.setText(QCoreApplication.translate("MainWindow", u"Run Analysis", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Analysis), QCoreApplication.translate("MainWindow", u"Analysis", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"Folder Name", None))
        self.pushButton_4.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.pushButton_5.setText(QCoreApplication.translate("MainWindow", u"Load Parameters", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"AoI Number", None))
        self.aoiMax.setText(QCoreApplication.translate("MainWindow", u"/0", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Parameters), QCoreApplication.translate("MainWindow", u"Parameters", None))
        self.pushButton_6.setText(QCoreApplication.translate("MainWindow", u"PushButton", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), QCoreApplication.translate("MainWindow", u"Images", None))
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
        self.menuHelp.setTitle(QCoreApplication.translate("MainWindow", u"Help", None))
    # retranslateUi

