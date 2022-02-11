#!/usr/bin/python3
# -*- coding: utf-8 -*-

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPalette, QPainter
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtWidgets import ( QLabel, 
                              QSizePolicy, QScrollArea, 
                              QMessageBox, QMainWindow, 
                              QMenu, QAction,
                              qApp, QFileDialog, 
                              QWidget, QHBoxLayout )

import argparse
import os

API_description = """
***** Radiomics Analysis Platform  *****
API Name: Tumor Displayer
Version:    1.0
Developer: Alvin Li
Email:     alvinli@gorilla-technology.com / d05548014@ntu.edu.tw
****************************************

"""

parser = argparse.ArgumentParser(prog='tumor_display.py',
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=API_description)

parser.add_argument('-Left', 
                    action = 'store', 
                    type = str, 
                    help = 'The absolute path to the image being showed in the left side.')

parser.add_argument('-Right', 
                    action = 'store', 
                    type = str, 
                    help = 'The absolute path to the image being showed in the left side.')

args = parser.parse_args()

class QImageViewSync(QWidget):

    def __init__(self, window=None):
        super().__init__()

        self.window = window
        self.printer = QPrinter()
        self.scaleFactor = 0.0

        self.imageLabelLeft = QLabel()
        self.imageLabelLeft.setBackgroundRole(QPalette.Base)
        self.imageLabelLeft.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabelLeft.setScaledContents(True)

        self.scrollAreaLeft = QScrollArea()
        self.scrollAreaLeft.setBackgroundRole(QPalette.Dark)
        self.scrollAreaLeft.setWidget(self.imageLabelLeft)
        self.scrollAreaLeft.setVisible(False)

        self.imageLabelRight = QLabel()
        self.imageLabelRight.setBackgroundRole(QPalette.Base)
        self.imageLabelRight.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabelRight.setScaledContents(True)

        self.scrollAreaRight = QScrollArea()
        self.scrollAreaRight.setBackgroundRole(QPalette.Dark)
        self.scrollAreaRight.setWidget(self.imageLabelRight)
        self.scrollAreaRight.setVisible(False)

        self.centralWidget = QWidget()
        self.layout = QHBoxLayout(self.centralWidget)
        self.layout.addWidget(self.scrollAreaLeft)
        self.layout.addWidget(self.scrollAreaRight)

        self.scrollAreaLeft.verticalScrollBar().valueChanged.connect(self.scrollAreaRight.verticalScrollBar().setValue)
        self.scrollAreaLeft.horizontalScrollBar().valueChanged.connect(self.scrollAreaRight.horizontalScrollBar().setValue)
        self.scrollAreaRight.verticalScrollBar().valueChanged.connect(self.scrollAreaLeft.verticalScrollBar().setValue)
        self.scrollAreaRight.horizontalScrollBar().valueChanged.connect(self.scrollAreaLeft.horizontalScrollBar().setValue)

        self.scrollAreaLeft.mouseMoveEvent = self.mouseMoveEventLeft
        self.scrollAreaLeft.mousePressEvent = self.mousePressEventLeft
        self.scrollAreaLeft.mouseReleaseEvent = self.mouseReleaseEventLeft

        self.scrollAreaRight.mouseMoveEvent = self.mouseMoveEventRight
        self.scrollAreaRight.mousePressEvent = self.mousePressEventRight
        self.scrollAreaRight.mouseReleaseEvent = self.mouseReleaseEventRight

        self.imageLabelLeft.setCursor(Qt.OpenHandCursor)
        self.imageLabelRight.setCursor(Qt.OpenHandCursor)

    def mousePressEventLeft(self, event):
        self.pressed = True
        self.imageLabelLeft.setCursor(Qt.ClosedHandCursor)
        self.initialPosX = self.scrollAreaLeft.horizontalScrollBar().value() + event.pos().x()
        self.initialPosY = self.scrollAreaLeft.verticalScrollBar().value() + event.pos().y()

    def mouseReleaseEventLeft(self, event):
        self.pressed = False
        self.imageLabelLeft.setCursor(Qt.OpenHandCursor)
        self.initialPosX = self.scrollAreaLeft.horizontalScrollBar().value()
        self.initialPosY = self.scrollAreaLeft.verticalScrollBar().value()

    def mouseMoveEventLeft(self, event):
        if self.pressed:
            self.scrollAreaLeft.horizontalScrollBar().setValue(self.initialPosX - event.pos().x())
            self.scrollAreaLeft.verticalScrollBar().setValue(self.initialPosY - event.pos().y())

    def mousePressEventRight(self, event):
        self.pressed = True
        self.imageLabelRight.setCursor(Qt.ClosedHandCursor)
        self.initialPosX = self.scrollAreaRight.horizontalScrollBar().value() + event.pos().x()
        self.initialPosY = self.scrollAreaRight.verticalScrollBar().value() + event.pos().y()

    def mouseReleaseEventRight(self, event):
        self.pressed = False
        self.imageLabelRight.setCursor(Qt.OpenHandCursor)
        self.initialPosX = self.scrollAreaRight.horizontalScrollBar().value()
        self.initialPosY = self.scrollAreaRight.verticalScrollBar().value()

    def mouseMoveEventRight(self, event):
        if self.pressed:
            self.scrollAreaRight.horizontalScrollBar().setValue(self.initialPosX - event.pos().x())
            self.scrollAreaRight.verticalScrollBar().setValue(self.initialPosY - event.pos().y())

    def open(self):
        options = QFileDialog.Options()
        # fileName = QFileDialog.getOpenFileName(self, "Open File", QDir.currentPath())
        fileName, _ = QFileDialog.getOpenFileName(self, 'QFileDialog.getOpenFileName()', '',
                                                  'Images (*.tif *.png *.jpeg *.jpg *.bmp *.gif)', options=options)
        if fileName:
            print(fileName)
            image = QImage(fileName)
            if image.isNull():
                QMessageBox.information(self, "Image Viewer", "Cannot load %s." % fileName)
                return

            self.imageLabelLeft.setPixmap(QPixmap.fromImage(image))
            self.imageLabelRight.setPixmap(QPixmap.fromImage(image))
            self.scaleFactor = 1.0

            self.scrollAreaLeft.setVisible(True)
            self.scrollAreaRight.setVisible(True)
            self.window.printLeftAct.setEnabled(True)
            self.window.printRightAct.setEnabled(True)
            self.window.fitToWindowAct.setEnabled(True)
            self.updateActions()

            if not self.window.fitToWindowAct.isChecked():
                self.imageLabelLeft.adjustSize()
                self.imageLabelRight.adjustSize()

    def openLeft(self):
        options = QFileDialog.Options()
        # fileName = QFileDialog.getOpenFileName(self, "Open File", QDir.currentPath())
        # fileName, _ = QFileDialog.getOpenFileName(self, 'QFileDialog.getOpenFileName()', '',
        #                                           'Images (*.tif *.png *.jpeg *.jpg *.bmp *.gif)', options=options)
        
        if args.Left != None:
            fileName = args.Left
        else:
            fileName, _ = QFileDialog.getOpenFileName(self, 'QFileDialog.getOpenFileName()', '',
                                                  'Images (*.tif *.png *.jpeg *.jpg *.bmp *.gif)', options=options)
        
        if os.path.exists(fileName) == False:
            fileName, _ = QFileDialog.getOpenFileName(self, 'QFileDialog.getOpenFileName()', '',
                                                  'Images (*.tif *.png *.jpeg *.jpg *.bmp *.gif)', options=options)
        
        if fileName:
            print(fileName)
            image = QImage(fileName)
            if image.isNull():
                QMessageBox.information(self, "Image Viewer", "Cannot load %s." % fileName)
                return

            self.imageLabelLeft.setPixmap(QPixmap.fromImage(image))
            self.scaleFactor = 1.0

            self.scrollAreaLeft.setVisible(True)
            self.window.printLeftAct.setEnabled(True)
            self.window.fitToWindowAct.setEnabled(True)
            self.updateActions()

            if not self.window.fitToWindowAct.isChecked():
                self.imageLabelLeft.adjustSize()
            
            self._left_empty = False

    def openRight(self):
        options = QFileDialog.Options()
        # # fileName = QFileDialog.getOpenFileName(self, "Open File", QDir.currentPath())
        # fileName, _ = QFileDialog.getOpenFileName(self, 'QFileDialog.getOpenFileName()', '',
        #                                           'Images (*.tif *.png *.jpeg *.jpg *.bmp *.gif)', options=options)
        
        if args.Right != None:
            fileName = args.Right
        else:
            fileName, _ = QFileDialog.getOpenFileName(self, 'QFileDialog.getOpenFileName()', '',
                                                  'Images (*.tif *.png *.jpeg *.jpg *.bmp *.gif)', options=options)
        
        if os.path.exists(fileName) == False:
            fileName, _ = QFileDialog.getOpenFileName(self, 'QFileDialog.getOpenFileName()', '',
                                                  'Images (*.tif *.png *.jpeg *.jpg *.bmp *.gif)', options=options)
        
        if fileName:
            print(fileName)
            image = QImage(fileName)
            if image.isNull():
                QMessageBox.information(self, "Image Viewer", "Cannot load %s." % fileName)
                return

            self.imageLabelRight.setPixmap(QPixmap.fromImage(image))
            self.scaleFactor = 1.0

            self.scrollAreaRight.setVisible(True)
            self.window.printRightAct.setEnabled(True)
            self.window.fitToWindowAct.setEnabled(True)
            self.updateActions()

            if not self.window.fitToWindowAct.isChecked():
                self.imageLabelRight.adjustSize()

            self._right_empty = False

    def printLeft(self):
        dialog = QPrintDialog(self.printer, self)
        if dialog.exec_():
            painter = QPainter(self.printer)
            rect = painter.viewport()
            size = self.imageLabelLeft.pixmap().size()
            size.scale(rect.size(), Qt.KeepAspectRatio)
            painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
            painter.setWindow(self.imageLabelLeft.pixmap().rect())
            painter.drawPixmap(0, 0, self.imageLabelLeft.pixmap())

    def printRight(self):
        dialog = QPrintDialog(self.printer, self)
        if dialog.exec_():
            painter = QPainter(self.printer)
            rect = painter.viewport()
            size = self.imageLabelRight.pixmap().size()
            size.scale(rect.size(), Qt.KeepAspectRatio)
            painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
            painter.setWindow(self.imageLabelRight.pixmap().rect())
            painter.drawPixmap(0, 0, self.imageLabelRight.pixmap())

    def zoomIn(self):
        self.scaleImage(1.25)

    def zoomOut(self):
        self.scaleImage(0.8)

    def normalSize(self):
        self.imageLabelLeft.adjustSize()
        self.imageLabelRight.adjustSize()
        self.scaleFactor = 1.0

    def about(self):
        QMessageBox.about(self, "Image View in the Main Window",
                          "<p>The <b>Image Viewer</b> example shows how to combine "
                          "QLabel and QScrollArea to display an image. QLabel is "
                          "typically used for displaying text, but it can also display "
                          "an image. QScrollArea provides a scrolling view around "
                          "another widget. If the child widget exceeds the size of the "
                          "frame, QScrollArea automatically provides scroll bars.</p>"
                          "<p>The example demonstrates how QLabel's ability to scale "
                          "its contents (QLabel.scaledContents), and QScrollArea's "
                          "ability to automatically resize its contents "
                          "(QScrollArea.widgetResizable), can be used to implement "
                          "zooming and scaling features.</p>"
                          "<p>In addition the example shows how to use QPainter to "
                          "print an image.</p>")

    def updateActions(self):
        self.window.zoomInAct.setEnabled(not self.window.fitToWindowAct.isChecked())
        self.window.zoomOutAct.setEnabled(not self.window.fitToWindowAct.isChecked())
        self.window.normalSizeAct.setEnabled(not self.window.fitToWindowAct.isChecked())

    def scaleImage(self, factor):
        self.scaleFactor *= factor
        self.imageLabelLeft.resize(self.scaleFactor * self.imageLabelLeft.pixmap().size())
        self.imageLabelRight.resize(self.scaleFactor * self.imageLabelRight.pixmap().size())

        self.adjustScrollBar(self.scrollAreaLeft.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.scrollAreaLeft.verticalScrollBar(), factor)
        self.adjustScrollBar(self.scrollAreaRight.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.scrollAreaRight.verticalScrollBar(), factor)

        self.window.zoomInAct.setEnabled(self.scaleFactor < 100.0)
        self.window.zoomOutAct.setEnabled(self.scaleFactor > 0.01)

    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value()
                               + ((factor - 1) * scrollBar.pageStep() / 2)))


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.imageViewSync = QImageViewSync(window=self)
        self.setCentralWidget(self.imageViewSync.centralWidget)

        self.createActions(self.imageViewSync)
        self.createMenus()

        self.setWindowTitle("UTUC Tumor Comparator")
        self.resize(600, 400)

    def fitToWindow(self):
        fitToWindow = self.fitToWindowAct.isChecked()
        self.imageViewSync.scrollAreaLeft.setWidgetResizable(fitToWindow)
        self.imageViewSync.scrollAreaRight.setWidgetResizable(fitToWindow)
        if not fitToWindow:
            self.imageViewSync.normalSize()

        self.imageViewSync.updateActions()

    def createActions(self, view):
        self.openLeftAct = QAction("&Open Left...", self, shortcut="Shift+L", triggered=view.openLeft)
        self.openRightAct = QAction("&Open Right...", self, shortcut="Shift+R", triggered=view.openRight)
        self.printLeftAct = QAction("&Print Left...", self, shortcut="Shift+P", enabled=False, triggered=view.printLeft)
        self.printRightAct = QAction("&Print Right...", self,
                                     shortcut="Shift+R+P", enabled=False, triggered=view.printRight)
        # self.exitAct = QAction("E&xit", self, shortcut="^+Q", triggered=image.close)
        self.zoomInAct = QAction("Zoom &In (25%)", self, shortcut="Shift+I", enabled=False, triggered=view.zoomIn)
        self.zoomOutAct = QAction("Zoom &Out (25%)", self, shortcut="Shift+O", enabled=False, triggered=view.zoomOut)
        self.normalSizeAct = QAction("&Normal Size", self, shortcut="Shift+S", enabled=False, triggered=view.normalSize)
        self.fitToWindowAct = QAction("&Fit to Window", self,
                                      enabled=False, checkable=True, shortcut="Shift+F", triggered=self.fitToWindow)
        self.aboutAct = QAction("&About", self, triggered=view.about)
        self.aboutQtAct = QAction("About &Qt", self, triggered=qApp.aboutQt)

    def createMenus(self):
        self.fileMenu = QMenu("&File", self)
        self.fileMenu.addAction(self.openLeftAct)
        self.fileMenu.addAction(self.openRightAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.printLeftAct)
        self.fileMenu.addAction(self.printRightAct)
        self.fileMenu.addSeparator()
        # self.fileMenu.addAction(self.exitAct)

        self.viewMenu = QMenu("&View", self)
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        self.viewMenu.addAction(self.normalSizeAct)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.fitToWindowAct)

        self.helpMenu = QMenu("&Help", self)
        self.helpMenu.addAction(self.aboutAct)
        self.helpMenu.addAction(self.aboutQtAct)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.helpMenu)


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())





# from PyQt5 import QtCore, QtGui, QtWidgets

# class PhotoViewer(QtWidgets.QGraphicsView):
#     photoClicked = QtCore.pyqtSignal(QtCore.QPoint)

#     def __init__(self, parent):
#         super(PhotoViewer, self).__init__(parent)
#         self._zoom = 0
#         self._empty = True
#         self._scene = QtWidgets.QGraphicsScene(self)
#         self._photo = QtWidgets.QGraphicsPixmapItem()
#         self._scene.addItem(self._photo)
#         self.setScene(self._scene)
#         self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
#         self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
#         self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
#         self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
#         self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
#         self.setFrameShape(QtWidgets.QFrame.NoFrame)

#     def hasPhoto(self):
#         return not self._empty

#     def fitInView(self, scale=True):
#         rect = QtCore.QRectF(self._photo.pixmap().rect())
#         if not rect.isNull():
#             self.setSceneRect(rect)
#             if self.hasPhoto():
#                 unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
#                 self.scale(1 / unity.width(), 1 / unity.height())
#                 viewrect = self.viewport().rect()
#                 scenerect = self.transform().mapRect(rect)
#                 factor = min(viewrect.width() / scenerect.width(),
#                              viewrect.height() / scenerect.height())
#                 self.scale(factor, factor)
#             self._zoom = 0

#     def setPhoto(self, pixmap=None):
#         self._zoom = 0
#         if pixmap and not pixmap.isNull():
#             self._empty = False
#             self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
#             self._photo.setPixmap(pixmap)
#         else:
#             self._empty = True
#             self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
#             self._photo.setPixmap(QtGui.QPixmap())
#         self.fitInView()

#     def wheelEvent(self, event):
#         if self.hasPhoto():
#             if event.angleDelta().y() > 0:
#                 factor = 1.25
#                 self._zoom += 1
#             else:
#                 factor = 0.8
#                 self._zoom -= 1
#             if self._zoom > 0:
#                 self.scale(factor, factor)
#             elif self._zoom == 0:
#                 self.fitInView()
#             else:
#                 self._zoom = 0

#     def toggleDragMode(self):
#         if self.dragMode() == QtWidgets.QGraphicsView.ScrollHandDrag:
#             self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
#         elif not self._photo.pixmap().isNull():
#             self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

#     def mousePressEvent(self, event):
#         if self._photo.isUnderMouse():
#             self.photoClicked.emit(self.mapToScene(event.pos()).toPoint())
#         super(PhotoViewer, self).mousePressEvent(event)


# class Window(QtWidgets.QWidget):
#     def __init__(self):
#         super(Window, self).__init__()
#         self.viewer1 = PhotoViewer(self)
#         self.viewer2 = PhotoViewer(self)
#         # 'Load image' button
#         self.btnLoad = QtWidgets.QToolButton(self)
#         self.btnLoad.setText('Load image')
#         self.btnLoad.clicked.connect(self.loadImage)
#         # Button to change from drag/pan to getting pixel info
#         self.btnPixInfo = QtWidgets.QToolButton(self)
#         self.btnPixInfo.setText('Enter pixel info mode')
#         self.btnPixInfo.clicked.connect(self.pixInfo)
#         self.editPixInfo = QtWidgets.QLineEdit(self)
#         self.editPixInfo.setReadOnly(True)
#         self.viewer1.photoClicked.connect(self.photoClicked)
#         self.viewer2.photoClicked.connect(self.photoClicked)
#         # Arrange layout
#         VBlayout = QtWidgets.QVBoxLayout(self)
#         VBlayout.addWidget(self.viewer1)
#         VBlayout.addWidget(self.viewer2)
        
#         VBlayout.addLayout(VBlayout)
        
#         HBlayout = QtWidgets.QHBoxLayout()
#         HBlayout.setAlignment(QtCore.Qt.AlignLeft)
#         HBlayout.addWidget(self.btnLoad)
#         HBlayout.addWidget(self.btnPixInfo)
#         HBlayout.addWidget(self.editPixInfo)
        
#         VBlayout.addLayout(HBlayout)

#     def loadImage(self):
#         self.viewer1.setPhoto(QtGui.QPixmap('image.jpg'))
#         self.viewer2.setPhoto(QtGui.QPixmap('image.jpg'))

#     def pixInfo(self):
#         self.viewer1.toggleDragMode()
#         self.viewer2.toggleDragMode()

#     def photoClicked(self, pos):
#         if self.viewer1.dragMode()  == QtWidgets.QGraphicsView.NoDrag:
#             self.editPixInfo.setText('%d, %d' % (pos.x(), pos.y()))
#         if self.viewer2.dragMode()  == QtWidgets.QGraphicsView.NoDrag:
#             self.editPixInfo.setText('%d, %d' % (pos.x(), pos.y()))


# if __name__ == '__main__':
#     import sys
#     app = QtWidgets.QApplication(sys.argv)
#     window = Window()
#     window.setGeometry(500, 300, 800, 600)
#     window.show()
#     sys.exit(app.exec_())


# # filename = '/media/share/DATA/Radiomics/UTUC_DATA/Tumor_Seg_VenousPhase/Output/vol/UC102_20180918_3/00000.tif'  # place path to your image here
# # #filename = 'd:/Data/yandex_z18_1-1.tif'  # huge TIFF file 1.4 GB
# # #filename = 'd:/Data/The_Garden_of_Earthly_Delights_by_Bosch_High_Resolution.jpg'
# # #filename = 'd:/Data/The_Garden_of_Earthly_Delights_by_Bosch_High_Resolution.tif'
# # #filename = 'd:/Data/heic1502a.tif'
# # #filename = 'd:/Data/land_shallow_topo_east.tif'
# # #filename = 'd:/Data/X1D5_B0002594.3FR'
# # app = MainWindow(tk.Tk(), path=filename)
# # app.mainloop()
