import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPainter, QPen, QPolygonF, QBrush
import cv2
import numpy as np
from PyQt5.QtWidgets import QGraphicsEllipseItem

class DraggablePoint(QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(parent)
        self.scene = scene
        self.setRenderHint(QPainter.Antialiasing)
        self.setScene(self.scene)
        self.drag_point = None
        self.setDragMode(QGraphicsView.RubberBandDrag)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            items = self.items(event.pos())
            for item in items:
                if isinstance(item, DraggableEllipse):
                    self.drag_point = item
                    break

            if self.drag_point is None:
                self.drag_point = DraggableEllipse(event.pos())
                self.scene.addItem(self.drag_point)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_point = None

    def mouseMoveEvent(self, event):
        if self.drag_point is not None:
            self.drag_point.setPos(self.mapToScene(event.pos()))
            self.parent().updateFill()


class DraggableEllipse(QGraphicsEllipseItem):
    def __init__(self, pos):
        super().__init__(-5, -5, 10, 10)
        self.setPos(pos)
        self.setPen(QPen(Qt.red))
        self.setFlag(self.ItemIsMovable, True)
        self.setFlag(self.ItemSendsGeometryChanges, True)

    def itemChange(self, change, value):
        if change == self.ItemPositionChange and self.scene() is not None:
            scene_pos = self.scene().sceneRect().contains(value)
            if not scene_pos:
                return self.pos()
        return super().itemChange(change, value)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Danger Zone Fill Adjuster")
        self.setGeometry(100, 100, 400, 400)

        self.danger_zone = QPolygonF([
            QPointF(50, 100),
            QPointF(200, 50),
            QPointF(350, 100),
            QPointF(350, 300),
            QPointF(50, 300)
        ])

        self.fill_poly = QPolygonF()
        self.fill_brush = QBrush(Qt.green)

        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(0, 0, 400, 400)
        self.fill_item = self.scene.addPolygon(self.fill_poly, QPen(Qt.NoPen), self.fill_brush)

        self.view = DraggablePoint(self.scene, self)
        self.view.setFixedSize(400, 400)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.setCentralWidget(self.view)

        self.updateFill()

    def updateFill(self):
        self.fill_poly = QPolygonF()
        for item in self.scene.items():
            if isinstance(item, DraggableEllipse):
                pos = item.scenePos()
                self.fill_poly.append(pos)

        self.fill_item.setPolygon(self.fill_poly)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(self.fill_brush.color(), 2.0, Qt.SolidLine, Qt.FlatCap, Qt.MiterJoin))
        painter.setBrush(self.fill_brush)

        painter.drawPolygon(self.fill_poly, Qt.WindingFill)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
