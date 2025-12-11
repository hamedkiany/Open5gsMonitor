#!/usr/bin/env python3
import sys
import re
import requests
from collections import deque

from PyQt5.QtCore import (
    Qt, QTimer, QPointF
)
from PyQt5.QtGui import QPen, QBrush, QColor, QFont
from PyQt5.QtWidgets import (
    QApplication, QGraphicsView, QGraphicsScene, QGraphicsRectItem,
    QGraphicsLineItem, QGraphicsSimpleTextItem, QGraphicsEllipseItem,
    QMainWindow, QWidget, QVBoxLayout, QTextEdit, QSplitter
)

# ================== CONFIG ==================

API_URL = "http://1.ibuild.es:8000/events?limit=500"

SERVICES = [
    "UE1", "GNB",
    "UE2", "GNB2",
    "AMF", "SMF", "UPF", "UDR",
    "NRF", "AUSF", "UDM", "PCF",
    "BSF", "NSSF", "SCP",
    "MME", "SGWC", "SGWU", "HSS", "PCRF",
]

# servicios que queremos en la barra de estado
STATUS_SERVICES = [
    "NRF", "SCP", "UPF", "SMF", "AMF",
    "AUSF", "UDM", "PCF", "NSSF", "BSF", "UDR",
]

# Posición de cada nodo (grafo principal)
NODE_POS = {
    "UE1": (-450,   0),
    "GNB": (-300,   0),
    "UE2": (-450,  60),
    "GNB2": (-300,  60),
    "AMF": (-150,   0),
    "SMF": (   0,   0),
    "UPF": ( 150,   0),
    "UDR": ( 300,   0),

    "UDM": (-300, -140),
    "AUSF":(-150, -140),
    "NRF": (   0, -140),
    "PCF": ( 150, -140),
    "SCP": ( 300, -140),
    "NSSF":( 150, -260),
    "BSF": (   0, -260),

    "HSS": (-300,  160),
    "MME": (-150,  160),
    "SGWC":(   0,  160),
    "SGWU":( 150,  160),
    "PCRF":( 300,  160),
}

POLL_INTERVAL_MS = 2000       # cada 2 s consulta el API
ANIMATION_DURATION_MS = 800   # duración de la bolita
ANIMATION_STEPS = 25          # pasos de la bolita
HIGHLIGHT_MS = 600            # cuánto tiempo está encendido el nodo/línea

# línea con conexión: " UPF(127.0.0.7) -> SMF(127.0.0.4) "
CONN_RE = re.compile(r"\s+([A-Za-z0-9_]+)\([^)]*\)\s*->\s*([A-Za-z0-9_]+)\([^)]*\)")

# colores
NODE_COLOR_BASE = QColor(40, 40, 80)
NODE_COLOR_ACTIVE_SRC = QColor("#1abc9c")   # verde-azulado
NODE_COLOR_ACTIVE_DST = QColor("#e67e22")   # naranja
EDGE_COLOR_BASE = QColor(150, 150, 150)
EDGE_COLOR_ACTIVE = QColor("#f1c40f")       # amarillo

DOT_COLOR = QColor("#2ecc71")               # verde

STATUS_UNKNOWN = QColor(70, 70, 70)
STATUS_UP = QColor("#2ecc71")      # verde
STATUS_DOWN = QColor("#e74c3c")    # rojo

# ================== DOT MANUAL (sin QObject) ==================

class MovingDot(QGraphicsEllipseItem):
    def __init__(self, p1: QPointF, p2: QPointF,
                 scene: QGraphicsScene,
                 duration_ms=ANIMATION_DURATION_MS,
                 steps=ANIMATION_STEPS):
        super().__init__(-5, -5, 10, 10)
        self.setBrush(QBrush(DOT_COLOR))
        self.p1 = p1
        self.p2 = p2
        self.scene = scene
        self.steps = max(steps, 1)
        self.step = 0

        self.timer = QTimer()
        interval = max(1, duration_ms // self.steps)
        self.timer.timeout.connect(self.advance)
        self.timer.start(interval)

        self.update_pos()

    def update_pos(self):
        t = self.step / float(self.steps)
        x = self.p1.x() + (self.p2.x() - self.p1.x()) * t
        y = self.p1.y() + (self.p2.y() - self.p1.y()) * t
        self.setPos(x, y)

    def advance(self):
        if self.step >= self.steps:
            self.timer.stop()
            self.scene.removeItem(self)
            return
        self.step += 1
        self.update_pos()

# ================== VENTANA PRINCIPAL ==================

class NetworkView(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Open5GS Traffic Viewer-Developed By FWS")
        self.resize(1400, 800)

        splitter = QSplitter()
        self.setCentralWidget(splitter)

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHints(self.view.renderHints())
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        right_layout.addWidget(self.log)

        splitter.addWidget(self.view)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        self.node_items = {}    # nombre -> (rect, text) (grafo principal)
        self.node_centers = {}  # nombre -> QPointF
        self.links = {}         # frozenset({src,dst}) -> QGraphicsLineItem

        self.status_items = {}  # nombre -> rectángulo en barra de estado

        self.seen_events = set()
        self.event_buffer = deque(maxlen=4000)

        self.dots = []          # referencias a MovingDot para que no las borre el GC

        self.create_nodes()
        self.create_status_bar()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll_api)
        self.timer.start(POLL_INTERVAL_MS)

    # ---------- nodos del grafo -----------

    def create_nodes(self):
        font = QFont()
        font.setPointSize(10)

        fallback_x = 500
        fallback_y_step = 100
        fallback_i = 0

        for name in SERVICES:
            if name in NODE_POS:
                x, y = NODE_POS[name]
            else:
                x = fallback_x
                y = fallback_i * fallback_y_step
                fallback_i += 1

            w, h = 80, 50
            rect = QGraphicsRectItem(x - w/2, y - h/2, w, h)
            rect.setBrush(QBrush(NODE_COLOR_BASE))
            rect.setPen(QPen(Qt.white, 2))
            self.scene.addItem(rect)

            text = QGraphicsSimpleTextItem(name)
            text.setFont(font)
            tr = text.boundingRect()
            text.setPos(x - tr.width()/2, y - tr.height()/2)
            text.setBrush(QBrush(Qt.white))
            self.scene.addItem(text)

            self.node_items[name] = (rect, text)
            self.node_centers[name] = QPointF(x, y)

        self.view.setSceneRect(-600, -400, 1200, 800)

    # ---------- barra de estado inferior -----------

    def create_status_bar(self):
        font = QFont()
        font.setPointSize(8)

        y = 260  # debajo de todo
        n = len(STATUS_SERVICES)
        spacing = 90
        start_x = - (n - 1) * spacing / 2

        for i, name in enumerate(STATUS_SERVICES):
            x = start_x + i * spacing

            w, h = 60, 30
            rect = QGraphicsRectItem(x - w/2, y - h/2, w, h)
            rect.setBrush(QBrush(STATUS_UNKNOWN))
            rect.setPen(QPen(Qt.white, 1))
            self.scene.addItem(rect)

            text = QGraphicsSimpleTextItem(name)
            text.setFont(font)
            tr = text.boundingRect()
            text.setPos(x - tr.width()/2, y - tr.height()/2)
            text.setBrush(QBrush(Qt.white))
            self.scene.addItem(text)

            self.status_items[name] = rect

    # ---------- API / eventos ----------

    def poll_api(self):
        try:
            resp = requests.get(API_URL, timeout=1.0)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                return
            for line in data:
                self.process_event_line(str(line))
        except Exception as e:
            self.append_log(f"[ERROR API] {e}")

    def append_log(self, text: str):
        self.log.append(text)
        self.log.moveCursor(self.log.textCursor().End)

    def process_event_line(self, line: str):
        # primero, manejar líneas de estado de servicios
        if line.startswith("[SERVICE]"):
            self.update_service_status(line)
            return

        if line in self.seen_events:
            return
        self.seen_events.add(line)
        self.event_buffer.append(line)

        self.append_log(line)

        m = CONN_RE.search(line)
        if not m:
            return
        src, dst = m.group(1), m.group(2)

        if src not in self.node_centers or dst not in self.node_centers:
            # Si aparece un nombre que no tenemos en el grafo, lo ignoramos
            return

        key = frozenset({src, dst})

        # crear línea si no existe aún
        if key not in self.links:
            p1 = self.node_centers[src]
            p2 = self.node_centers[dst]
            line_item = QGraphicsLineItem(p1.x(), p1.y(), p2.x(), p2.y())
            line_item.setPen(QPen(EDGE_COLOR_BASE, 1.5, Qt.SolidLine))
            self.scene.addItem(line_item)
            self.links[key] = line_item

        # resaltar nodos y arista
        self.highlight_node(src, NODE_COLOR_ACTIVE_SRC)
        self.highlight_node(dst, NODE_COLOR_ACTIVE_DST)
        self.highlight_edge(key, EDGE_COLOR_ACTIVE)

        # animar bolita
        self.animate_dot(src, dst)

    # ---------- actualizar barra de estado ----------

    def update_service_status(self, line: str):
        """
        Línea tipo:
        [SERVICE] AMF=UP SMF=UP UPF=DOWN ...
        """
        try:
            parts = line.strip().split()[1:]  # quitamos "[SERVICE]"
            for p in parts:
                if "=" not in p:
                    continue
                name, state = p.split("=", 1)
                name = name.strip()
                state = state.strip().upper()
                rect = self.status_items.get(name)
                if not rect:
                    continue
                if state == "UP":
                    rect.setBrush(QBrush(STATUS_UP))
                elif state == "DOWN":
                    rect.setBrush(QBrush(STATUS_DOWN))
                else:
                    rect.setBrush(QBrush(STATUS_UNKNOWN))
        except Exception as e:
            self.append_log(f"[ERROR STATUS PARSE] {e}")

    # ---------- highlight / animaciones ----------

    def highlight_node(self, name, color):
        if name not in self.node_items:
            return
        rect, _ = self.node_items[name]
        original = rect.brush().color()
        rect.setBrush(QBrush(color))

        QTimer.singleShot(
            HIGHLIGHT_MS,
            lambda n=name, c=original: self.reset_node_color(n, c)
        )

    def reset_node_color(self, name, color):
        if name not in self.node_items:
            return
        rect, _ = self.node_items[name]
        rect.setBrush(QBrush(color))

    def highlight_edge(self, key, color):
        if key not in self.links:
            return
        line_item = self.links[key]
        original_pen = line_item.pen()
        original_color = original_pen.color()
        original_width = original_pen.widthF()

        pen = line_item.pen()
        pen.setColor(color)
        pen.setWidthF(2.5)
        line_item.setPen(pen)

        QTimer.singleShot(
            HIGHLIGHT_MS,
            lambda k=key, c=original_color, w=original_width:
                self.reset_edge_color(k, c, w)
        )

    def reset_edge_color(self, key, color, width):
        if key not in self.links:
            return
        line_item = self.links[key]
        pen = line_item.pen()
        pen.setColor(color)
        pen.setWidthF(width)
        line_item.setPen(pen)

    def animate_dot(self, src, dst):
        p1 = self.node_centers[src]
        p2 = self.node_centers[dst]

        dot = MovingDot(p1, p2, self.scene)
        self.scene.addItem(dot)

        # guardamos referencia para que el recolector no lo borre demasiado pronto
        self.dots.append(dot)
        # limpieza básica: eliminar referencias a dots que ya no tienen timer activo
        self.dots = [d for d in self.dots if d.timer.isActive()]

# ================== MAIN ==================

def main():
    app = QApplication(sys.argv)
    w = NetworkView()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
