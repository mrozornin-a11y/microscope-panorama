"""Desktop GUI (PySide6)."""

from __future__ import annotations

import os
import sys
import traceback
from dataclasses import fields

import cv2
import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QDesktopServices, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsPixmapItem, QGraphicsScene,
    QGraphicsView, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSpinBox, QSplitter,
    QTabWidget, QToolButton, QVBoxLayout, QWidget,
)

from .config import Settings
from .fov import FieldOfView, detect_fov
from .pipeline import Pipeline
from .video import VideoSource


def to_pixmap(img_bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


# --------------------------------------------------------------------------
class ImageView(QGraphicsView):
    """Zoomable / pannable image view (wheel = zoom, drag = pan)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.pix = QGraphicsPixmapItem()
        self.pix.setTransformationMode(Qt.SmoothTransformation)
        self.scene().addItem(self.pix)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor(40, 40, 40)))

    def set_image(self, img_bgr: np.ndarray, fit: bool = True):
        self.pix.setPixmap(to_pixmap(img_bgr))
        self.scene().setSceneRect(QRectF(self.pix.pixmap().rect()))
        if fit:
            self.fit()

    def fit(self):
        if not self.pix.pixmap().isNull():
            self.fitInView(self.pix, Qt.KeepAspectRatio)

    def wheelEvent(self, ev):
        f = 1.25 if ev.angleDelta().y() > 0 else 0.8
        self.scale(f, f)


class CircleView(ImageView):
    """First frame with the field-of-view circle.  Ctrl/Shift + left drag
    moves the centre, right drag changes the radius."""

    circleChanged = Signal(float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        pen = QPen(QColor(255, 220, 0), 2)
        pen.setCosmetic(True)
        self.circle = QGraphicsEllipseItem()
        self.circle.setPen(pen)
        pen2 = QPen(QColor(0, 220, 0), 1, Qt.DashLine)
        pen2.setCosmetic(True)
        self.inner = QGraphicsEllipseItem()
        self.inner.setPen(pen2)
        pen3 = QPen(QColor(255, 60, 60), 2)
        pen3.setCosmetic(True)
        self.cross = [QGraphicsLineItem(), QGraphicsLineItem()]
        for it in [self.circle, self.inner] + self.cross:
            if isinstance(it, QGraphicsLineItem):
                it.setPen(pen3)
            self.scene().addItem(it)
            it.setVisible(False)
        self.cx = self.cy = self.r = 0.0
        self.margin = 0.04
        self._mode = None

    def set_circle(self, cx, cy, r, margin=None, emit=False):
        self.cx, self.cy, self.r = cx, cy, r
        if margin is not None:
            self.margin = margin
        self.circle.setRect(QRectF(cx - r, cy - r, 2 * r, 2 * r))
        ri = r * (1 - self.margin)
        self.inner.setRect(QRectF(cx - ri, cy - ri, 2 * ri, 2 * ri))
        s = max(6.0, r * 0.05)
        self.cross[0].setLine(cx - s, cy, cx + s, cy)
        self.cross[1].setLine(cx, cy - s, cx, cy + s)
        for it in [self.circle, self.inner] + self.cross:
            it.setVisible(True)
        if emit:
            self.circleChanged.emit(cx, cy, r)

    def mousePressEvent(self, ev):
        mods = ev.modifiers()
        if ev.button() == Qt.RightButton:
            self._mode = "radius"
        elif ev.button() == Qt.LeftButton and mods & (Qt.ControlModifier | Qt.ShiftModifier):
            self._mode = "move"
            self._start = self.mapToScene(ev.position().toPoint())
            self._c0 = (self.cx, self.cy)
        else:
            self._mode = None
            return super().mousePressEvent(ev)
        self._apply(ev)

    def mouseMoveEvent(self, ev):
        if self._mode:
            self._apply(ev)
        else:
            super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._mode:
            self._mode = None
        else:
            super().mouseReleaseEvent(ev)

    def _apply(self, ev):
        p = self.mapToScene(ev.position().toPoint())
        if self._mode == "radius":
            r = float(np.hypot(p.x() - self.cx, p.y() - self.cy))
            self.set_circle(self.cx, self.cy, max(r, 10.0), emit=True)
        elif self._mode == "move":
            dx, dy = p.x() - self._start.x(), p.y() - self._start.y()
            self.set_circle(self._c0[0] + dx, self._c0[1] + dy, self.r, emit=True)


# --------------------------------------------------------------------------
class Worker(QObject):
    progress = Signal(float, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, video, out_dir, settings, fov):
        super().__init__()
        self.args = (video, out_dir, settings, fov)
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        video, out_dir, settings, fov = self.args
        try:
            res = Pipeline(video, out_dir, settings, fov,
                           progress=lambda f, m: self.progress.emit(f, m),
                           cancel=lambda: self._cancel).run()
            self.finished.emit(res)
        except InterruptedError:
            self.failed.emit("Cancelled")
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


# --------------------------------------------------------------------------
# label, tooltip, (min, max, step, decimals) for numeric settings
_MAIN = [
    ("min_sharpness", "Min sharpness (relative)",
     "Frame is skipped as blurred if its sharpness is below this fraction of the "
     "median sharpness of neighbouring frames.", (0.0, 1.0, 0.05, 2)),
    ("keyframe_min_shift", "Min shift for keyframe",
     "New keyframe when the stage moved by this fraction of the field diameter.",
     (0.02, 0.8, 0.02, 2)),
    ("expected_overlap", "Expected overlap of passes",
     "Overlap between neighbouring passes of the snake path (fraction of diameter).",
     (0.05, 0.9, 0.05, 2)),
    ("max_rotation_deg", "Max rotation, deg",
     "0 = pure translation (recommended). Allow small rotation if the camera "
     "or the stage rotates slightly.", (0.0, 10.0, 0.5, 1)),
    ("analysis_fps", "Analysed frames per second",
     "How many video frames per second are analysed.", (1.0, 60.0, 1.0, 1)),
]
_ADVANCED = [
    ("mask_margin", "Mask margin", "Fraction of the radius trimmed at the circle edge.",
     (0.0, 0.3, 0.01, 2)),
    ("min_quality", "Min registration quality",
     "Minimal correlation after fine alignment; less reliable matches are discarded.",
     (0.1, 0.99, 0.05, 2)),
    ("min_ncc", "Min coarse correlation", "", (0.05, 0.9, 0.05, 2)),
    ("max_peak_ratio", "Max ambiguity ratio",
     "Second-best / best correlation peak; protects against repetitive texture.",
     (0.5, 1.0, 0.02, 2)),
    ("min_overlap_area", "Min overlap area", "", (0.02, 0.8, 0.02, 2)),
    ("work_diameter", "Working diameter, px", "", (128, 4096, 64, 0)),
    ("coarse_diameter", "Coarse diameter, px", "", (64, 1024, 32, 0)),
    ("max_neighbours", "Cross-pass candidates / frame", "", (0, 50, 1, 0)),
    ("outlier_threshold", "Outlier threshold",
     "Fraction of the diameter; inconsistent matches above it are removed.",
     (0.001, 0.1, 0.001, 3)),
    ("lost_patience", "Lost-tracking patience, frames", "", (1, 50, 1, 0)),
    ("blend_levels", "Multiband levels", "", (1, 9, 1, 0)),
    ("feather_power", "Feather power", "", (0.5, 6.0, 0.25, 2)),
    ("tile_size", "Render tile, px", "", (512, 8192, 256, 0)),
    ("preview_max_size", "Preview size, px", "", (256, 10000, 100, 0)),
    ("threads", "Threads (0 = auto)", "", (0, 128, 1, 0)),
]


class SettingsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.widgets = {}
        d = Settings()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        main = QGroupBox("Settings")
        fm = QFormLayout(main)
        for name, label, tip, rng in _MAIN:
            fm.addRow(label, self._num(name, getattr(d, name), tip, rng))
        self.blending = QComboBox()
        self.blending.addItems(["Fast (feather)", "High (multiband)", "None (hard seams)"])
        self.blending.setCurrentIndex({"feather": 0, "multiband": 1, "none": 2}[d.blending])
        self.blending.setToolTip("Blending quality in overlapping regions.")
        fm.addRow("Blending quality", self.blending)
        lay.addWidget(main)

        self.toggle = QToolButton()
        self.toggle.setText("Advanced settings")
        self.toggle.setCheckable(True)
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.RightArrow)
        self.toggle.toggled.connect(self._toggle)
        lay.addWidget(self.toggle)
        self.adv = QGroupBox()
        fa = QFormLayout(self.adv)
        for name, label, tip, rng in _ADVANCED:
            fa.addRow(label, self._num(name, getattr(d, name), tip, rng))
        self.flat = QCheckBox("Flat-field (vignetting) correction")
        self.flat.setChecked(d.flat_field)
        self.loop = QCheckBox("Cross-pass matching (loop closure)")
        self.loop.setChecked(d.loop_closure)
        self.interp = QComboBox()
        self.interp.addItems(["linear", "cubic", "lanczos"])
        self.interp.setCurrentText(d.interpolation)
        self.comp = QComboBox()
        self.comp.addItems(["zlib", "none"])
        fa.addRow(self.flat)
        fa.addRow(self.loop)
        fa.addRow("Interpolation", self.interp)
        fa.addRow("TIFF compression", self.comp)
        self.adv.setVisible(False)
        lay.addWidget(self.adv)
        lay.addStretch(1)

    def _num(self, name, value, tip, rng):
        lo, hi, step, dec = rng
        if dec == 0:
            w = QSpinBox()
            w.setRange(int(lo), int(hi))
            w.setSingleStep(int(step))
            w.setValue(int(value))
        else:
            w = QDoubleSpinBox()
            w.setDecimals(dec)
            w.setRange(lo, hi)
            w.setSingleStep(step)
            w.setValue(float(value))
        if tip:
            w.setToolTip(tip)
        self.widgets[name] = w
        return w

    def _toggle(self, on):
        self.adv.setVisible(on)
        self.toggle.setArrowType(Qt.DownArrow if on else Qt.RightArrow)

    def settings(self) -> Settings:
        st = Settings()
        types = {f.name: type(f.default) for f in fields(Settings)}
        for name, w in self.widgets.items():
            setattr(st, name, types[name](w.value()))
        st.blending = ["feather", "multiband", "none"][self.blending.currentIndex()]
        st.flat_field = self.flat.isChecked()
        st.loop_closure = self.loop.isChecked()
        st.interpolation = self.interp.currentText()
        st.tiff_compression = self.comp.currentText()
        return st


# --------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Microscope Panorama")
        self.resize(1300, 820)
        self.video: VideoSource | None = None
        self.first = None
        self.thread = None
        self.worker = None
        self.result = None

        # left: controls
        left = QWidget()
        ll = QVBoxLayout(left)
        self.btn_open = QPushButton("Open video…")
        self.btn_open.clicked.connect(self.open_video)
        ll.addWidget(self.btn_open)
        self.lbl_video = QLabel("No video loaded")
        self.lbl_video.setWordWrap(True)
        ll.addWidget(self.lbl_video)

        grp = QGroupBox("Field of view (circle)")
        fc = QFormLayout(grp)
        self.sp_cx, self.sp_cy, self.sp_r = QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox()
        for sp in (self.sp_cx, self.sp_cy, self.sp_r):
            sp.setRange(-100000, 100000)
            sp.setDecimals(1)
            sp.valueChanged.connect(self._spin_changed)
        fc.addRow("Centre X", self.sp_cx)
        fc.addRow("Centre Y", self.sp_cy)
        fc.addRow("Radius", self.sp_r)
        self.btn_detect = QPushButton("Auto-detect")
        self.btn_detect.clicked.connect(self.detect)
        fc.addRow(self.btn_detect)
        hint = QLabel("Shift+drag: move centre · right drag: radius · wheel: zoom")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray")
        fc.addRow(hint)
        ll.addWidget(grp)

        self.panel = SettingsPanel()
        self.panel.widgets["mask_margin"].valueChanged.connect(self._margin_changed)
        ll.addWidget(self.panel)

        scroll = QScrollArea()
        scroll.setWidget(left)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(360)

        # bottom of the left column (always visible): output, build, progress
        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        out_row = QHBoxLayout()
        self.ed_out = QLineEdit()
        self.ed_out.setPlaceholderText("Output folder")
        b = QPushButton("…")
        b.setFixedWidth(30)
        b.clicked.connect(self.choose_out)
        out_row.addWidget(self.ed_out)
        out_row.addWidget(b)
        bl.addLayout(out_row)

        self.btn_build = QPushButton("Build mosaic")
        self.btn_build.setMinimumHeight(36)
        f = self.btn_build.font()
        f.setBold(True)
        self.btn_build.setFont(f)
        self.btn_build.setEnabled(False)
        self.btn_build.clicked.connect(self.build)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel)
        row = QHBoxLayout()
        row.addWidget(self.btn_build, 3)
        row.addWidget(self.btn_cancel, 1)
        bl.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(True)
        self.progress.setFormat("%p%")
        bl.addWidget(self.progress)
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        bl.addWidget(self.lbl_status)

        side = QWidget()
        sl = QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addWidget(scroll, 1)
        sl.addWidget(bottom)

        # right: tabs
        self.tabs = QTabWidget()
        self.view_frame = CircleView()
        self.view_frame.circleChanged.connect(self._view_changed)
        self.view_result = ImageView()
        res_w = QWidget()
        rl = QVBoxLayout(res_w)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.view_result)
        rrow = QHBoxLayout()
        self.lbl_result = QLabel("")
        self.btn_folder = QPushButton("Open output folder")
        self.btn_folder.setEnabled(False)
        self.btn_folder.clicked.connect(self.open_folder)
        rrow.addWidget(self.lbl_result, 1)
        rrow.addWidget(self.btn_folder)
        rl.addLayout(rrow)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.tabs.addTab(self.view_frame, "First frame")
        self.tabs.addTab(res_w, "Result")
        self.tabs.addTab(self.log, "Log")

        split = QSplitter()
        split.addWidget(side)
        split.addWidget(self.tabs)
        split.setStretchFactor(1, 1)
        split.setSizes([380, 920])
        self.setCentralWidget(split)

        act = QAction("Open video…", self)
        act.setShortcut("Ctrl+O")
        act.triggered.connect(self.open_video)
        self.addAction(act)
        self._updating = False

    # ---------------------------------------------------------------- video
    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open microscope video", "",
            "Video (*.mov *.mp4 *.m4v *.avi *.mkv);;All files (*)")
        if path:
            self.load_video(path)

    def load_video(self, path: str):
        try:
            self.video = VideoSource(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Error", str(e))
            return
        v = self.video
        self.first = v.first_frame()
        self.lbl_video.setText(f"{os.path.basename(path)}\n{v.width}×{v.height}, "
                               f"{v.fps:.2f} fps, {v.duration:.1f} s")
        self.ed_out.setText(os.path.splitext(path)[0] + "_mosaic")
        self.view_frame.set_image(self.first)
        self.tabs.setCurrentIndex(0)
        self.detect()
        self.btn_build.setEnabled(True)

    def detect(self):
        if self.video is None:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            fov = detect_fov(self.video.sample_frames(12), self.panel.widgets["mask_margin"].value())
        finally:
            QApplication.restoreOverrideCursor()
        self._set_circle(fov.cx, fov.cy, fov.radius)

    def _set_circle(self, cx, cy, r):
        self._updating = True
        self.sp_cx.setValue(cx)
        self.sp_cy.setValue(cy)
        self.sp_r.setValue(r)
        self._updating = False
        self.view_frame.set_circle(cx, cy, r, self.panel.widgets["mask_margin"].value())

    def _spin_changed(self):
        if not self._updating:
            self.view_frame.set_circle(self.sp_cx.value(), self.sp_cy.value(), self.sp_r.value())

    def _view_changed(self, cx, cy, r):
        self._updating = True
        self.sp_cx.setValue(cx)
        self.sp_cy.setValue(cy)
        self.sp_r.setValue(r)
        self._updating = False

    def _margin_changed(self, m):
        if self.video is not None:
            self.view_frame.set_circle(self.sp_cx.value(), self.sp_cy.value(),
                                       self.sp_r.value(), m)

    def choose_out(self):
        d = QFileDialog.getExistingDirectory(self, "Output folder", self.ed_out.text())
        if d:
            self.ed_out.setText(d)

    # ---------------------------------------------------------------- build
    def build(self):
        if self.video is None:
            return
        st = self.panel.settings()
        fov = FieldOfView(self.sp_cx.value(), self.sp_cy.value(), self.sp_r.value(),
                          self.video.width, self.video.height, st.mask_margin)
        out = self.ed_out.text().strip() or os.path.splitext(self.video.path)[0] + "_mosaic"
        self.log.clear()
        self.progress.setValue(0)
        self._busy(True)
        self.thread = QThread()
        self.worker = Worker(self.video.path, out, st, fov)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.start()

    def cancel(self):
        if self.worker:
            self.worker.cancel()
            self.lbl_status.setText("Cancelling…")

    def _busy(self, on):
        for w in (self.btn_build, self.btn_open, self.btn_detect, self.panel):
            w.setEnabled(not on)
        self.btn_cancel.setEnabled(on)

    def _on_progress(self, f, msg):
        self.progress.setValue(int(f * 1000))
        self.lbl_status.setText(msg)

    def _on_done(self, res):
        self._busy(False)
        self.result = res
        self.progress.setValue(1000)
        self.lbl_status.setText(f"Done: {res.n_used}/{res.n_keyframes} keyframes used")
        self.log.setPlainText("\n".join(res.log))
        if res.preview is not None:
            self.view_result.set_image(res.preview)
        self.lbl_result.setText(f"mosaic.tif: {res.width} × {res.height} px")
        self.btn_folder.setEnabled(True)
        self.tabs.setCurrentIndex(1)

    def _on_failed(self, msg):
        self._busy(False)
        self.lbl_status.setText(msg.splitlines()[0])
        self.log.setPlainText(msg)
        if msg != "Cancelled":
            QMessageBox.warning(self, "Mosaic failed", msg.splitlines()[0])

    def open_folder(self):
        if self.result:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(self.result.tiff_path)))

    def closeEvent(self, ev):
        if self.thread is not None and self.thread.isRunning():
            self.worker.cancel()
            self.thread.quit()
            self.thread.wait(10000)
        super().closeEvent(ev)


def main(argv=None):
    app = QApplication.instance() or QApplication(sys.argv if argv is None else argv)
    w = MainWindow()
    w.show()
    args = [a for a in (sys.argv[1:] if argv is None else argv[1:]) if not a.startswith("--")]
    if args and os.path.exists(args[0]):
        w.load_video(args[0])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
