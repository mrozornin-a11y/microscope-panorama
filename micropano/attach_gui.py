"""Dialog for manually attaching an unlinked component to the main mosaic."""

from __future__ import annotations

import math
from typing import List, Optional

import cv2
import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QDialog, QDoubleSpinBox, QFormLayout, QGraphicsEllipseItem,
    QGraphicsItem, QGraphicsPixmapItem, QGraphicsSimpleTextItem, QGraphicsView, QGroupBox,
    QHBoxLayout, QLabel, QMessageBox, QPushButton, QSlider, QSplitter, QTabWidget,
    QVBoxLayout, QWidget,
)

from .diagnostics import link_preview
from .gui import ImageView, to_pixmap
from .merge import AlignResult, MergeSession, transform_from_points


def _rgba_pixmap(img_bgr: np.ndarray) -> QPixmap:
    """Pixmap whose black background is transparent."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    alpha = ((img_bgr.astype(np.int32).sum(axis=2) > 24) * 255).astype(np.uint8)
    rgba = np.ascontiguousarray(np.dstack([rgb, alpha]))
    h, w = rgba.shape[:2]
    return QPixmap.fromImage(QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888).copy())


class _Overlay(QGraphicsPixmapItem):
    def __init__(self, on_change):
        super().__init__()
        self._on_change = on_change
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsGeometryChanges)
        self.setCursor(Qt.SizeAllCursor)
        self.setZValue(1)

    def itemChange(self, change, value):
        if change in (QGraphicsItem.ItemPositionHasChanged, QGraphicsItem.ItemRotationHasChanged):
            self._on_change()
        return super().itemChange(change, value)


class AlignView(ImageView):
    """Main mosaic with the component as a draggable, rotatable overlay.
    Drag the component with the mouse, Shift+wheel rotates it, arrow keys
    nudge it (Shift = 10x)."""

    rotated = Signal(float)

    def __init__(self, on_change, parent=None):
        super().__init__(parent)
        self.overlay = _Overlay(on_change)
        self.overlay.setOpacity(0.6)
        self.scene().addItem(self.overlay)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, ev):
        if ev.modifiers() & Qt.ShiftModifier:
            d = ev.angleDelta().y() or ev.angleDelta().x()
            step = 0.25 if d > 0 else -0.25
            self.overlay.setRotation(self.overlay.rotation() + step)
            self.rotated.emit(self.overlay.rotation())
        else:
            super().wheelEvent(ev)

    def keyPressEvent(self, ev):
        k = 10.0 if ev.modifiers() & Qt.ShiftModifier else 1.0
        d = {Qt.Key_Left: (-k, 0), Qt.Key_Right: (k, 0), Qt.Key_Up: (0, -k),
             Qt.Key_Down: (0, k)}.get(ev.key())
        if d:
            self.overlay.setPos(self.overlay.pos() + QPointF(*d))
        else:
            super().keyPressEvent(ev)


class PointView(ImageView):
    """Click to place numbered corresponding points; right click removes the
    last one."""

    changed = Signal()

    def __init__(self, color, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.NoDrag)
        self.color = QColor(color)
        self.points: List[QPointF] = []
        self._items = []

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            p = self.mapToScene(ev.position().toPoint())
            self.points.append(p)
            self._draw()
            self.changed.emit()
        elif ev.button() == Qt.RightButton and self.points:
            self.points.pop()
            self._draw()
            self.changed.emit()

    def clear_points(self):
        self.points.clear()
        self._draw()
        self.changed.emit()

    def _draw(self):
        for it in self._items:
            self.scene().removeItem(it)
        self._items = []
        pen = QPen(self.color, 2)
        pen.setCosmetic(True)
        s = 6 / max(self.transform().m11(), 1e-6)
        font = QFont()
        font.setPointSizeF(max(6.0, 11 / max(self.transform().m11(), 1e-6)))
        for k, p in enumerate(self.points):
            e = QGraphicsEllipseItem(p.x() - s, p.y() - s, 2 * s, 2 * s)
            e.setPen(pen)
            t = QGraphicsSimpleTextItem(str(k + 1))
            t.setBrush(QBrush(self.color))
            t.setFont(font)
            t.setPos(p.x() + s, p.y() + s)
            for it in (e, t):
                it.setZValue(2)
                self.scene().addItem(it)
                self._items.append(it)


class AttachDialog(QDialog):
    """Returns (via .decision) a dict {T, method, quality} when the user
    confirms; nothing is changed on disk by the dialog itself."""

    def __init__(self, session: MergeSession, comp: int, parent=None):
        super().__init__(parent)
        self.ses = session
        self.c = comp
        self.decision: Optional[dict] = None
        self.result: Optional[AlignResult] = None
        self.refined_T: Optional[np.ndarray] = None
        self.setWindowTitle(f"Attach component {comp} to the main mosaic")
        self.resize(1400, 860)

        mo, co = session.output(None), session.output(comp)
        self.fm = session.preview_scale(mo)
        self.om = np.array(mo["origin"])
        self.oc = np.array(co["origin"])
        self.fc = session.preview_scale(co)
        main_img = session.preview_image(None)
        comp_img = session.preview_image(comp)
        k = self.fm / self.fc
        ov = cv2.resize(comp_img, None, fx=k, fy=k,
                        interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)

        # ---- views
        self.tabs = QTabWidget()
        self.view = AlignView(self._overlay_changed)
        self.view.set_image(main_img)
        self.view.overlay.setPixmap(_rgba_pixmap(ov))
        self.cp = np.array([ov.shape[1] / 2.0, ov.shape[0] / 2.0])
        self.view.overlay.setTransformOriginPoint(QPointF(*self.cp))
        self.view.rotated.connect(self._view_rotated)
        self.tabs.addTab(self.view, "Move / rotate")

        pts = QWidget()
        pl = QVBoxLayout(pts)
        pl.setContentsMargins(0, 0, 0, 0)
        hint = QLabel("Click corresponding features: point 1 on the main mosaic (left) and "
                      "point 1 on the component (right), then 2, 3…  Right click removes the "
                      "last point. 1 pair = shift only, 2-3 pairs = shift + rotation.")
        hint.setWordWrap(True)
        pl.addWidget(hint)
        sp = QSplitter()
        self.pv_main = PointView("#ff3cff")
        self.pv_main.set_image(main_img)
        self.pv_comp = PointView("#3cff3c")
        self.pv_comp.set_image(comp_img)
        for v in (self.pv_main, self.pv_comp):
            v.changed.connect(self._points_changed)
            sp.addWidget(v)
        pl.addWidget(sp, 1)
        prow = QHBoxLayout()
        self.lbl_pts = QLabel("0 / 0 points")
        b_clear = QPushButton("Clear points")
        b_clear.clicked.connect(lambda: (self.pv_main.clear_points(), self.pv_comp.clear_points()))
        self.btn_apply_pts = QPushButton("Apply points")
        self.btn_apply_pts.setEnabled(False)
        self.btn_apply_pts.clicked.connect(self._apply_points)
        prow.addWidget(self.lbl_pts, 1)
        prow.addWidget(b_clear)
        prow.addWidget(self.btn_apply_pts)
        pl.addLayout(prow)
        self.tabs.addTab(pts, "Corresponding points")

        # ---- side panel
        side = QWidget()
        side.setMinimumWidth(380)
        sl = QVBoxLayout(side)
        info = QLabel(f"<b>Component {comp}</b>: {session.frame_info(comp)}<br>"
                      f"<span style='color:gray'>Not linked automatically: "
                      f"{session.comp_status(comp).get('reason', '')}</span>")
        info.setWordWrap(True)
        sl.addWidget(info)

        g1 = QGroupBox("1. Rough position")
        f1 = QFormLayout(g1)
        self.sp_rot = QDoubleSpinBox()
        self.sp_rot.setRange(-15, 15)
        self.sp_rot.setSingleStep(0.1)
        self.sp_rot.setDecimals(2)
        self.sp_rot.setSuffix(" °")
        self.sp_rot.valueChanged.connect(self._rot_spin)
        f1.addRow("Rotation", self.sp_rot)
        self.sl_op = QSlider(Qt.Horizontal)
        self.sl_op.setRange(10, 100)
        self.sl_op.setValue(60)
        self.sl_op.valueChanged.connect(lambda v: self.view.overlay.setOpacity(v / 100))
        f1.addRow("Overlay opacity", self.sl_op)
        h = QLabel("Drag the component; Shift+wheel rotates; arrows nudge (Shift ×10). "
                   "Or use the 'Corresponding points' tab.")
        h.setWordWrap(True)
        h.setStyleSheet("color: gray")
        f1.addRow(h)
        sl.addWidget(g1)

        g2 = QGroupBox("2. Automatic refinement (NCC/ECC)")
        f2 = QFormLayout(g2)
        self.sp_rad = QDoubleSpinBox()
        self.sp_rad.setRange(0.05, 0.6)
        self.sp_rad.setSingleStep(0.05)
        self.sp_rad.setValue(0.3)
        self.sp_rad.setToolTip("Search window around the manual position, as a fraction of "
                               "the field-of-view diameter")
        f2.addRow("Search radius (× diameter)", self.sp_rad)
        self.btn_refine = QPushButton("Refine position automatically")
        self.btn_refine.clicked.connect(self._refine)
        f2.addRow(self.btn_refine)
        self.lbl_q = QLabel("Not refined yet.")
        self.lbl_q.setWordWrap(True)
        f2.addRow(self.lbl_q)
        self.lbl_pair = QLabel()
        self.lbl_pair.setAlignment(Qt.AlignCenter)
        self.lbl_pair.setMinimumHeight(220)
        f2.addRow(self.lbl_pair)
        sl.addWidget(g2, 1)

        g3 = QGroupBox("3. Merge")
        v3 = QVBoxLayout(g3)
        self.btn_merge_reg = QPushButton("Merge — registered position")
        self.btn_merge_reg.setEnabled(False)
        self.btn_merge_reg.clicked.connect(self._merge_registered)
        self.btn_merge_man = QPushButton("Merge — keep MANUAL position…")
        self.btn_merge_man.clicked.connect(self._merge_manual)
        b_cancel = QPushButton("Cancel")
        b_cancel.clicked.connect(self.reject)
        note = QLabel("Merging re-renders mosaic_merged.tif; mosaic.tif and the unlinked "
                      "TIFFs are not modified.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray")
        for w in (self.btn_merge_reg, self.btn_merge_man, b_cancel, note):
            v3.addWidget(w)
        sl.addWidget(g3)

        split = QSplitter()
        split.addWidget(self.tabs)
        split.addWidget(side)
        split.setStretchFactor(0, 1)
        lay = QVBoxLayout(self)
        lay.addWidget(split)

        # initial position: component centred on the main mosaic
        mc = np.mean([session.world_pose(i)[:2] for i in session.main_ids()], axis=0)
        cc = np.mean([session.pose(i)[:2] for i in session.comp_ids(comp)], axis=0)
        self.set_T(np.array([*(mc - cc), 0.0]))

    # ----------------------------------------------------------- transform
    def T(self) -> np.ndarray:
        it = self.view.overlay
        th = math.radians(it.rotation())
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        pos = np.array([it.pos().x(), it.pos().y()])
        b = pos + self.cp - R @ self.cp
        t = b / self.fm - R @ self.oc + self.om
        return np.array([t[0], t[1], th])

    def set_T(self, T):
        tx, ty, th = T
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        b = self.fm * (R @ self.oc + np.array([tx, ty]) - self.om)
        pos = b - self.cp + R @ self.cp
        it = self.view.overlay
        it.setRotation(math.degrees(th))
        it.setPos(QPointF(*pos))
        self.sp_rot.blockSignals(True)
        self.sp_rot.setValue(math.degrees(th))
        self.sp_rot.blockSignals(False)

    def _rot_spin(self, deg):
        self.view.overlay.setRotation(deg)

    def _view_rotated(self, deg):
        self.sp_rot.blockSignals(True)
        self.sp_rot.setValue(deg)
        self.sp_rot.blockSignals(False)

    def _overlay_changed(self):
        # a registered position is only valid until the user moves it again
        if self.refined_T is not None:
            d = self.T() - self.refined_T
            if math.hypot(d[0], d[1]) > 0.5 or abs(d[2]) > 1e-4:
                self.refined_T = None
                self.btn_merge_reg.setEnabled(False)
                self.lbl_q.setText(self.lbl_q.text().split("<br><i>")[0] +
                                   "<br><i>Position changed after refinement — refine again "
                                   "to merge as registered.</i>")

    # -------------------------------------------------------------- points
    def _points_changed(self):
        n1, n2 = len(self.pv_main.points), len(self.pv_comp.points)
        self.lbl_pts.setText(f"{n1} point(s) on main / {n2} on component")
        self.btn_apply_pts.setEnabled(min(n1, n2) >= 1)

    def _apply_points(self):
        n = min(len(self.pv_main.points), len(self.pv_comp.points))
        mp = [np.array([p.x(), p.y()]) / self.fm + self.om for p in self.pv_main.points[:n]]
        cp = [np.array([p.x(), p.y()]) / self.fc + self.oc for p in self.pv_comp.points[:n]]
        T, rms = transform_from_points(mp, cp)
        if abs(math.degrees(T[2])) > 15:
            QMessageBox.warning(self, "Points", f"The points imply a rotation of "
                                f"{math.degrees(T[2]):.1f}°, which is implausible for a "
                                f"stage that only translates. Check the point order.")
            return
        self.set_T(T)
        self.tabs.setCurrentIndex(0)
        self.lbl_q.setText(f"Position from {n} point pair(s), point RMS {rms:.1f} px "
                           f"(full resolution). Now refine automatically.")

    # -------------------------------------------------------------- refine
    def _refine(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            T0 = self.T()
            r = self.ses.refine(self.c, T0, self.sp_rad.value())
        except Exception as e:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Refinement failed", str(e))
            return
        QApplication.restoreOverrideCursor()
        self.result = r
        if r.n_matched:
            self.set_T(r.T)
        col = "#1e8449" if r.confirmed else "#c0392b"
        self.lbl_q.setText(f"<span style='color:{col}'><b>{r.message}</b></span><br>"
                           + r.summary().replace(r.message, ""))
        self.refined_T = self.T() if r.confirmed else None
        self.btn_merge_reg.setEnabled(r.confirmed)
        self._show_pair(r)

    def _show_pair(self, r: AlignResult):
        good = [p for p in r.pairs if p.rel is not None]
        if not good:
            self.lbl_pair.clear()
            return
        p = max(good, key=lambda p: (p.consistent, p.rel.quality))
        title = (f"best pair: main kf {p.m} <- component kf {p.s}\n"
                 f"ECC {p.rel.quality:.3f}  residual {p.residual:.1f} px  "
                 f"{'consistent' if p.consistent else 'INCONSISTENT'}")
        img = link_preview(self.ses.reg, self.ses.keyframe(p.m), self.ses.keyframe(p.s),
                           p.rel, title)
        pm = to_pixmap(img).scaled(360, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.lbl_pair.setPixmap(pm)

    # --------------------------------------------------------------- merge
    def _merge_registered(self):
        r = self.result
        self.decision = dict(T=self.T(), method="registered",
                             quality=r.summary() if r else "")
        self.accept()

    def _merge_manual(self):
        q = ("The component will be merged at the position you set by hand, WITHOUT "
             "confirmed automatic registration.\n\nIt will be marked as 'manually "
             "positioned' (status merged-manual in frame_positions_merged.csv and in "
             "merge_log.txt).\n\nContinue?")
        if self.refined_T is not None:
            q = ("The refined position is confirmed — 'Merge — registered position' is "
                 "recommended.\n\n") + q
        if QMessageBox.question(self, "Merge manually positioned component", q) \
                != QMessageBox.Yes:
            return
        r = self.result
        qual = "manual position"
        if r is not None:
            qual += "; last refinement: " + r.summary()
        self.decision = dict(T=self.T(), method="manual", quality=qual)
        self.accept()
