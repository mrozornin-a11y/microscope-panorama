"""Detection of the circular field of view of the microscope."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import cv2
import numpy as np


@dataclass
class FieldOfView:
    """Circle of the microscope field in full-frame pixel coordinates."""

    cx: float
    cy: float
    radius: float
    frame_w: int
    frame_h: int
    margin: float = 0.04

    @property
    def r_eff(self) -> float:
        """Radius of the region that is actually used."""
        return self.radius * (1.0 - self.margin)

    @property
    def diameter(self) -> float:
        return 2.0 * self.r_eff

    def crop_box(self) -> Tuple[int, int, int, int]:
        """(x0, y0, x1, y1) of the bounding box of the used region, clipped."""
        r = self.r_eff
        x0 = max(0, int(math.floor(self.cx - r)))
        y0 = max(0, int(math.floor(self.cy - r)))
        x1 = min(self.frame_w, int(math.ceil(self.cx + r)) + 1)
        y1 = min(self.frame_h, int(math.ceil(self.cy + r)) + 1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            raise ValueError("Field of view lies outside the frame")
        return x0, y0, x1, y1

    def crop(self, frame: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self.crop_box()
        return frame[y0:y1, x0:x1]

    @property
    def crop_size(self) -> Tuple[int, int]:
        x0, y0, x1, y1 = self.crop_box()
        return x1 - x0, y1 - y0

    def center_in_crop(self, scale: float = 1.0) -> Tuple[float, float]:
        """Circle centre in crop coordinates of an image resized by `scale`.

        cv2.resize with fx=fy=scale maps pixel centres as
        p_s = (p + 0.5) * scale - 0.5.
        """
        x0, y0, _, _ = self.crop_box()
        return ((self.cx - x0 + 0.5) * scale - 0.5,
                (self.cy - y0 + 0.5) * scale - 0.5)

    def scaled_size(self, scale: float) -> Tuple[int, int]:
        w, h = self.crop_size
        return int(round(w * scale)), int(round(h * scale))

    def mask(self, scale: float = 1.0, shrink_px: float = 0.0) -> np.ndarray:
        """uint8 mask (0/255) of the used region in (scaled) crop coords."""
        w, h = self.scaled_size(scale)
        cx, cy = self.center_in_crop(scale)
        r = self.r_eff * scale - shrink_px
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        m = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r
        if shrink_px > 0:
            s = int(math.ceil(shrink_px))
            m[:s, :] = False
            m[-s:, :] = False
            m[:, :s] = False
            m[:, -s:] = False
        return (m.astype(np.uint8)) * 255

    def to_dict(self) -> dict:
        return dict(cx=self.cx, cy=self.cy, radius=self.radius,
                    frame_w=self.frame_w, frame_h=self.frame_h,
                    margin=self.margin)


def _fit_circle(points: np.ndarray) -> Tuple[float, float, float]:
    """Algebraic (Kasa) least-squares circle fit."""
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    a = np.stack([x, y, np.ones_like(x)], axis=1)
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r = math.sqrt(max(sol[2] + cx * cx + cy * cy, 1e-9))
    return cx, cy, r


def detect_fov(frames: Sequence[np.ndarray], margin: float = 0.04) -> FieldOfView:
    """Find the bright circular field of view on the black background.

    The per-pixel maximum over several frames is thresholded (Otsu), the
    largest blob is taken and a circle is fitted to its contour, ignoring the
    contour points lying on the image border (so that a circle partially
    cut by the frame edges is still recovered correctly).
    """
    h, w = frames[0].shape[:2]
    scale = min(1.0, 800.0 / max(h, w))
    acc = None
    for fr in frames:
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr
        if scale < 1:
            g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        g = cv2.GaussianBlur(g, (5, 5), 0)
        acc = g if acc is None else np.maximum(acc, g)
    _, th = cv2.threshold(acc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # The black background may be noisy: require a sensible minimum level.
    lvl = max(12, int(np.percentile(acc, 5)) + 8)
    th &= ((acc > lvl).astype(np.uint8) * 255)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(th)
    sh, sw = th.shape
    if n <= 1:
        # Nothing found: assume the whole frame is useful.
        return FieldOfView(w / 2, h / 2, math.hypot(w, h) / 2, w, h, margin)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    blob = (lab == best).astype(np.uint8) * 255
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea).reshape(-1, 2)
    hull = cv2.convexHull(cnt.reshape(-1, 1, 2)).reshape(-1, 2)
    # Densify convex hull so that every part of the boundary is represented.
    pts = []
    for i in range(len(hull)):
        p, q = hull[i].astype(np.float64), hull[(i + 1) % len(hull)].astype(np.float64)
        steps = max(1, int(np.linalg.norm(q - p)))
        t = np.linspace(0, 1, steps, endpoint=False)[:, None]
        pts.append(p + (q - p) * t)
    pts = np.concatenate(pts)
    border = 3
    inner = (pts[:, 0] > border) & (pts[:, 0] < sw - 1 - border) & \
            (pts[:, 1] > border) & (pts[:, 1] < sh - 1 - border)
    if inner.sum() >= 20:
        cx, cy, r = _fit_circle(pts[inner])
        # one robust re-fit iteration
        d = np.abs(np.hypot(pts[inner, 0] - cx, pts[inner, 1] - cy) - r)
        good = d < max(2.0, 3 * np.median(d))
        if good.sum() >= 20:
            cx, cy, r = _fit_circle(pts[inner][good])
    else:
        (cx, cy), r = cv2.minEnclosingCircle(cnt.astype(np.float32))
    # back to full resolution (pixel-centre convention)
    cx = (cx + 0.5) / scale - 0.5
    cy = (cy + 0.5) / scale - 0.5
    r = r / scale
    return FieldOfView(float(cx), float(cy), float(r), w, h, margin)


def draw_fov(frame: np.ndarray, fov: FieldOfView) -> np.ndarray:
    out = frame.copy()
    t = max(2, int(round(max(frame.shape[:2]) / 400)))
    c = (int(round(fov.cx)), int(round(fov.cy)))
    cv2.circle(out, c, int(round(fov.radius)), (0, 255, 255), t, cv2.LINE_AA)
    cv2.circle(out, c, int(round(fov.r_eff)), (0, 200, 0), t, cv2.LINE_AA)
    cv2.drawMarker(out, c, (0, 0, 255), cv2.MARKER_CROSS, 8 * t, t)
    return out
