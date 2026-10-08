"""Tile-based mosaic rendering with feather or multi-band blending.

The mosaic is rendered in square tiles and streamed directly into a tiled
(Big)TIFF, so the full-resolution panorama never has to fit in memory.
"""

from __future__ import annotations

import math
import warnings
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import cv2
import numpy as np
import tifffile

from .config import Settings
from .fov import FieldOfView

TIFF_TILE = 256

_INTERP = {"linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC,
           "lanczos": cv2.INTER_LANCZOS4}


@dataclass
class RenderFrame:
    path: str
    x: float          # world position of the circle centre
    y: float
    theta: float      # radians


# --------------------------------------------------------------------------
# flat field
# --------------------------------------------------------------------------
def estimate_flat_field(thumbs: Sequence[np.ndarray], fov: FieldOfView,
                        thumb_scale: float, out_size,
                        dark_level: float = 25.0) -> Optional[np.ndarray]:
    """Estimate the illumination profile (vignetting) of the microscope.

    The per-pixel median over many keyframes taken at different stage
    positions averages out the specimen; a smooth 2-D polynomial fitted to it
    is the illumination.  Dark background around the specimen is excluded
    from the median (otherwise frames along the specimen edge would make the
    illumination look dark on that side and over-brighten it).  Returns a
    gain image (full crop size, float32, 3 channels) to multiply frames
    with, or None.
    """
    if len(thumbs) < 8:
        return None
    stack = np.stack([t.astype(np.float32) for t in thumbs])
    gray = stack.mean(axis=3)
    h, w = stack.shape[1:3]
    mask = fov.mask(thumb_scale, shrink_px=1)[:h, :w] > 0
    p95 = np.percentile(gray[:, mask], 95, axis=1)
    bg = gray < np.maximum(dark_level, 0.3 * p95)[:, None, None]
    stack[bg] = np.nan
    n_valid = (~bg).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-background pixels
        med = np.nanmedian(stack, axis=0)
    mask &= n_valid >= max(5, 0.3 * len(thumbs))
    if mask.sum() < 0.5 * (fov.mask(thumb_scale, shrink_px=1)[:h, :w] > 0).sum():
        return None     # too few specimen samples to estimate the profile
    cx, cy = fov.center_in_crop(thumb_scale)
    r = fov.r_eff * thumb_scale
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    u, v = (xx - cx) / r, (yy - cy) / r

    def basis(u, v):
        terms = []
        for deg in range(5):
            for k in range(deg + 1):
                terms.append(u ** (deg - k) * v ** k)
        return np.stack(terms, axis=-1)

    A = basis(u[mask], v[mask])
    W, H = out_size
    # evaluate on a reduced grid, then upscale (the profile is smooth)
    gh, gw = max(2, H // 8), max(2, W // 8)
    gy, gx = np.mgrid[0:gh, 0:gw].astype(np.float64)
    fx, fy = gx * (W / gw), gy * (H / gh)
    fcx, fcy = fov.center_in_crop(1.0)
    B = basis((fx - fcx) / fov.r_eff, (fy - fcy) / fov.r_eff)
    gain = np.empty((gh, gw, 3), np.float32)
    for c in range(3):
        b = med[..., c][mask].astype(np.float64)
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        fit_in = A @ coef
        mean = float(fit_in.mean())
        if mean <= 1:
            return None
        illum = (B @ coef) / mean
        gain[..., c] = (1.0 / np.clip(illum, 0.3, 3.0)).astype(np.float32)
    return cv2.resize(gain, (W, H), interpolation=cv2.INTER_LINEAR)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _fill_invalid(img: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Push-pull extrapolation of `img` into pixels where valid == 0, so that
    a Laplacian pyramid does not see the hard edge of the circle."""
    v = valid.astype(np.float32)
    a = img * v[..., None]
    pa, pv = [a], [v]
    while min(pa[-1].shape[:2]) > 4:
        pa.append(cv2.pyrDown(pa[-1]))
        pv.append(cv2.pyrDown(pv[-1]))
    f = pa[-1] / np.maximum(pv[-1], 1e-6)[..., None]
    for k in range(len(pa) - 2, -1, -1):
        h, w = pa[k].shape[:2]
        up = cv2.pyrUp(f, dstsize=(w, h))
        if up.ndim == 2:
            up = up[..., None]
        # pa is premultiplied by the coverage pv, so this keeps valid pixels
        # and fills the rest from the coarser level
        f = pa[k] + (1 - np.clip(pv[k], 0, 1))[..., None] * up
    return f.astype(np.float32)


class _FrameCache:
    def __init__(self, loader: Callable[[int], np.ndarray], maxsize: int = 12):
        self.loader = loader
        self.maxsize = maxsize
        self.data: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self.lock = threading.Lock()

    def get(self, k: int) -> np.ndarray:
        with self.lock:
            if k in self.data:
                self.data.move_to_end(k)
                return self.data[k]
        img = self.loader(k)
        with self.lock:
            self.data[k] = img
            while len(self.data) > self.maxsize:
                self.data.popitem(last=False)
        return img


# --------------------------------------------------------------------------
class MosaicRenderer:
    def __init__(self, fov: FieldOfView, frames: List[RenderFrame],
                 settings: Settings, gain: Optional[np.ndarray] = None):
        self.fov = fov
        self.frames = frames
        self.st = settings
        self.gain = gain
        self.c = np.array(fov.center_in_crop(1.0))
        r = fov.r_eff + 2
        xs = np.array([f.x for f in frames])
        ys = np.array([f.y for f in frames])
        self.origin = np.array([math.floor(xs.min() - r), math.floor(ys.min() - r)])
        self.width = int(math.ceil(xs.max() + r) - self.origin[0]) + 1
        self.height = int(math.ceil(ys.max() + r) - self.origin[1]) + 1
        self.bx0 = np.floor(xs - r - self.origin[0]).astype(int)
        self.bx1 = np.ceil(xs + r - self.origin[0]).astype(int) + 1
        self.by0 = np.floor(ys - r - self.origin[1]).astype(int)
        self.by1 = np.ceil(ys + r - self.origin[1]).astype(int) + 1
        mask = fov.mask(1.0, shrink_px=2)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        dist /= max(float(dist.max()), 1e-6)
        if settings.blending == "feather":
            dist = dist ** settings.feather_power
        self.weight = dist.astype(np.float32)
        self.interp = _INTERP.get(settings.interpolation, cv2.INTER_CUBIC)
        self.cache = _FrameCache(self._load, maxsize=16)

    # ------------------------------------------------------------------
    def _load(self, k: int) -> np.ndarray:
        img = cv2.imread(self.frames[k].path, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Cannot read cached frame {self.frames[k].path}")
        if self.gain is not None:
            g = self.gain[:img.shape[0], :img.shape[1]]
            img = np.clip(img.astype(np.float32) * g, 0, 255).astype(np.uint8)
        return img

    def _matrix(self, k: int, ox: float, oy: float) -> np.ndarray:
        f = self.frames[k]
        c, s = math.cos(f.theta), math.sin(f.theta)
        R = np.array([[c, -s], [s, c]])
        t = np.array([f.x, f.y]) - R @ self.c - self.origin - np.array([ox, oy])
        return np.hstack([R, t[:, None]]).astype(np.float64)

    def _frames_in(self, x0, y0, x1, y1) -> np.ndarray:
        return np.flatnonzero((self.bx0 < x1) & (self.bx1 > x0) &
                              (self.by0 < y1) & (self.by1 > y0))

    def _warp(self, k, M, w, h):
        img = cv2.warpAffine(self.cache.get(k), M, (w, h), flags=self.interp,
                             borderMode=cv2.BORDER_CONSTANT)
        wt = cv2.warpAffine(self.weight, M, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT)
        return img, wt

    # ------------------------------------------------------------------
    def render_tile(self, x0: int, y0: int, tw: int, th: int) -> np.ndarray:
        if self.st.blending == "multiband":
            return self._tile_multiband(x0, y0, tw, th)
        return self._tile_simple(x0, y0, tw, th)

    def _roi(self, k, x0, y0, x1, y1, align=1):
        rx0 = max(self.bx0[k], x0)
        ry0 = max(self.by0[k], y0)
        rx1 = min(self.bx1[k], x1)
        ry1 = min(self.by1[k], y1)
        if align > 1:
            rx0 = x0 + ((rx0 - x0) // align) * align
            ry0 = y0 + ((ry0 - y0) // align) * align
            rx1 = min(x1, x0 + -(-(rx1 - x0) // align) * align)
            ry1 = min(y1, y0 + -(-(ry1 - y0) // align) * align)
        if rx1 <= rx0 or ry1 <= ry0:
            return None
        return int(rx0), int(ry0), int(rx1), int(ry1)

    def _tile_simple(self, x0, y0, tw, th) -> np.ndarray:
        x1, y1 = x0 + tw, y0 + th
        acc = np.zeros((th, tw, 3), np.float32)
        ws = np.zeros((th, tw), np.float32)
        hard = self.st.blending == "none"
        for k in self._frames_in(x0, y0, x1, y1):
            roi = self._roi(k, x0, y0, x1, y1)
            if roi is None:
                continue
            rx0, ry0, rx1, ry1 = roi
            M = self._matrix(k, rx0, ry0)
            img, w = self._warp(k, M, rx1 - rx0, ry1 - ry0)
            sl = (slice(ry0 - y0, ry1 - y0), slice(rx0 - x0, rx1 - x0))
            if hard:
                upd = w > ws[sl]
                acc[sl][upd] = img[upd]
                ws[sl][upd] = w[upd]
            else:
                acc[sl] += img.astype(np.float32) * w[..., None]
                ws[sl] += w
        if hard:
            out = acc
        else:
            out = acc / np.maximum(ws, 1e-8)[..., None]
        out[ws <= 0] = 0
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)

    def _tile_multiband(self, x0, y0, tw, th) -> np.ndarray:
        L = max(1, int(self.st.blend_levels))
        a = 2 ** L
        pad = 4 * a
        px0, py0 = x0 - pad, y0 - pad
        PW = -(-(tw + 2 * pad) // a) * a
        PH = -(-(th + 2 * pad) // a) * a
        px1, py1 = px0 + PW, py0 + PH
        ks = self._frames_in(px0, py0, px1, py1)
        if len(ks) == 0:
            return np.zeros((th, tw, 3), np.uint8)
        best = np.zeros((PH, PW), np.float32)
        arg = np.full((PH, PW), -1, np.int32)
        rois = {}
        for k in ks:
            roi = self._roi(k, px0, py0, px1, py1, align=a)
            if roi is None:
                continue
            rois[k] = roi
            rx0, ry0, rx1, ry1 = roi
            M = self._matrix(k, rx0, ry0)
            w = cv2.warpAffine(self.weight, M, (rx1 - rx0, ry1 - ry0),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            sl = (slice(ry0 - py0, ry1 - py0), slice(rx0 - px0, rx1 - px0))
            b = best[sl]
            upd = w > b
            b[upd] = w[upd]
            arg[sl][upd] = k
        acc = [np.zeros((PH >> l, PW >> l, 3), np.float32) for l in range(L + 1)]
        wsum = [np.zeros((PH >> l, PW >> l), np.float32) for l in range(L + 1)]
        for k, (rx0, ry0, rx1, ry1) in rois.items():
            sl = (slice(ry0 - py0, ry1 - py0), slice(rx0 - px0, rx1 - px0))
            win = (arg[sl] == k)
            if not win.any():
                continue
            M = self._matrix(k, rx0, ry0)
            img, w = self._warp(k, M, rx1 - rx0, ry1 - ry0)
            filled = _fill_invalid(img.astype(np.float32), (w > 0).astype(np.float32))
            G = [filled]
            Mk = [win.astype(np.float32)]
            for l in range(L):
                G.append(cv2.pyrDown(G[-1]))
                Mk.append(cv2.pyrDown(Mk[-1]))
            for l in range(L + 1):
                if l < L:
                    h, w_ = G[l].shape[:2]
                    lap = G[l] - cv2.pyrUp(G[l + 1], dstsize=(w_, h))
                else:
                    lap = G[l]
                oy, ox = (ry0 - py0) >> l, (rx0 - px0) >> l
                hh, ww = lap.shape[:2]
                acc[l][oy:oy + hh, ox:ox + ww] += lap * Mk[l][..., None]
                wsum[l][oy:oy + hh, ox:ox + ww] += Mk[l]
        R = acc[L] / np.maximum(wsum[L], 1e-6)[..., None]
        for l in range(L - 1, -1, -1):
            h, w_ = acc[l].shape[:2]
            band = acc[l] / np.maximum(wsum[l], 1e-6)[..., None]
            R = cv2.pyrUp(R, dstsize=(w_, h)) + band
        out = R[pad:pad + th, pad:pad + tw]
        cover = best[pad:pad + th, pad:pad + tw] > 0
        out[~cover] = 0
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    def render(self, tiff_path: str, preview_path: Optional[str] = None,
               progress: Optional[Callable[[float, str], None]] = None,
               cancel: Optional[Callable[[], bool]] = None) -> np.ndarray:
        """Render the whole mosaic into a tiled TIFF; return the preview."""
        st = self.st
        W, H = self.width, self.height
        T = max(TIFF_TILE, (int(st.tile_size) // TIFF_TILE) * TIFF_TILE)
        Wp = -(-W // TIFF_TILE) * TIFF_TILE
        f = min(1.0, st.preview_max_size / max(W, H))
        pw, ph = max(1, int(round(W * f))), max(1, int(round(H * f)))
        preview = np.zeros((ph, pw, 3), np.uint8)
        n_bands = -(-H // T)
        n_cols = -(-Wp // T)
        threads = st.threads or None

        def bands():
            with ThreadPoolExecutor(threads) as ex:
                for bi, y0 in enumerate(range(0, H, T)):
                    if cancel and cancel():
                        raise InterruptedError
                    th = T
                    band = np.zeros((th, n_cols * T, 3), np.uint8)
                    xs = list(range(0, n_cols * T, T))
                    tiles = list(ex.map(lambda x: self.render_tile(x, y0, T, th), xs))
                    for x, t in zip(xs, tiles):
                        band[:, x:x + T] = t
                    band = band[:, :Wp]
                    hb = min(T, H - y0)
                    # preview
                    r0, r1 = int(round(y0 * f)), int(round((y0 + hb) * f))
                    if r1 > r0:
                        small = cv2.resize(band[:hb, :W], (pw, r1 - r0),
                                           interpolation=cv2.INTER_AREA)
                        preview[r0:r1] = small
                    if progress:
                        progress((bi + 1) / n_bands, f"Rendering band {bi + 1}/{n_bands}")
                    rows = -(-hb // TIFF_TILE)
                    rgb = cv2.cvtColor(band[:rows * TIFF_TILE], cv2.COLOR_BGR2RGB)
                    for ty in range(rows):
                        for tx in range(0, Wp, TIFF_TILE):
                            yield rgb[ty * TIFF_TILE:(ty + 1) * TIFF_TILE,
                                      tx:tx + TIFF_TILE]

        comp = None if st.tiff_compression in ("", "none", None) else st.tiff_compression
        big = W * H * 3 > 2 ** 31
        tifffile.imwrite(tiff_path, data=bands(), shape=(H, W, 3), dtype=np.uint8,
                         tile=(TIFF_TILE, TIFF_TILE), photometric="rgb",
                         compression=comp, bigtiff=big, metadata=None)
        if preview_path:
            cv2.imwrite(preview_path, preview, [cv2.IMWRITE_JPEG_QUALITY, int(st.jpeg_quality)])
        return preview
