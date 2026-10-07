"""Pairwise registration of frames inside the circular mask.

Two stages:

1. Coarse global search: masked normalised cross-correlation (Padfield,
   "Masked object registration in the Fourier domain", 2012) on strongly
   downscaled, high-pass filtered images.  Only pixels inside the circular
   field of view participate, so the black background never correlates.
   An ambiguity test (second-best peak) rejects matches on repetitive
   texture; an optional predicted shift restricts the search window.
2. Fine alignment: ECC (enhanced correlation coefficient) at the working
   scale, translation or rigid (translation + small rotation), initialised
   from the coarse result.  The ECC coefficient is the registration quality.

Coordinate conventions
----------------------
Each frame has a *local* coordinate system in full-resolution pixels with the
origin at the centre of the field-of-view circle.  A relative pose (edge)
between frames i and j is stored as

    x_i = R(phi) @ x_j + d

i.e. it maps local coordinates of frame j into local coordinates of frame i.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy import fft as sfft

from .config import Settings
from .fov import FieldOfView


@dataclass
class RelPose:
    dx: float
    dy: float
    phi: float = 0.0          # radians
    quality: float = 0.0      # ECC correlation coefficient
    ncc: float = 0.0          # coarse NCC peak
    overlap: float = 0.0      # overlap area / mask area

    @property
    def shift(self) -> float:
        return math.hypot(self.dx, self.dy)


@dataclass
class CoarseMatch:
    sx: float                 # p_A = p_B + s  (coarse pixels)
    sy: float
    peak: float
    second: float
    overlap: float


@dataclass
class Prepared:
    """Per-frame data needed for registration."""
    work: np.ndarray                      # uint8 gray, working scale
    coarse: np.ndarray                    # float32 high-pass, coarse scale
    sharpness: float
    _work_hp: Optional[np.ndarray] = field(default=None, repr=False)
    _spectra: Optional[tuple] = field(default=None, repr=False)
    _link: Optional["LinkData"] = field(default=None, repr=False)


@dataclass
class LinkData:
    """Specimen-only representation of a frame for cross-segment matching:
    the black background around the specimen and a band along the outer
    specimen boundary are excluded, so that the long high-contrast edge of
    the section cannot dominate the correlation."""
    core_w: np.ndarray            # uint8 0/255 mask at working scale
    work_hp: np.ndarray           # float32 high-pass, zero outside core_w
    spectra: tuple                # (F, F2, M) at coarse scale
    frac: float                   # core area / field-of-view area


def resize_exact(img: np.ndarray, scale: float, size: Tuple[int, int],
                 interp=cv2.INTER_AREA) -> np.ndarray:
    """Resize with exactly `scale` (pixel-centre convention) and pad/crop the
    result to `size` = (w, h)."""
    if abs(scale - 1.0) < 1e-12:
        r = img
    else:
        r = cv2.resize(img, None, fx=scale, fy=scale, interpolation=interp)
    w, h = size
    if r.shape[1] == w and r.shape[0] == h:
        return r
    out = np.zeros((h, w) + r.shape[2:], r.dtype)
    hh, ww = min(h, r.shape[0]), min(w, r.shape[1])
    out[:hh, :ww] = r[:hh, :ww]
    if hh < h:
        out[hh:, :ww] = r[hh - 1:hh, :ww]
    if ww < w:
        out[:, ww:] = out[:, ww - 1:ww]
    return out


def _smooth(img: np.ndarray, sigma: float) -> np.ndarray:
    """Large-sigma Gaussian blur computed on a downscaled image (the result is
    a smooth low-pass, so the shortcut is visually exact and much faster)."""
    k = int(sigma // 3)
    if k < 2:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // k), max(1, h // k)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / k)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def highpass(gray: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Remove illumination gradients: subtract a mask-normalised Gaussian
    blur, normalise to unit variance, clip outliers (specular glints) and
    zero the background."""
    g = gray.astype(np.float32)
    m = (mask > 0).astype(np.float32)
    num = _smooth(g * m, sigma)
    den = _smooth(m, sigma)
    low = num / np.maximum(den, 1e-3)
    hp = (g - low) * m
    sd = float(np.sqrt((hp ** 2).sum() / max(m.sum(), 1.0)))
    if sd > 0:
        hp = np.clip(hp / sd, -4.0, 4.0)
    return hp


class MaskedNCC:
    """Masked normalised cross-correlation for a fixed mask shared by both
    images.  Returns NCC as a function of shift s, where content at p in the
    moving image corresponds to p + s in the fixed image.

    Computations are in float32; images are expected to be normalised
    (see `highpass`) so the precision is ample."""

    def __init__(self, mask: np.ndarray, min_overlap_px: float):
        self.h, self.w = mask.shape
        self.m = (mask > 0).astype(np.float32)
        self.shape = (sfft.next_fast_len(2 * self.h - 1, real=True),
                      sfft.next_fast_len(2 * self.w - 1, real=True))
        self.M = sfft.rfft2(self.m, s=self.shape)
        self.iy = np.arange(-(self.h - 1), self.h) % self.shape[0]
        self.ix = np.arange(-(self.w - 1), self.w) % self.shape[1]
        ov = self._corr(self.M, self.M)
        self.overlap = np.round(ov)
        self.valid = self.overlap >= max(min_overlap_px, 16)
        self.ov_safe = np.maximum(self.overlap, 1.0).astype(np.float32)
        self.area = float(self.m.sum())

    def _corr(self, a_hat, b_hat) -> np.ndarray:
        c = sfft.irfft2(a_hat * np.conj(b_hat), s=self.shape)
        return c[np.ix_(self.iy, self.ix)]

    def spectra(self, img: np.ndarray):
        f = img.astype(np.float32) * self.m
        return sfft.rfft2(f, s=self.shape), sfft.rfft2(f * f, s=self.shape)

    def __call__(self, fixed, moving) -> np.ndarray:
        """`fixed`/`moving`: images or precomputed `spectra()` tuples."""
        F, F2 = fixed if isinstance(fixed, tuple) else self.spectra(fixed)
        G, G2 = moving if isinstance(moving, tuple) else self.spectra(moving)
        M = self.M
        fg = self._corr(F, G)
        fm = self._corr(F, M)
        mg = self._corr(M, G)
        ov = self.ov_safe
        num = fg - fm * mg / ov
        d1 = self._corr(F2, M) - fm * fm / ov
        d2 = self._corr(M, G2) - mg * mg / ov
        den = np.sqrt(np.maximum(d1, 0) * np.maximum(d2, 0))
        tol = 1e-4 * max(float(np.max(den)), 1e-12)
        ncc = np.where((den > tol) & self.valid, num / np.maximum(den, tol), 0.0)
        return np.clip(ncc, -1.0, 1.0)


def _masked_ncc_general(eng: "MaskedNCC", A: tuple, B: tuple, min_overlap_px: float):
    """Padfield masked NCC with different masks for both images.
    A, B: (F, F2, M) spectra of image*mask, image^2*mask and mask."""
    F, F2, Mf = A
    G, G2, Mg = B
    ov = np.round(eng._corr(Mf, Mg))
    valid = ov >= max(min_overlap_px, 16)
    ovs = np.maximum(ov, 1.0)
    fm = eng._corr(F, Mg)
    mg = eng._corr(Mf, G)
    num = eng._corr(F, G) - fm * mg / ovs
    d1 = eng._corr(F2, Mg) - fm * fm / ovs
    d2 = eng._corr(Mf, G2) - mg * mg / ovs
    den = np.sqrt(np.maximum(d1, 0) * np.maximum(d2, 0))
    tol = 1e-4 * max(float(np.max(den)), 1e-12)
    ncc = np.where((den > tol) & valid, num / np.maximum(den, tol), 0.0)
    return np.clip(ncc, -1.0, 1.0), valid, ov


class Registrar:
    def __init__(self, fov: FieldOfView, settings: Settings):
        self.fov = fov
        self.settings = settings
        D = fov.diameter
        self.D = D
        self.s_work = min(1.0, settings.work_diameter / D)
        self.s_coarse = min(self.s_work, settings.coarse_diameter / D)
        self.work_size = fov.scaled_size(self.s_work)
        self.coarse_size = fov.scaled_size(self.s_coarse)
        self.mask_w = fov.mask(self.s_work)
        self.mask_c = fov.mask(self.s_coarse)
        self.mask_w_inner = fov.mask(self.s_work, shrink_px=4)
        self.c_work = np.array(fov.center_in_crop(self.s_work))
        self.sigma_w = max(2.0, D * self.s_work / 40.0)
        self.sigma_c = max(1.5, D * self.s_coarse / 40.0)
        area_c = float((self.mask_c > 0).sum())
        self.ncc = MaskedNCC(self.mask_c, settings.min_overlap_area * area_c)
        self.Dc = D * self.s_coarse
        self.f_cw = self.s_work / self.s_coarse
        self.area_w = float((self.mask_w > 0).sum())
        self._grid = np.ogrid[-(self.ncc.h - 1):self.ncc.h, -(self.ncc.w - 1):self.ncc.w]

    # ----------------------------------------------------------------- prep
    def prepare(self, crop_bgr: np.ndarray) -> Prepared:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
        work = resize_exact(gray, self.s_work, self.work_size)
        lap = cv2.Laplacian(work.astype(np.float32), cv2.CV_32F, ksize=3)
        m = self.mask_w_inner > 0
        sharp = float(lap[m].var()) if m.any() else 0.0
        coarse = resize_exact(work, self.s_coarse / self.s_work, self.coarse_size)
        coarse_hp = highpass(coarse, self.mask_c, self.sigma_c)
        return Prepared(work=work, coarse=coarse_hp, sharpness=sharp)

    def work_hp(self, p: Prepared) -> np.ndarray:
        if p._work_hp is not None:
            return p._work_hp
        return highpass(p.work, self.mask_w, self.sigma_w)

    def _spectra(self, p: Prepared):
        if p._spectra is None:
            p._spectra = self.ncc.spectra(p.coarse)
        return p._spectra

    # --------------------------------------------------------------- coarse
    def coarse_match(self, a: Prepared, b: Prepared,
                     pred: Optional[Tuple[float, float]] = None,
                     radius: Optional[float] = None,
                     strict: bool = False) -> Optional[CoarseMatch]:
        st = self.settings
        ncc = self.ncc(self._spectra(a), self._spectra(b))
        return self._peak(ncc, self.ncc.valid, self.ncc.overlap, self.ncc.area,
                          pred, radius, strict)

    def _peak(self, ncc, valid, overlap_map, area, pred, radius,
              strict) -> Optional[CoarseMatch]:
        st = self.settings
        score = np.where(valid, ncc, -np.inf)
        yy, xx = self._grid
        if pred is not None and radius is not None:
            region = (yy - pred[1]) ** 2 + (xx - pred[0]) ** 2 <= radius * radius
            score = np.where(region, score, -np.inf)
        k = int(np.argmax(score))
        py, px = np.unravel_index(k, score.shape)
        peak = float(score[py, px])
        if not np.isfinite(peak):
            return None
        excl = max(3.0, 0.06 * self.Dc)
        sy0, sx0 = py - (self.ncc.h - 1), px - (self.ncc.w - 1)
        far = (yy - sy0) ** 2 + (xx - sx0) ** 2 > excl * excl
        rest = np.where(far, score, -np.inf)
        second = float(np.max(rest))
        if not np.isfinite(second):
            second = 0.0
        second = max(second, 0.0)
        # sub-pixel parabolic refinement
        dx = dy = 0.0
        if 0 < px < ncc.shape[1] - 1:
            l, c, r = ncc[py, px - 1], ncc[py, px], ncc[py, px + 1]
            den = l - 2 * c + r
            if den < 0:
                dx = float(np.clip(0.5 * (l - r) / den, -0.5, 0.5))
        if 0 < py < ncc.shape[0] - 1:
            u, c, d = ncc[py - 1, px], ncc[py, px], ncc[py + 1, px]
            den = u - 2 * c + d
            if den < 0:
                dy = float(np.clip(0.5 * (u - d) / den, -0.5, 0.5))
        overlap = float(overlap_map[py, px]) / area
        m = CoarseMatch(sx0 + dx, sy0 + dy, peak, second, overlap)
        min_ncc = st.min_ncc * (1.5 if strict else 1.0)
        max_ratio = st.max_peak_ratio * (0.9 if strict else 1.0)
        if peak < min_ncc:
            return None
        if second > max_ratio * peak:
            return None
        return m

    # ----------------------------------------------------------------- fine
    def refine(self, a: Prepared, b: Prepared, cm: CoarseMatch,
               ha: Optional[np.ndarray] = None, hb: Optional[np.ndarray] = None,
               mask_a: Optional[np.ndarray] = None,
               mask_b: Optional[np.ndarray] = None) -> Optional[RelPose]:
        """ECC alignment.  By default the whole field of view is used; the
        cross-segment linking passes specimen-only images and masks."""
        st = self.settings
        if ha is None:
            ha = self.work_hp(a)
        if hb is None:
            hb = self.work_hp(b)
        if mask_a is None:
            mask_a = self.mask_w
        if mask_b is None:
            mask_b = self.mask_w
        tw0 = np.array([-cm.sx * self.f_cw, -cm.sy * self.f_cw])  # p_B = p_A + tw
        W = np.array([[1, 0, tw0[0]], [0, 1, tw0[1]]], np.float32)
        h, w = self.mask_w.shape
        rot_allowed = st.max_rotation_deg > 0
        # valid input pixels: inside B's circle AND inside A's circle mapped
        # into B coordinates (with a safety erosion).
        ma = cv2.warpAffine(mask_a, W, (w, h), flags=cv2.INTER_NEAREST)
        erode = 3 + int(math.ceil(math.radians(st.max_rotation_deg) * self.D * self.s_work / 2))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode + 1, 2 * erode + 1))
        mask_in = cv2.erode(ma & mask_b, k)
        ov = float((mask_in > 0).sum()) / self.area_w
        if ov < st.min_overlap_area * 0.7:
            return None
        motion = cv2.MOTION_EUCLIDEAN if rot_allowed else cv2.MOTION_TRANSLATION
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5)
        try:
            rho, W = cv2.findTransformECC(ha, hb, W, motion, crit, mask_in, 3)
        except cv2.error:
            return None
        if not np.all(np.isfinite(W)):
            return None
        Rw = W[:, :2].astype(np.float64)
        tw = W[:, 2].astype(np.float64)
        ang = math.atan2(Rw[1, 0], Rw[0, 0])
        if rot_allowed and abs(math.degrees(ang)) > st.max_rotation_deg:
            return None
        # ECC must stay close to the coarse estimate, otherwise it diverged.
        # Compare where both warps send the centre of the overlap region
        # (with rotation the translation part alone depends on the origin).
        ys, xs = np.nonzero(mask_in)
        p_a = np.array([xs.mean(), ys.mean()]) - tw0
        if np.linalg.norm(Rw @ p_a + tw - (p_a + tw0)) > max(4.0, 2.5 * self.f_cw):
            return None
        if rho < st.min_quality:
            return None
        e = Rw @ self.c_work + tw - self.c_work
        RwT = Rw.T
        d = -RwT @ e / self.s_work
        return RelPose(float(d[0]), float(d[1]), -ang, float(rho), cm.peak, ov)

    # ------------------------------------------------- cross-segment linking
    def link_data(self, p: Prepared, cache: bool = True) -> LinkData:
        """Specimen mask: large dark regions inside the field of view are the
        background around the section; they are removed together with a band
        of `link_edge_band` x diameter along their boundary.  Small dark spots
        (pores, opaque grains) stay part of the specimen."""
        if p._link is not None:
            return p._link
        st = self.settings
        fov = self.mask_w > 0
        g = cv2.GaussianBlur(p.work, (0, 0), 2.0)
        vals = g[fov]
        thr = max(st.link_dark_level, 0.3 * float(np.percentile(vals, 95)) if vals.size else 0)
        dark = ((g < thr) & fov).astype(np.uint8)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(dark)
        big = stats[:, cv2.CC_STAT_AREA] > 0.01 * fov.sum()
        big[0] = False
        bg = big[lab]
        content = (fov & ~bg).astype(np.uint8) * 255
        band = max(2, int(round(st.link_edge_band * self.D * self.s_work)))
        kb = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
        core = content & ~cv2.dilate(bg.astype(np.uint8) * 255, kb)
        core = cv2.erode(core, np.ones((5, 5), np.uint8))   # off the FOV rim too
        frac = float((core > 0).sum()) / max(float(fov.sum()), 1.0)

        def hp_on(gray, cont, cor, sigma):
            h = highpass(gray, cont, sigma) * (cor > 0)
            sd = float(np.sqrt((h ** 2).sum() / max((cor > 0).sum(), 1)))
            return (h / sd if sd > 0 else h).astype(np.float32)

        work_hp = hp_on(p.work, content, core, self.sigma_w)
        f = self.s_coarse / self.s_work
        gc = resize_exact(p.work, f, self.coarse_size)
        cont_c = (resize_exact(content, f, self.coarse_size) > 127).astype(np.uint8) * 255
        core_c = (resize_exact(core, f, self.coarse_size) > 250).astype(np.uint8) * 255
        hc = hp_on(gc, cont_c, core_c, self.sigma_c)
        mc = (core_c > 0).astype(np.float32)
        sh = self.ncc.shape
        spectra = (sfft.rfft2(hc * mc, s=sh), sfft.rfft2(hc * hc * mc, s=sh),
                   sfft.rfft2(mc, s=sh))
        ld = LinkData(core, work_hp, spectra, frac)
        if cache:
            p._link = ld
        return ld

    def link_coarse(self, la: LinkData, lb: LinkData) -> Optional[CoarseMatch]:
        """Global specimen-only correlation search with strict thresholds."""
        st = self.settings
        if min(la.frac, lb.frac) < st.link_min_content:
            return None
        ncc, valid, ov = _masked_ncc_general(self.ncc, la.spectra, lb.spectra,
                                             st.min_overlap_area * self.ncc.area)
        return self._peak(ncc, valid, ov, self.ncc.area, None, None, strict=True)

    def link_refine(self, a: Prepared, b: Prepared, cm: CoarseMatch) -> Optional[RelPose]:
        la, lb = self.link_data(a, cache=False), self.link_data(b, cache=False)
        return self.refine(a, b, cm, la.work_hp, lb.work_hp, la.core_w, lb.core_w)

    # ---------------------------------------------------------------- utils
    def register(self, a: Prepared, b: Prepared,
                 pred: Optional[RelPose] = None, radius_frac: Optional[float] = None,
                 strict: bool = False) -> Optional[RelPose]:
        """Full registration of b relative to a; returns pose of b in a."""
        p = r = None
        if pred is not None and radius_frac is not None:
            p = self.pose_to_coarse_shift(pred)
            r = radius_frac * self.Dc
        cm = self.coarse_match(a, b, p, r, strict=strict)
        if cm is None:
            return None
        return self.refine(a, b, cm)

    def pose_to_coarse_shift(self, pose: RelPose) -> Tuple[float, float]:
        # translation-only approximation: p_A = p_B + d * s
        return pose.dx * self.s_coarse, pose.dy * self.s_coarse

    def coarse_shift_to_full(self, cm: CoarseMatch) -> Tuple[float, float]:
        return cm.sx / self.s_coarse, cm.sy / self.s_coarse
