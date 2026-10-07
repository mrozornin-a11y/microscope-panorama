"""Streaming keyframe selection with sequential registration."""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import cv2
import numpy as np

from .config import Settings
from .fov import FieldOfView
from .registration import Prepared, Registrar, RelPose
from .video import VideoSource


@dataclass
class KeyFrame:
    id: int
    frame_index: int
    time: float
    segment: int
    sharpness: float
    prep: Prepared
    crop_path: str
    thumb: np.ndarray            # small colour image (for flat-field)


@dataclass
class Edge:
    i: int
    j: int
    pose: RelPose                # x_i = R(phi) x_j + d
    kind: str = "seq"           # 'seq' | 'loop' | 'link'


@dataclass
class TrackStats:
    analysed: int = 0
    blurred: int = 0
    duplicates: int = 0
    failed: int = 0
    segments: int = 1


@dataclass
class _Candidate:
    frame_index: int
    crop: np.ndarray
    prep: Prepared
    shift_c: tuple               # coarse shift relative to current keyframe
    peak: float


class KeyframeSelector:
    def __init__(self, video: VideoSource, fov: FieldOfView, settings: Settings,
                 registrar: Registrar, cache_dir: str):
        self.video = video
        self.fov = fov
        self.st = settings
        self.reg = registrar
        self.cache_dir = cache_dir
        self.keyframes: List[KeyFrame] = []
        self.edges: List[Edge] = []
        self.stats = TrackStats()
        self.thumb_scale = min(1.0, 160.0 / fov.diameter)
        self.thumb_size = fov.scaled_size(self.thumb_scale)

    # ------------------------------------------------------------------
    def _add_keyframe(self, frame_index: int, crop: np.ndarray, prep: Prepared,
                      segment: int) -> KeyFrame:
        kid = len(self.keyframes)
        path = os.path.join(self.cache_dir, f"kf_{kid:05d}.png")
        cv2.imwrite(path, crop, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        thumb = cv2.resize(crop, self.thumb_size, interpolation=cv2.INTER_AREA)
        kf = KeyFrame(kid, frame_index, frame_index / self.video.fps, segment,
                      prep.sharpness, prep, path, thumb)
        self.keyframes.append(kf)
        return kf

    def run(self, progress: Optional[Callable[[float, str], None]] = None,
            cancel: Optional[Callable[[], bool]] = None):
        st, reg = self.st, self.reg
        step = max(1, int(round(self.video.fps / max(st.analysis_fps, 0.1))))
        total = max(self.video.frame_count, 1)
        min_shift_c = st.keyframe_min_shift * reg.Dc
        window = deque(maxlen=31)
        kf: Optional[KeyFrame] = None
        cand: Optional[_Candidate] = None
        last_shift = None          # last coarse shift relative to kf
        velocity = np.zeros(2)     # coarse px per analysed frame
        segment = 0
        lost = 0

        for fidx, frame in self.video.iterate(step):
            if cancel and cancel():
                raise InterruptedError
            self.stats.analysed += 1
            if progress and self.stats.analysed % 5 == 0:
                progress(min(fidx / total, 1.0),
                         f"Frame {fidx}/{total}: {len(self.keyframes)} keyframes")
            crop = np.ascontiguousarray(self.fov.crop(frame))
            prep = reg.prepare(crop)
            window.append(prep.sharpness)
            med = float(np.median(window))
            if prep.sharpness < st.min_sharpness * med or prep.sharpness < st.min_sharpness_abs:
                self.stats.blurred += 1
                continue

            if kf is None:
                kf = self._add_keyframe(fidx, crop, prep, segment)
                last_shift, cand, lost = np.zeros(2), None, 0
                continue

            pred = None if last_shift is None else last_shift + velocity * step
            cm = self._track(kf.prep, prep, pred)

            if cm is None and cand is not None:
                # Lost the current keyframe: promote the last frame that still
                # registered to it, and try again from there.
                promoted = self._promote(kf, cand)
                cand = None
                if promoted is not None:
                    kf = promoted
                    last_shift = np.zeros(2)
                    cm = self._track(kf.prep, prep, None)

            if cm is None:
                lost += 1
                self.stats.failed += 1
                if lost >= st.lost_patience:
                    segment += 1
                    self.stats.segments = segment + 1
                    kf = self._add_keyframe(fidx, crop, prep, segment)
                    last_shift, velocity, cand, lost = np.zeros(2), np.zeros(2), None, 0
                continue
            lost = 0

            shift = np.array([cm.sx, cm.sy])
            if last_shift is not None:
                v = (shift - last_shift) / step
                velocity = 0.5 * velocity + 0.5 * v
            last_shift = shift

            if np.hypot(*shift) >= min_shift_c:
                rel = reg.refine(kf.prep, prep, cm)
                if rel is not None:
                    new = self._add_keyframe(fidx, crop, prep, segment)
                    self.edges.append(Edge(kf.id, new.id, rel, "seq"))
                    kf.prep._spectra = None      # free memory
                    kf, cand = new, None
                    last_shift = np.zeros(2)
                    continue
                # fine alignment failed - keep the previous candidate
                continue
            self.stats.duplicates += 1
            cand = _Candidate(fidx, crop, prep, tuple(shift), cm.peak)

        if progress:
            progress(1.0, f"{len(self.keyframes)} keyframes selected")
        return self.keyframes, self.edges

    # ------------------------------------------------------------------
    def _track(self, a: Prepared, b: Prepared, pred):
        reg = self.reg
        if pred is not None:
            cm = reg.coarse_match(a, b, (pred[0], pred[1]), 0.25 * reg.Dc)
            if cm is not None:
                return cm
        return reg.coarse_match(a, b, None, None, strict=pred is not None)

    def _promote(self, kf: KeyFrame, cand: _Candidate) -> Optional[KeyFrame]:
        if np.hypot(*cand.shift_c) < 0.02 * self.reg.Dc:
            return None
        cm = self.reg.coarse_match(kf.prep, cand.prep,
                                   cand.shift_c, 0.05 * self.reg.Dc)
        if cm is None:
            return None
        rel = self.reg.refine(kf.prep, cand.prep, cm)
        if rel is None:
            return None
        new = self._add_keyframe(cand.frame_index, cand.crop, cand.prep, kf.segment)
        self.edges.append(Edge(kf.id, new.id, rel, "seq"))
        return new
