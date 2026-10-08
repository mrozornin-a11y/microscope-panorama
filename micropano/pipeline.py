"""End-to-end processing: video -> keyframes -> pose graph -> mosaic."""

from __future__ import annotations

import csv
import math
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from .config import Settings
from .fov import FieldOfView, detect_fov
from .graph import PoseGraph
from .keyframes import KeyframeSelector
from .registration import Registrar
from .render import MosaicRenderer, RenderFrame, estimate_flat_field
from .video import VideoSource

ProgressFn = Callable[[float, str], None]

# share of the total progress bar taken by each stage
_STAGES = {"keyframes": (0.0, 0.45), "graph": (0.45, 0.6), "render": (0.6, 1.0)}


@dataclass
class Result:
    tiff_path: str
    preview_path: str
    csv_path: str
    width: int
    height: int
    n_keyframes: int
    n_used: int
    preview: Optional[np.ndarray] = None
    log: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    components: List["Component"] = field(default_factory=list)


@dataclass
class Component:
    """A group of keyframes that could not be linked to the main mosaic and
    was rendered separately."""
    index: int
    tiff_path: str
    preview_path: str
    n_keyframes: int
    first_frame: int
    last_frame: int
    width: int
    height: int
    reason: str
    preview: Optional[np.ndarray] = None


class Pipeline:
    def __init__(self, video_path: str, out_dir: str, settings: Optional[Settings] = None,
                 fov: Optional[FieldOfView] = None,
                 progress: Optional[ProgressFn] = None,
                 cancel: Optional[Callable[[], bool]] = None):
        self.video_path = video_path
        self.out_dir = out_dir
        self.st = settings or Settings()
        self.fov = fov
        self._progress = progress
        self.cancel = cancel
        self.log: List[str] = []
        self.warnings: List[str] = []

    def _say(self, msg: str):
        self.log.append(msg)

    def _stage(self, name: str) -> ProgressFn:
        a, b = _STAGES[name]

        def f(frac: float, msg: str):
            if self._progress:
                self._progress(a + (b - a) * max(0.0, min(1.0, frac)), msg)
        return f

    def run(self) -> Result:
        st = self.st
        t0 = time.time()
        os.makedirs(self.out_dir, exist_ok=True)
        video = VideoSource(self.video_path)
        self._say(f"video: {video.width}x{video.height}, {video.fps:.2f} fps, "
                  f"{video.frame_count} frames")
        if self.fov is None:
            self.fov = detect_fov(video.sample_frames(15), st.mask_margin)
        else:
            self.fov.margin = st.mask_margin
        fov = self.fov
        self._say(f"field of view: centre=({fov.cx:.1f}, {fov.cy:.1f}) r={fov.radius:.1f} px")

        if st.cache_dir:
            cache = st.cache_dir
            os.makedirs(cache, exist_ok=True)
        else:
            cache = tempfile.mkdtemp(prefix="micropano_")
        try:
            reg = Registrar(fov, st)
            sel = KeyframeSelector(video, fov, st, reg, cache)
            kfs, edges = sel.run(self._stage("keyframes"), self.cancel)
            t1 = time.time()
            s = sel.stats
            self._say(f"analysed {s.analysed} frames: {len(kfs)} keyframes, "
                      f"{s.blurred} blurred, {s.duplicates} near-duplicates, "
                      f"{s.failed} unregistered, {s.segments} tracking segment(s) "
                      f"[{t1 - t0:.1f} s]")
            if not kfs:
                raise RuntimeError("No usable frames found (all frames blurred?)")

            graph = PoseGraph(kfs, edges, reg, st)
            graph.run(self._stage("graph"), self.cancel)
            self.log.extend(graph.log)
            self.warnings = list(graph.warnings)
            for w in graph.warnings:
                self._say("WARNING: " + w)
            if st.link_diagnostics and graph.link_records:
                from .diagnostics import write_link_diagnostics
                p = write_link_diagnostics(self.out_dir, graph.link_records, kfs, reg)
                self._say(f"cross-segment link diagnostics: {p}")
            t2 = time.time()
            self._say(f"pose graph: {len(graph.edges)} edges [{t2 - t1:.1f} s]")
            used = np.flatnonzero(graph.active)
            if len(used) == 0:
                raise RuntimeError("Registration failed for all frames")

            # orientation: rotate each component so that its mean frame
            # rotation is 0
            poses = graph.poses.copy()
            n_comp = int(graph.comp.max()) if graph.kept.any() else 0
            if graph.use_rot:
                for c in range(n_comp + 1):
                    ids = np.flatnonzero(graph.comp == c)
                    mean_th = float(np.mean(poses[ids, 2]))
                    c_, s_ = math.cos(-mean_th), math.sin(-mean_th)
                    R = np.array([[c_, -s_], [s_, c_]])
                    poses[ids, :2] = poses[ids, :2] @ R.T
                    poses[ids, 2] -= mean_th

            gain = None
            if st.flat_field:
                kept = np.flatnonzero(graph.kept)
                gain = estimate_flat_field([kfs[i].thumb for i in kept], fov,
                                           sel.thumb_scale, fov.crop_size)
                self._say("flat-field correction: " + ("applied" if gain is not None
                                                        else "skipped (too few keyframes)"))

            # renderers: main mosaic + one per unlinked component
            jobs = []
            for c in range(n_comp + 1):
                ids = np.flatnonzero(graph.comp == c)
                frames = [RenderFrame(kfs[i].crop_path, poses[i, 0], poses[i, 1], poses[i, 2])
                          for i in ids]
                r = MosaicRenderer(fov, frames, st, gain)
                if c == 0:
                    tiff = os.path.join(self.out_dir, "mosaic.tif")
                    prev = os.path.join(self.out_dir, "preview.jpg")
                else:
                    d = os.path.join(self.out_dir, "unlinked")
                    os.makedirs(d, exist_ok=True)
                    tiff = os.path.join(d, f"component_{c:02d}.tif")
                    prev = os.path.join(d, f"component_{c:02d}_preview.jpg")
                jobs.append((c, ids, r, tiff, prev))
            areas = np.array([float(r.width) * r.height for _, _, r, _, _ in jobs])
            bounds = np.concatenate([[0], np.cumsum(areas) / areas.sum()])
            a0, a1 = _STAGES["render"]
            renderers = {}
            comps = []
            preview = None
            for k, (c, ids, r, tiff, prev) in enumerate(jobs):
                lo, hi = bounds[k], bounds[k + 1]
                name = "mosaic" if c == 0 else f"unlinked component {c}"

                def prog(f, m, lo=lo, hi=hi, name=name):
                    if self._progress:
                        self._progress(a0 + (a1 - a0) * (lo + (hi - lo) * f), f"{name}: {m}")
                self._say(f"{name}: {len(ids)} keyframes, {r.width} x {r.height} px -> "
                          f"{os.path.relpath(tiff, self.out_dir)}")
                pv = r.render(tiff, prev, prog, self.cancel)
                renderers[c] = r
                if c == 0:
                    preview, main_r = pv, r
                else:
                    f0, f1 = kfs[ids[0]].frame_index, kfs[ids[-1]].frame_index
                    reason = graph.unlinked_reasons.get(tuple(int(i) for i in ids), "")
                    comps.append(Component(c, tiff, prev, len(ids), f0, f1,
                                           r.width, r.height, reason, pv))
                    self.warnings.append(
                        f"unlinked component {c} ({len(ids)} keyframes, video frames "
                        f"{f0}-{f1}) saved separately: {os.path.relpath(tiff, self.out_dir)}")

            csv_path = os.path.join(self.out_dir, "frame_positions.csv")
            self._write_csv(csv_path, kfs, graph, poses, renderers)
            for w in self.warnings[len(graph.warnings):]:
                self._say("WARNING: " + w)
            self._say(f"done in {time.time() - t0:.1f} s")
            with open(os.path.join(self.out_dir, "log.txt"), "w", encoding="utf-8") as fh:
                fh.write("\n".join(self.log) + "\n\nsettings:\n" + st.to_json() + "\n")
            return Result(jobs[0][3], jobs[0][4], csv_path, main_r.width, main_r.height,
                          len(kfs), len(used), preview, list(self.log),
                          list(self.warnings), comps)
        finally:
            if not st.keep_cache and not st.cache_dir:
                shutil.rmtree(cache, ignore_errors=True)

    def _write_csv(self, path, kfs, graph: PoseGraph, poses, renderers):
        """x, y are pixel coordinates of the frame centre in the TIFF of the
        frame's component (0 = mosaic.tif, k = unlinked/component_k.tif)."""
        q, cnt = graph.frame_quality()
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["keyframe", "frame", "time_s", "x", "y", "rotation_deg",
                        "quality", "n_links", "sharpness", "segment", "component", "status"])
            for k in kfs:
                c = int(graph.comp[k.id])
                head = [k.id, k.frame_index, f"{k.time:.3f}"]
                tail = [f"{q[k.id]:.4f}", int(cnt[k.id]), f"{k.sharpness:.1f}", k.segment]
                if c >= 0:
                    ox, oy = renderers[c].origin
                    x, y, th = poses[k.id]
                    w.writerow(head + [f"{x - ox:.2f}", f"{y - oy:.2f}",
                                       f"{math.degrees(th):.4f}"] + tail +
                               [c, "used" if c == 0 else "unlinked"])
                else:
                    w.writerow(head + ["", "", ""] + tail + ["", "dropped"])
