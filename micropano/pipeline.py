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

            # orientation: rotate everything so that the mean frame rotation is 0
            poses = graph.poses.copy()
            if graph.use_rot:
                mean_th = float(np.mean(poses[used, 2]))
                c, s_ = math.cos(-mean_th), math.sin(-mean_th)
                R = np.array([[c, -s_], [s_, c]])
                poses[used, :2] = poses[used, :2] @ R.T
                poses[used, 2] -= mean_th

            gain = None
            if st.flat_field:
                gain = estimate_flat_field([kfs[i].thumb for i in used], fov,
                                           sel.thumb_scale, fov.crop_size)
                self._say("flat-field correction: " + ("applied" if gain is not None
                                                        else "skipped (too few keyframes)"))

            frames = [RenderFrame(kfs[i].crop_path, poses[i, 0], poses[i, 1], poses[i, 2])
                      for i in used]
            renderer = MosaicRenderer(fov, frames, st, gain)
            tiff = os.path.join(self.out_dir, "mosaic.tif")
            prev = os.path.join(self.out_dir, "preview.jpg")
            self._say(f"mosaic size: {renderer.width} x {renderer.height} px")
            preview = renderer.render(tiff, prev, self._stage("render"), self.cancel)

            csv_path = os.path.join(self.out_dir, "frame_positions.csv")
            self._write_csv(csv_path, kfs, graph, poses, renderer)
            self._say(f"done in {time.time() - t0:.1f} s")
            with open(os.path.join(self.out_dir, "log.txt"), "w", encoding="utf-8") as fh:
                fh.write("\n".join(self.log) + "\n\nsettings:\n" + st.to_json() + "\n")
            return Result(tiff, prev, csv_path, renderer.width, renderer.height,
                          len(kfs), len(used), preview, list(self.log),
                          list(graph.warnings))
        finally:
            if not st.keep_cache and not st.cache_dir:
                shutil.rmtree(cache, ignore_errors=True)

    def _write_csv(self, path, kfs, graph: PoseGraph, poses, renderer: MosaicRenderer):
        q, cnt = graph.frame_quality()
        ox, oy = renderer.origin
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["keyframe", "frame", "time_s", "x", "y", "rotation_deg",
                        "quality", "n_links", "sharpness", "status"])
            for k in kfs:
                ok = bool(graph.active[k.id])
                if ok:
                    x, y, th = poses[k.id]
                    w.writerow([k.id, k.frame_index, f"{k.time:.3f}", f"{x - ox:.2f}",
                                f"{y - oy:.2f}", f"{math.degrees(th):.4f}",
                                f"{q[k.id]:.4f}", int(cnt[k.id]), f"{k.sharpness:.1f}", "used"])
                else:
                    w.writerow([k.id, k.frame_index, f"{k.time:.3f}", "", "", "",
                                f"{q[k.id]:.4f}", int(cnt[k.id]), f"{k.sharpness:.1f}",
                                "dropped"])
