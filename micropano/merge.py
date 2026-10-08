"""Manual attachment of unlinked components to the main mosaic.

After processing, if some components could not be linked automatically, the
pipeline stores everything needed to re-render (keyframes, poses, field of
view, settings, flat field) in ``<out>/merge_data``.  A :class:`MergeSession`
then lets the user

* propose a rough rigid transform component -> main (dragging/rotating in
  the GUI, or from 2-3 corresponding points),
* refine it by local specimen-only NCC/ECC registration of predicted frame
  pairs in a limited window, with a quality report,
* merge the component and re-render ``mosaic_merged.tif`` (the original
  ``mosaic.tif`` and unlinked TIFFs are never overwritten).

Transforms are (tx, ty, theta): X_main = R(theta) X_comp + (tx, ty), in
full-resolution world pixels of the respective pose frames.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import Settings
from .fov import FieldOfView
from .registration import Registrar, RelPose
from .render import MosaicRenderer, RenderFrame

STATE_DIR = "merge_data"
STATE_FILE = "state.json"


def _rot(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s], [s, c]])


def apply_T(T: Sequence[float], pose: Sequence[float]) -> np.ndarray:
    """Pose of a component frame expressed in main-mosaic world coordinates."""
    tx, ty, th = T
    p = _rot(th) @ np.asarray(pose[:2], float) + [tx, ty]
    return np.array([p[0], p[1], pose[2] + th])


def transform_from_points(main_pts: Sequence[Sequence[float]],
                          comp_pts: Sequence[Sequence[float]]) -> Tuple[np.ndarray, float]:
    """Rigid transform (no scale) comp -> main from corresponding points.
    One pair gives a pure translation.  Returns (T, rms residual in px)."""
    a = np.asarray(comp_pts, float)
    b = np.asarray(main_pts, float)
    if len(a) == 0 or len(a) != len(b):
        raise ValueError("need the same number (>= 1) of points on both images")
    ca, cb = a.mean(0), b.mean(0)
    if len(a) == 1:
        th = 0.0
    else:
        u, v = a - ca, b - cb
        th = math.atan2(float((u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]).sum()),
                        float((u * v).sum()))
    t = cb - _rot(th) @ ca
    res = (a @ _rot(th).T + t) - b
    return np.array([t[0], t[1], th]), float(np.sqrt((res ** 2).sum(1).mean()))


# --------------------------------------------------------------------------
def save_merge_state(out_dir: str, video_path: str, fov: FieldOfView, st: Settings,
                     kfs, comp: np.ndarray, poses: np.ndarray, quality: np.ndarray,
                     renderers: Dict[int, MosaicRenderer], outputs: Dict[int, Tuple[str, str]],
                     previews: Dict[int, np.ndarray], gain: Optional[np.ndarray],
                     reasons: Dict[int, str]) -> str:
    """Persist keyframes and poses so that components can be attached later.
    Keyframe images are moved out of the temporary cache."""
    d = os.path.join(out_dir, STATE_DIR)
    kd = os.path.join(d, "keyframes")
    os.makedirs(kd, exist_ok=True)
    frames = []
    for k in kfs:
        c = int(comp[k.id])
        path = ""
        if c >= 0:
            path = os.path.join("keyframes", os.path.basename(k.crop_path))
            dst = os.path.join(d, path)
            if os.path.abspath(k.crop_path) != os.path.abspath(dst):
                shutil.copy2(k.crop_path, dst)
        x, y, th = (poses[k.id] if c >= 0 else (None, None, None))
        frames.append(dict(id=k.id, frame=k.frame_index, time=k.time, segment=k.segment,
                           comp=c, x=x, y=y, theta=th, path=path,
                           quality=float(quality[k.id]), sharpness=float(k.sharpness)))
    if gain is not None:
        gh, gw = max(2, gain.shape[0] // 8), max(2, gain.shape[1] // 8)
        np.save(os.path.join(d, "gain.npy"),
                cv2.resize(gain, (gw, gh), interpolation=cv2.INTER_AREA))
    outs = {}
    for c, r in renderers.items():
        tiff, prev = outputs[c]
        ph, pw = previews[c].shape[:2]
        outs[str(c)] = dict(tiff=os.path.relpath(tiff, out_dir),
                            preview=os.path.relpath(prev, out_dir),
                            origin=[float(r.origin[0]), float(r.origin[1])],
                            width=r.width, height=r.height, preview_w=pw, preview_h=ph)
    comps = {str(c): dict(status="main" if c == 0 else "unlinked",
                          reason=reasons.get(c, "")) for c in renderers}
    state = dict(version=1, video=os.path.abspath(video_path), fov=fov.to_dict(),
                 settings=json.loads(st.to_json()), frames=frames, outputs=outs,
                 components=comps, main_output="0", merges=[])
    with open(os.path.join(d, STATE_FILE), "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=1)
    return d


# --------------------------------------------------------------------------
@dataclass
class PairMatch:
    m: int
    s: int
    rel: Optional[RelPose]
    T: Optional[np.ndarray]          # transform implied by this match
    residual: float = float("nan")   # px, w.r.t. the final transform
    consistent: bool = False


@dataclass
class AlignResult:
    T: np.ndarray
    confirmed: bool
    n_tested: int
    n_matched: int
    n_consistent: int
    mean_quality: float
    rms_residual: float
    shift_from_initial: float        # px, how far refinement moved the component
    pairs: List[PairMatch] = field(default_factory=list)
    message: str = ""

    def summary(self) -> str:
        if self.n_matched == 0:
            return self.message
        return (f"{self.n_consistent} of {self.n_tested} frame pairs confirm the position "
                f"({self.n_matched} matched), mean ECC {self.mean_quality:.3f}, "
                f"RMS residual {self.rms_residual:.1f} px, moved "
                f"{self.shift_from_initial:.0f} px from the initial position. "
                + self.message)


@dataclass
class MergeOutput:
    tiff_path: str
    preview_path: str
    csv_path: str
    width: int
    height: int
    preview: np.ndarray


class MergeSession:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.dir = os.path.join(out_dir, STATE_DIR)
        with open(os.path.join(self.dir, STATE_FILE), encoding="utf-8") as fh:
            self.state = json.load(fh)
        self.st = Settings.from_dict(self.state["settings"])
        f = self.state["fov"]
        self.fov = FieldOfView(f["cx"], f["cy"], f["radius"], f["frame_w"], f["frame_h"],
                               f["margin"])
        self.reg = Registrar(self.fov, self.st)
        self.D = self.reg.D
        self.frames = {fr["id"]: fr for fr in self.state["frames"]}
        self._prep: Dict[int, object] = {}
        gp = os.path.join(self.dir, "gain.npy")
        self.gain = None
        if os.path.exists(gp):
            W, H = self.fov.crop_size
            self.gain = cv2.resize(np.load(gp), (W, H), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def available(out_dir: str) -> bool:
        return os.path.exists(os.path.join(out_dir, STATE_DIR, STATE_FILE))

    # ------------------------------------------------------------ queries
    def comp_status(self, c: int) -> dict:
        return self.state["components"][str(c)]

    def unlinked(self) -> List[int]:
        return sorted(int(c) for c, v in self.state["components"].items()
                      if v["status"] == "unlinked")

    def comp_ids(self, c: int) -> List[int]:
        return [i for i, fr in self.frames.items() if fr["comp"] == c]

    def _merged_T(self, c: int) -> Optional[np.ndarray]:
        v = self.comp_status(c)
        if c == 0:
            return np.zeros(3)
        if v["status"] == "merged":
            return np.array(v["T"], float)
        return None

    def main_ids(self) -> List[int]:
        return [i for i, fr in self.frames.items()
                if fr["comp"] >= 0 and self._merged_T(fr["comp"]) is not None]

    def pose(self, i: int) -> np.ndarray:
        fr = self.frames[i]
        return np.array([fr["x"], fr["y"], fr["theta"]], float)

    def world_pose(self, i: int) -> np.ndarray:
        """Pose in main-mosaic world coordinates (for main / merged frames)."""
        return apply_T(self._merged_T(self.frames[i]["comp"]), self.pose(i))

    def output(self, c: Optional[int] = None) -> dict:
        """Output info of the main mosaic (c None) or of a component."""
        key = self.state["main_output"] if c is None else str(c)
        return self.state["outputs"][key]

    def preview_image(self, c: Optional[int] = None) -> np.ndarray:
        o = self.output(c)
        img = cv2.imread(os.path.join(self.out_dir, o["preview"]), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"cannot read {o['preview']}")
        return img

    @staticmethod
    def preview_scale(o: dict) -> float:
        return o["preview_w"] / o["width"]

    def frame_info(self, c: int) -> str:
        ids = self.comp_ids(c)
        f0 = min(self.frames[i]["frame"] for i in ids)
        f1 = max(self.frames[i]["frame"] for i in ids)
        return f"{len(ids)} keyframes, video frames {f0}-{f1}"

    # ------------------------------------------------------ registration
    def _prepared(self, i: int):
        p = self._prep.get(i)
        if p is None:
            img = cv2.imread(os.path.join(self.dir, self.frames[i]["path"]), cv2.IMREAD_COLOR)
            if img is None:
                raise IOError(f"missing keyframe image for keyframe {i}")
            p = self.reg.prepare(img)
            if len(self._prep) > 80:
                self._prep.pop(next(iter(self._prep)))
            self._prep[i] = p
        return p

    def keyframe(self, i: int):
        """Light object with `.prep` (for diagnostics.link_preview)."""
        class _K:
            pass
        k = _K()
        k.prep = self._prepared(i)
        k.id = i
        return k

    def _select_pairs(self, c: int, T: np.ndarray, max_pairs: int, dmax: float):
        main = self.main_ids()
        if not main:
            return []
        mpos = np.array([self.world_pose(m)[:2] for m in main])
        comp = self.comp_ids(c)
        cand = []
        for s in comp:
            ps = apply_T(T, self.pose(s))[:2]
            d = np.linalg.norm(mpos - ps, axis=1)
            k = int(np.argmin(d))
            if d[k] < dmax:
                cand.append((float(d[k]), main[k], s))
        cand.sort()
        pairs, used_m, used_s = [], set(), set()
        for d, m, s in cand:
            if m in used_m or s in used_s:
                continue
            pairs.append((m, s))
            used_m.add(m)
            used_s.add(s)
            if len(pairs) >= max_pairs:
                break
        return pairs

    def _match(self, c: int, T: np.ndarray, pairs, radius_frac: float) -> List[PairMatch]:
        out = []
        for m, s in pairs:
            Pm, Ps = self.world_pose(m), self.pose(s)
            ps = apply_T(T, Ps)
            d = _rot(Pm[2]).T @ (ps[:2] - Pm[:2])
            pred = RelPose(float(d[0]), float(d[1]), float(ps[2] - Pm[2]))
            cm, rel = self.reg.link_local(self._prepared(m), self._prepared(s), pred,
                                          radius_frac)
            Tk = None
            if rel is not None:
                th = Pm[2] + rel.phi
                t = Pm[:2] + _rot(Pm[2]) @ np.array([rel.dx, rel.dy])
                thT = th - Ps[2]
                tT = t - _rot(thT) @ Ps[:2]
                Tk = np.array([tT[0], tT[1], thT])
            out.append(PairMatch(m, s, rel, Tk))
        return out

    def _centre(self, c: int) -> np.ndarray:
        return np.mean([self.pose(s)[:2] for s in self.comp_ids(c)], axis=0)

    def _dist(self, c: int, T1, T2) -> float:
        ctr = self._centre(c)
        return float(np.linalg.norm(apply_T(T1, [*ctr, 0])[:2] - apply_T(T2, [*ctr, 0])[:2]))

    def refine(self, c: int, T0: Sequence[float], radius_frac: float = 0.3,
               max_pairs: int = 12) -> AlignResult:
        """Local automatic registration around the manual position T0."""
        st = self.st
        T0 = np.asarray(T0, float)
        tol = st.link_tolerance * self.D
        rot_tol = math.radians(max(1.0, st.max_rotation_deg, 2.0))
        dmax = 0.85 * self.D + radius_frac * self.D

        def cluster(matches):
            ok = [p for p in matches if p.T is not None]
            best = []
            for a in ok:
                grp = [b for b in ok if abs(b.T[2] - a.T[2]) < rot_tol
                       and self._dist(c, a.T, b.T) < tol]
                if len(grp) > len(best) or (len(grp) == len(best) and grp and
                                            sum(p.rel.quality for p in grp) >
                                            sum(p.rel.quality for p in best)):
                    best = grp
            return best

        def mean_T(grp):
            th = float(np.mean([p.T[2] for p in grp]))
            ctr = self._centre(c)
            cm = np.mean([apply_T(p.T, [*ctr, 0])[:2] for p in grp], axis=0)
            t = cm - _rot(th) @ ctr
            return np.array([t[0], t[1], th])

        # round 1: wide window around the manual position
        pairs = self._select_pairs(c, T0, max_pairs, dmax)
        if not pairs:
            return AlignResult(T0, False, 0, 0, 0, 0.0, 0.0, 0.0, [],
                               "The component does not overlap the main mosaic at this "
                               "position - move it closer.")
        m1 = self._match(c, T0, pairs, radius_frac)
        g1 = cluster(m1)
        if not g1:
            return AlignResult(T0, False, len(m1), 0, 0, 0.0, 0.0, 0.0, m1,
                               "No frame pair could be registered within the search "
                               "window - adjust the position or enlarge the window.")
        T1 = mean_T(g1)
        # round 2: verification with a narrow window around the refined pose
        pairs2 = self._select_pairs(c, T1, max_pairs, 0.85 * self.D)
        m2 = self._match(c, T1, pairs2, max(st.link_tolerance, 0.02))
        good = [p for p in m2 if p.T is not None and abs(p.T[2] - T1[2]) < rot_tol
                and self._dist(c, p.T, T1) < tol]
        T = mean_T(good) if good else T1
        matched = [p for p in m2 if p.T is not None]
        for p in m2:
            if p.T is not None:
                p.residual = self._dist(c, p.T, T)
                p.consistent = p.residual < tol and abs(p.T[2] - T[2]) < rot_tol
        cons = [p for p in m2 if p.consistent]
        need = max(1, int(st.manual_min_pairs))
        confirmed = len(cons) >= need and len(cons) >= 0.5 * max(len(matched), 1)
        rms = float(np.sqrt(np.mean([p.residual ** 2 for p in cons]))) if cons else 0.0
        mq = float(np.mean([p.rel.quality for p in cons])) if cons else 0.0
        msg = ("Registration CONFIRMED." if confirmed else
               f"Registration NOT confirmed (needs at least {need} consistent pairs "
               f"and a majority of the matched ones).")
        return AlignResult(T, confirmed, len(m2), len(matched), len(cons), mq, rms,
                           self._dist(c, T, T0), m2, msg)

    # ------------------------------------------------------------- merge
    def merge(self, c: int, T: Sequence[float], method: str, quality: str = "",
              progress: Optional[Callable[[float, str], None]] = None,
              cancel: Optional[Callable[[], bool]] = None) -> MergeOutput:
        """Attach component c with transform T and re-render the merged
        mosaic.  method: 'registered' or 'manual'."""
        assert method in ("registered", "manual")
        v = self.comp_status(c)
        if v["status"] != "unlinked":
            raise ValueError(f"component {c} is not unlinked")
        T = np.asarray(T, float)
        prev_state = json.loads(json.dumps(self.state))
        v.update(status="merged", method=method, T=[float(x) for x in T],
                 quality=quality, time=time.strftime("%Y-%m-%d %H:%M:%S"))
        try:
            ids = self.main_ids()
            frames = []
            for i in ids:
                p = self.world_pose(i)
                frames.append(RenderFrame(os.path.join(self.dir, self.frames[i]["path"]),
                                          p[0], p[1], p[2]))
            r = MosaicRenderer(self.fov, frames, self.st, self.gain)
            tiff = os.path.join(self.out_dir, "mosaic_merged.tif")
            prev = os.path.join(self.out_dir, "preview_merged.jpg")
            tmp = tiff + ".part"
            pv = r.render(tmp, prev, progress, cancel)
            os.replace(tmp, tiff)
        except BaseException:
            self.state = prev_state
            if os.path.exists(tiff + ".part"):
                os.remove(tiff + ".part")
            raise
        ph, pw = pv.shape[:2]
        self.state["outputs"]["merged"] = dict(
            tiff=os.path.relpath(tiff, self.out_dir), preview=os.path.relpath(prev, self.out_dir),
            origin=[float(r.origin[0]), float(r.origin[1])], width=r.width, height=r.height,
            preview_w=pw, preview_h=ph)
        self.state["main_output"] = "merged"
        self.state["merges"].append(dict(component=c, method=method,
                                         T=[float(x) for x in T], quality=quality,
                                         time=v["time"]))
        csv_path = os.path.join(self.out_dir, "frame_positions_merged.csv")
        self._write_csv(csv_path, r)
        self._save()
        with open(os.path.join(self.out_dir, "merge_log.txt"), "a", encoding="utf-8") as fh:
            fh.write(f"{v['time']}  component {c} ({self.frame_info(c)}) attached "
                     f"{'by REGISTRATION' if method == 'registered' else 'MANUALLY POSITIONED'}"
                     f": tx={T[0]:.2f} ty={T[1]:.2f} rot={math.degrees(T[2]):.3f} deg"
                     + (f"; {quality}" if quality else "") + "\n")
        return MergeOutput(tiff, prev, csv_path, r.width, r.height, pv)

    def _save(self):
        p = os.path.join(self.dir, STATE_FILE)
        with open(p + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(self.state, fh, indent=1)
        os.replace(p + ".tmp", p)

    def _write_csv(self, path: str, r: MosaicRenderer):
        ox, oy = r.origin
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["keyframe", "frame", "time_s", "x", "y", "rotation_deg", "quality",
                        "segment", "component", "status"])
            for i in sorted(self.frames):
                fr = self.frames[i]
                c = fr["comp"]
                head = [i, fr["frame"], f"{fr['time']:.3f}"]
                tail = [f"{fr['quality']:.4f}", fr["segment"], c if c >= 0 else ""]
                if c < 0:
                    w.writerow(head + ["", "", ""] + tail + ["dropped"])
                    continue
                T = self._merged_T(c)
                if T is None:
                    w.writerow(head + ["", "", ""] + tail + ["unlinked"])
                    continue
                x, y, th = self.world_pose(i)
                status = "used" if c == 0 else (
                    "merged" if self.comp_status(c)["method"] == "registered"
                    else "merged-manual")
                w.writerow(head + [f"{x - ox:.2f}", f"{y - oy:.2f}",
                                   f"{math.degrees(th):.4f}"] + tail + [status])
