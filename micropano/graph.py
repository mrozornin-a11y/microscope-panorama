"""Pose graph: cross-pass matching and global optimisation.

Every keyframe i has a pose (x_i, y_i, theta_i) mapping its local
coordinates into mosaic (world) coordinates:  X = R(theta_i) x + t_i.
An edge (i, j, phi, d) states x_i = R(phi) x_j + d, i.e.

    theta_j = theta_i + phi,      t_j = t_i + R(theta_i) d.

The graph contains sequential edges (from tracking), link edges (re-attaching
segments after the tracking was lost) and loop edges (frames of neighbouring
passes, found by predicted position).  All poses are solved jointly with a
robust least-squares, after which inconsistent edges are removed and the
problem is re-solved; frames that lose all reliable edges are dropped.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .config import Settings
from .keyframes import Edge, KeyFrame
from .registration import Registrar, RelPose


def _rot(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s], [s, c]])


def relative_pose(poses: np.ndarray, i: int, j: int) -> RelPose:
    """Predicted pose of j in the frame of i."""
    d = _rot(poses[i, 2]).T @ (poses[j, :2] - poses[i, :2])
    return RelPose(float(d[0]), float(d[1]), float(poses[j, 2] - poses[i, 2]))


def components(n: int, edges: Sequence[Edge]) -> np.ndarray:
    if not edges:
        return np.arange(n)
    i = np.array([e.i for e in edges])
    j = np.array([e.j for e in edges])
    g = sparse.coo_matrix((np.ones(len(edges)), (i, j)), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    return lab


def integrate(n: int, edges: Sequence[Edge], root: int,
              poses: Optional[np.ndarray] = None) -> np.ndarray:
    """Breadth-first composition of edges starting from `root`.
    Nodes not reachable from root get NaN (unless given in `poses`)."""
    out = np.full((n, 3), np.nan) if poses is None else poses.copy()
    if np.isnan(out[root]).any():
        out[root] = 0.0
    adj: Dict[int, List[Tuple[int, Edge]]] = {}
    for e in edges:
        adj.setdefault(e.i, []).append((e.j, e))
        adj.setdefault(e.j, []).append((e.i, e))
    seen = {root}
    queue = [root]
    while queue:
        u = queue.pop(0)
        for v, e in adj.get(u, []):
            if v in seen:
                continue
            seen.add(v)
            if not np.isnan(out[v]).any() and poses is not None and v != root:
                queue.append(v)
                continue
            d = np.array([e.pose.dx, e.pose.dy])
            if e.i == u:          # forward: v = j
                th = out[u, 2] + e.pose.phi
                t = out[u, :2] + _rot(out[u, 2]) @ d
            else:                 # backward: v = i, u = j
                th = out[u, 2] - e.pose.phi
                t = out[u, :2] - _rot(th) @ d
            out[v] = [t[0], t[1], th]
            queue.append(v)
    return out


def solve(poses0: np.ndarray, edges: Sequence[Edge], nodes: Sequence[int],
          use_rotation: bool, D: float) -> np.ndarray:
    """Robust non-linear least squares over the poses of `nodes`.
    The first node is the gauge (fixed)."""
    nodes = list(nodes)
    if len(nodes) < 2:
        return poses0.copy()
    pid = {n: k for k, n in enumerate(nodes)}
    E = [e for e in edges if e.i in pid and e.j in pid]
    if not E:
        return poses0.copy()
    nv = 3 if use_rotation else 2
    nn = len(nodes)
    I = np.array([pid[e.i] for e in E])
    J = np.array([pid[e.j] for e in E])
    dx = np.array([e.pose.dx for e in E])
    dy = np.array([e.pose.dy for e in E])
    ph = np.array([e.pose.phi for e in E])
    w = np.array([max(0.2, e.pose.quality) for e in E])
    rs = D / 2.0                       # rotation residual scale (px)
    ne = len(E)
    nres = ne * nv + nv
    p0 = poses0[nodes][:, :nv].reshape(-1).copy()
    anchor = p0[:nv].copy()
    big = 1e3

    def unpack(p):
        q = p.reshape(nn, nv)
        th = q[:, 2] if use_rotation else np.zeros(nn)
        return q[:, 0], q[:, 1], th

    def fun(p):
        x, y, th = unpack(p)
        c, s = np.cos(th[I]), np.sin(th[I])
        rx = x[J] - x[I] - (c * dx - s * dy)
        ry = y[J] - y[I] - (s * dx + c * dy)
        parts = [w * rx, w * ry]
        if use_rotation:
            parts.append(w * rs * (th[J] - th[I] - ph))
        r = np.empty(nres)
        r[:ne * nv] = np.stack(parts, axis=1).reshape(-1)
        r[ne * nv:] = big * (p[:nv] - anchor)
        return r

    rows_base = np.arange(ne) * nv

    def jac(p):
        x, y, th = unpack(p)
        rows, cols, vals = [], [], []

        def add(r, c, v):
            rows.append(r)
            cols.append(c)
            vals.append(v)

        # rx
        add(rows_base, J * nv + 0, w)
        add(rows_base, I * nv + 0, -w)
        add(rows_base + 1, J * nv + 1, w)
        add(rows_base + 1, I * nv + 1, -w)
        if use_rotation:
            c, s = np.cos(th[I]), np.sin(th[I])
            add(rows_base, I * nv + 2, w * (s * dx + c * dy))
            add(rows_base + 1, I * nv + 2, -w * (c * dx - s * dy))
            add(rows_base + 2, J * nv + 2, w * rs)
            add(rows_base + 2, I * nv + 2, -w * rs)
        for k in range(nv):
            add(np.array([ne * nv + k]), np.array([k]), np.array([big]))
        return sparse.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(nres, nn * nv))

    f_scale = max(1.0, 0.002 * D)
    res = least_squares(fun, p0, jac=jac, method="trf", tr_solver="lsmr",
                        loss="soft_l1", f_scale=f_scale, max_nfev=200)
    out = poses0.copy()
    q = res.x.reshape(nn, nv)
    out[nodes, 0] = q[:, 0]
    out[nodes, 1] = q[:, 1]
    if use_rotation:
        out[nodes, 2] = q[:, 2]
    else:
        out[nodes, 2] = 0.0
    return out


def edge_residual(poses: np.ndarray, e: Edge) -> float:
    pred = poses[e.i, :2] + _rot(poses[e.i, 2]) @ np.array([e.pose.dx, e.pose.dy])
    return float(np.linalg.norm(poses[e.j, :2] - pred))


def _lens_fraction(d: float, r: float) -> float:
    """Overlap area of two circles of radius r at distance d / circle area."""
    if d >= 2 * r:
        return 0.0
    a = 2 * r * r * math.acos(d / (2 * r)) - 0.5 * d * math.sqrt(max(4 * r * r - d * d, 0))
    return a / (math.pi * r * r)


def max_pair_distance(D: float, min_area_frac: float) -> float:
    r = D / 2
    lo, hi = 0.0, D
    for _ in range(50):
        mid = (lo + hi) / 2
        if _lens_fraction(mid, r) >= min_area_frac:
            lo = mid
        else:
            hi = mid
    return lo


class PoseGraph:
    def __init__(self, keyframes: List[KeyFrame], edges: List[Edge],
                 registrar: Registrar, settings: Settings):
        self.kf = keyframes
        self.edges = list(edges)
        self.reg = registrar
        self.st = settings
        self.D = registrar.D
        self.n = len(keyframes)
        self.use_rot = settings.max_rotation_deg > 0
        self.tested: Set[Tuple[int, int]] = {(min(e.i, e.j), max(e.i, e.j)) for e in edges}
        self.poses = np.full((self.n, 3), np.nan)
        self.active: np.ndarray = np.zeros(self.n, bool)
        self.log: List[str] = []
        self.threads = settings.threads or None

    # ------------------------------------------------------------------
    def _register_many(self, jobs, cancel=None):
        """jobs: list of (i, j, pred RelPose|None, radius_frac|None, strict)."""
        def work(job):
            if cancel and cancel():
                return None
            i, j, pred, rad, strict = job
            return self.reg.register(self.kf[i].prep, self.kf[j].prep, pred, rad, strict)
        with ThreadPoolExecutor(self.threads) as ex:
            return list(ex.map(work, jobs))

    def _main_component(self) -> np.ndarray:
        lab = components(self.n, self.edges)
        counts = np.bincount(lab)
        return np.flatnonzero(lab == int(np.argmax(counts)))

    # ------------------------------------------------------------------
    def link_segments(self, progress=None, cancel=None):
        """Re-attach components disconnected by tracking failures."""
        if self.n == 0:
            return
        for _ in range(20):
            main = self._main_component()
            if len(main) == self.n:
                break
            self.poses = integrate(self.n, self.edges, int(main[0]))
            lab = components(self.n, self.edges)
            main_lab = lab[main[0]]
            others = [c for c in np.unique(lab) if c != main_lab]
            added = False
            for c in sorted(others, key=lambda c: -(lab == c).sum()):
                members = np.flatnonzero(lab == c)
                samples = members[np.linspace(0, len(members) - 1, min(5, len(members))).astype(int)]
                jobs = [(int(m), int(s), None, None, True) for s in samples for m in main]
                # cheap coarse screening first
                scored = []
                for (m, s, *_ ) in jobs:
                    if cancel and cancel():
                        raise InterruptedError
                    cm = self.reg.coarse_match(self.kf[m].prep, self.kf[s].prep, strict=True)
                    if cm is not None:
                        scored.append((cm.peak, m, s, cm))
                scored.sort(reverse=True)
                for peak, m, s, cm in scored[:5]:
                    rel = self.reg.refine(self.kf[m].prep, self.kf[s].prep, cm)
                    if rel is not None:
                        self.edges.append(Edge(m, s, rel, "link"))
                        self.tested.add((min(m, s), max(m, s)))
                        self.log.append(f"segment of {len(members)} keyframes linked via {m}-{s}")
                        added = True
                        break
                if added:
                    break
            if not added:
                break

    def loop_closure(self, radius_frac: float, progress=None, cancel=None) -> int:
        st = self.st
        idx = np.flatnonzero(self.active)
        if len(idx) < 3:
            return 0
        pos = self.poses[idx, :2]
        dmax = max_pair_distance(self.D, st.min_overlap_area * 1.3)
        tree = cKDTree(pos)
        pairs = tree.query_pairs(dmax, output_type="ndarray")
        per_node_near: Dict[int, int] = {}
        per_node_far: Dict[int, int] = {}
        cand = []
        for a, b in pairs:
            i, j = int(idx[a]), int(idx[b])
            if i > j:
                i, j = j, i
            if (i, j) in self.tested:
                continue
            dist = float(np.linalg.norm(pos[a] - pos[b]))
            cand.append((abs(i - j) <= 3, dist, i, j))
        cand.sort(key=lambda c: c[1])
        jobs = []
        for near, dist, i, j in cand:
            book = per_node_near if near else per_node_far
            lim = 2 if near else st.max_neighbours
            if book.get(i, 0) >= lim or book.get(j, 0) >= lim:
                continue
            book[i] = book.get(i, 0) + 1
            book[j] = book.get(j, 0) + 1
            jobs.append((i, j, relative_pose(self.poses, i, j), radius_frac, False))
        if not jobs:
            return 0
        results = []
        chunk = 64
        for k in range(0, len(jobs), chunk):
            if cancel and cancel():
                raise InterruptedError
            results += self._register_many(jobs[k:k + chunk], cancel)
            if progress:
                progress(min(1.0, (k + chunk) / len(jobs)),
                         f"Cross-pass matching {min(k + chunk, len(jobs))}/{len(jobs)}")
        for k in self.kf:
            k.prep._spectra = None           # free memory
        added = 0
        for (i, j, pred, *_), rel in zip(jobs, results):
            self.tested.add((i, j))
            if rel is None:
                continue
            err = math.hypot(rel.dx - pred.dx, rel.dy - pred.dy)
            if err > radius_frac * self.D * 1.2:
                continue
            self.edges.append(Edge(i, j, rel, "loop"))
            added += 1
        return added

    def optimise(self):
        """Solve, remove outlier edges, re-solve; keep main component."""
        thr = self.st.outlier_threshold * self.D
        removed = 0
        for _ in range(30):
            main = self._main_component()
            known = main[~np.isnan(self.poses[main]).any(axis=1)]
            # start from a node whose pose is already known, so that existing
            # poses and newly integrated ones share the same gauge
            root = int(known[0]) if len(known) else int(main[0])
            init = integrate(self.n, self.edges, root,
                             self.poses if len(known) else None)
            # nodes of main component that still have NaN get chain values
            fresh = integrate(self.n, self.edges, root)
            bad = np.isnan(init).any(axis=1)
            init[bad] = fresh[bad]
            # gauge: keep root where it was
            self.poses = solve(init, self.edges, main, self.use_rot, self.D)
            in_main = np.zeros(self.n, bool)
            in_main[main] = True
            res = np.array([edge_residual(self.poses, e) if in_main[e.i] and in_main[e.j] else 0.0
                            for e in self.edges])
            if len(res) == 0 or res.max() <= thr:
                break
            cut = max(thr, 0.5 * res.max())
            keep = res <= cut
            removed += int((~keep).sum())
            self.edges = [e for e, k in zip(self.edges, keep) if k]
        main = self._main_component()
        self.active = np.zeros(self.n, bool)
        self.active[main] = True
        if self.n == 1:
            self.active[:] = True
            self.poses[:] = 0
        self.poses[~self.active] = np.nan
        self.log.append(f"optimisation: removed {removed} inconsistent edges, "
                        f"{self.active.sum()}/{self.n} keyframes kept")
        return removed

    def run(self, progress: Optional[Callable[[float, str], None]] = None,
            cancel: Optional[Callable[[], bool]] = None):
        def sub(a, b):
            if progress is None:
                return None
            return lambda f, m: progress(a + (b - a) * f, m)

        if self.n == 0:
            return
        if progress:
            progress(0.0, "Linking segments")
        self.link_segments(cancel=cancel)
        self.optimise()
        if self.st.loop_closure and self.n > 2:
            added = self.loop_closure(0.2, sub(0.05, 0.5), cancel)
            self.log.append(f"loop closure round 1: {added} edges")
            self.optimise()
            added = self.loop_closure(0.06, sub(0.5, 0.95), cancel)
            self.log.append(f"loop closure round 2: {added} edges")
            self.optimise()
        if progress:
            progress(1.0, "Pose graph optimised")

    # ------------------------------------------------------------------
    def frame_quality(self) -> np.ndarray:
        q = np.zeros(self.n)
        cnt = np.zeros(self.n)
        for e in self.edges:
            for k in (e.i, e.j):
                q[k] += e.pose.quality
                cnt[k] += 1
        return np.where(cnt > 0, q / np.maximum(cnt, 1), 0.0), cnt
