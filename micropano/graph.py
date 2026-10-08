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
        self.warnings: List[str] = []
        self.link_records: List[dict] = []
        self.unlinked_reasons: Dict[Tuple[int, ...], str] = {}
        self.comp = np.zeros(self.n, int)       # 0 = main, k>0 unlinked, -1 dropped
        self.kept = np.zeros(self.n, bool)
        self._prev_lab: Optional[np.ndarray] = None
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
    def _component_poses(self) -> np.ndarray:
        """Poses of every connected component in its own coordinate frame."""
        lab = components(self.n, self.edges)
        P = np.full((self.n, 3), np.nan)
        for c in np.unique(lab):
            P = integrate(self.n, self.edges, int(np.flatnonzero(lab == c)[0]), P)
        return P

    def link_segments(self, progress=None, cancel=None):
        """Re-attach components disconnected by tracking failures.

        1. Local gap recovery for temporally adjacent tracking segments: the
           stage motion before the loss predicts where the next segment
           starts; a few keyframes on both sides of the gap are registered in
           an enlarged window around that prediction.
        2. General cross-segment search of the remaining components against
           the main component (specimen-only global correlation).

        Any single match is only an anchor hypothesis.  A component is
        attached if either `link_min_matches` independent matches of the
        search agree, or the anchor is confirmed by local registration of
        `link_verify_min` neighbouring frame pairs predicted from it.
        Otherwise the component is kept separately and a warning is issued.
        """
        if self.n == 0:
            return
        self._attempt = getattr(self, "_attempt", 0)
        self.unlinked_reasons: Dict[Tuple[int, ...], str] = {}
        self._gap_recovery(progress, cancel)

        failed: Dict[Tuple[int, ...], Tuple[int, str]] = {}
        while True:
            main = self._main_component()
            if len(main) == self.n:
                break
            P = self._component_poses()
            lab = components(self.n, self.edges)
            main_lab = lab[main[0]]
            others = sorted((np.flatnonzero(lab == c) for c in np.unique(lab) if c != main_lab),
                            key=lambda m: -len(m))
            linked = False
            for members in others:
                key = tuple(int(m) for m in members)
                if key in failed and failed[key][0] == len(main):
                    continue          # nothing changed since the last attempt
                if progress:
                    progress(0.0, f"Linking segment of {len(members)} keyframes")
                ok, reason = self._try_link(members, main, P, cancel)
                if ok:
                    self.log.append(reason)
                    linked = True
                    break
                failed[key] = (len(main), reason)
            if not linked:
                break
        # report what stayed disconnected
        main = self._main_component()
        lab = components(self.n, self.edges)
        small = 0
        for c in np.unique(lab):
            members = np.flatnonzero(lab == c)
            if lab[main[0]] == c:
                continue
            key = tuple(int(m) for m in members)
            reason = failed.get(key, (0, "not attempted"))[1]
            self.unlinked_reasons[key] = reason
            if len(members) < self.st.min_component_keyframes:
                small += len(members)
                continue
            f0, f1 = self.kf[members[0]].frame_index, self.kf[members[-1]].frame_index
            self.warnings.append(
                f"tracking segment of {len(members)} keyframes (video frames {f0}-{f1}) "
                f"was NOT linked to the main mosaic: {reason}")
        if small:
            self.warnings.append(f"{small} keyframes in very short tracking segments "
                                 f"were not linked and are dropped")

    # ------------------------------------------------------------------
    def _candidate(self, m, s, rel, cm, P, source):
        """Match of segment keyframe s to main keyframe m and the transform
        (segment frame -> main frame) it implies."""
        th = P[m, 2] + rel.phi
        t = P[m, :2] + _rot(P[m, 2]) @ np.array([rel.dx, rel.dy])
        thT = th - P[s, 2]
        tT = t - _rot(thT) @ P[s, :2]
        return dict(m=int(m), s=int(s), rel=rel, cm=cm, th=thT, t=tT, source=source,
                    res=float("nan"))

    def _record(self, attempt, seg_size, c, status):
        nan = float("nan")
        cm, rel = c.get("cm"), c["rel"]
        self.link_records.append(dict(
            attempt=attempt, segment_size=seg_size, source=c["source"],
            main_kf=c["m"], seg_kf=c["s"],
            main_frame=self.kf[c["m"]].frame_index, seg_frame=self.kf[c["s"]].frame_index,
            ncc=cm.peak if cm is not None else nan,
            second_peak=cm.second if cm is not None else nan,
            quality=rel.quality if c.get("measured", True) else nan,
            overlap=rel.overlap if c.get("measured", True) else nan,
            tx=c["t"][0], ty=c["t"][1], rot_deg=math.degrees(c["th"]),
            residual_px=c["res"], status=status, rel=rel,
            predicted=not c.get("measured", True)))

    def _verify(self, anchor, main_nodes, seg_nodes, P, cancel=None):
        """Predict neighbouring frame pairs from the anchor's transform and
        register them locally.  Returns (confirmations, tested pairs)."""
        st, reg = self.st, self.reg
        tol = st.link_tolerance * self.D
        thT, tT = anchor["th"], anchor["t"]
        R = _rot(thT)
        main_nodes = [int(m) for m in main_nodes]
        mpos = P[main_nodes, :2]
        # segment keyframes closest (in time) to the anchor, anchor included
        seg_near = sorted((int(s) for s in seg_nodes),
                          key=lambda s: abs(s - anchor["s"]))[:3 * st.link_verify_pairs]
        dmax = max_pair_distance(self.D, 1.5 * st.min_overlap_area)
        options = []
        for s in seg_near:
            ps = R @ P[s, :2] + tT
            d = np.linalg.norm(mpos - ps, axis=1)
            for k in np.flatnonzero(d < dmax):
                m = main_nodes[k]
                if m == anchor["m"] and s == anchor["s"]:
                    continue
                # prefer pairs that share no image with the anchor, then
                # pairs close to the anchor in time and with large overlap
                shared = int(m == anchor["m"]) + int(s == anchor["s"])
                options.append((shared, abs(s - anchor["s"]), float(d[k]), m, s, ps))
        options.sort(key=lambda o: o[:3])
        pairs, per_s, used = [], {}, set()
        for shared, _, _, m, s, ps in options:
            if per_s.get(s, 0) >= 2 or (m, s) in used:
                continue
            pairs.append((m, s, ps))
            per_s[s] = per_s.get(s, 0) + 1
            used.add((m, s))
            if len(pairs) >= st.link_verify_pairs:
                break
        conf, tested = [], []
        for m, s, ps in pairs:
            if cancel and cancel():
                raise InterruptedError
            dpred = _rot(P[m, 2]).T @ (ps - P[m, :2])
            pred = RelPose(float(dpred[0]), float(dpred[1]), float(thT + P[s, 2] - P[m, 2]))
            cm, rel = reg.link_local(self.kf[m].prep, self.kf[s].prep, pred,
                                     st.link_tolerance)
            c = self._candidate(m, s, rel if rel is not None else pred, cm, P, "verify")
            c["measured"] = rel is not None
            if rel is not None:
                c["res"] = float(math.hypot(rel.dx - pred.dx, rel.dy - pred.dy))
            if rel is not None and c["res"] <= tol:
                conf.append(c)
            tested.append(c)
        return conf, tested

    def _decide(self, cands, main_nodes, seg_nodes, P, seg_size, cancel=None,
                what="segment") -> Tuple[bool, str]:
        """Accept a link between two components from candidate matches."""
        st = self.st
        self._attempt += 1
        attempt = self._attempt
        need = max(1, int(st.link_min_matches))
        need_v = max(1, int(st.link_verify_min))
        tol = st.link_tolerance * self.D
        rot_tol = math.radians(max(1.0, st.max_rotation_deg))
        if not cands:
            return False, "no reliable specimen match found"
        centre = np.nanmean(P[list(seg_nodes), :2], axis=0)

        def mapped(c):
            return _rot(c["th"]) @ centre + c["t"]

        def agree(a, b):
            return (abs(a["th"] - b["th"]) < rot_tol and
                    np.linalg.norm(mapped(a) - mapped(b)) < tol)

        def support(group):
            """Independent matches: distinct, non-adjacent keyframes on both
            sides."""
            def count(ids):
                n, last = 0, None
                for i in sorted(set(ids)):
                    if last is None or i - last >= 2:
                        n += 1
                        last = i
                return n
            return min(count(c["s"] for c in group), count(c["m"] for c in group))

        def best_cluster(pool):
            best, best_key = [], (0, 0.0)
            for c in pool:
                grp = [d for d in pool if agree(c, d)]
                key = (support(grp), sum(d["rel"].quality for d in grp))
                if key > best_key:
                    best, best_key = grp, key
            return best, best_key[0]

        def residuals(group, pool):
            thT = float(np.mean([c["th"] for c in group]))
            cT = np.mean([mapped(c) for c in group], axis=0)
            tT = cT - _rot(thT) @ centre
            for c in pool:
                c["res"] = float(np.linalg.norm(
                    _rot(thT) @ P[c["s"], :2] + tT - (_rot(c["th"]) @ P[c["s"], :2] + c["t"])))

        def accept(group, extra, how):
            uniq, seen = [], set()
            for c in group:                    # one edge per frame pair
                if (c["m"], c["s"]) not in seen:
                    seen.add((c["m"], c["s"]))
                    uniq.append(c)
            group[:] = uniq
            done = []
            for c in cands + extra:
                if any(c is d for d in group):
                    if any(c is d for d in done):
                        continue
                    done.append(c)
                    self._record(attempt, seg_size, c, "accepted")
                    self.edges.append(Edge(c["m"], c["s"], c["rel"], "link"))
                    self.tested.add((min(c["m"], c["s"]), max(c["m"], c["s"])))
                elif c.get("measured", True):
                    self._record(attempt, seg_size, c,
                                 "outlier" if c["source"] != "verify" else "verify-failed")
                else:
                    self._record(attempt, seg_size, c, "verify-failed")
            resid = max(c["res"] for c in group)
            return True, (f"{what} of {seg_size} keyframes linked {how}; "
                          f"{len(group)} matches, max residual {resid:.1f} px: "
                          + ", ".join(f"{c['m']}-{c['s']}" for c in group[:8])
                          + (" ..." if len(group) > 8 else ""))

        # (a) independent consensus of the search results
        inl, sup = best_cluster(cands)
        residuals(inl, cands)
        inl = [c for c in inl if c["res"] <= tol]
        sup = support(inl)
        rest = [c for c in cands if not any(c is d for d in inl)]
        rival, rival_sup = best_cluster(rest)
        if sup >= need and rival_sup < need:
            return accept(inl, [], f"by {sup} independent matches")

        # (b) anchor hypotheses verified by predicted neighbouring pairs
        anchors = sorted(cands, key=lambda c: -(c["rel"].quality * max(c["rel"].ncc, 0.01)))
        verified = []          # (anchor group, confirmations)
        all_tested = []
        for a in anchors[:6]:
            if any(agree(a, g[0][0]) for g in verified):
                continue      # same hypothesis already verified
            conf, tested = self._verify(a, main_nodes, seg_nodes, P, cancel)
            all_tested += tested
            if len(conf) >= need_v and len(conf) >= 0.5 * len(tested):
                group = [a] + [c for c in cands if c is not a and agree(a, c)] + conf
                verified.append((group, conf))
        if len(verified) == 1:
            group, conf = verified[0]
            residuals([group[0]], [c for c in group if c["source"] != "verify"])
            group = [c for c in group if c["res"] <= tol]
            return accept(group, all_tested,
                          f"by anchor {group[0]['m']}-{group[0]['s']} confirmed by "
                          f"{len(conf)} neighbouring pair(s)")
        for c in cands:
            self._record(attempt, seg_size, c, "rejected")
        for c in all_tested:
            self._record(attempt, seg_size, c, "verify-failed")
        if len(verified) > 1:
            return False, (f"ambiguous: {len(verified)} different link hypotheses were "
                           f"confirmed by neighbouring pairs")
        if rival_sup >= need:
            return False, (f"ambiguous: two inconsistent groups of matches "
                           f"({sup} vs {rival_sup} independent) and none confirmed locally")
        n_conf = max((sum(1 for c in all_tested if c.get("measured") and c["res"] <= tol)), 0)
        return False, (f"only {sup} independent match(es) of {need} required "
                       f"({len(cands)} candidate matches), and no anchor was confirmed "
                       f"by {need_v} neighbouring pairs ({len(all_tested)} pairs tested, "
                       f"{n_conf} consistent)")

    # ------------------------------------------------------------------
    def _gap_recovery(self, progress=None, cancel=None):
        """Link temporally adjacent tracking segments using the stage motion
        before the loss of tracking."""
        st, reg = self.st, self.reg
        segs: Dict[int, List[int]] = {}
        for k in self.kf:
            segs.setdefault(k.segment, []).append(k.id)
        order = sorted(segs)
        for sa, sb in zip(order, order[1:]):
            A, B = segs[sa], segs[sb]
            lab = components(self.n, self.edges)
            if lab[A[-1]] == lab[B[0]]:
                continue
            if cancel and cancel():
                raise InterruptedError
            if progress:
                progress(0.0, f"Gap recovery between tracking segments {sa} and {sb}")
            P = self._component_poses()
            L, F = A[-1], B[0]
            tail = A[-4:]
            v = np.zeros(2)
            dt_tail = self.kf[tail[-1]].time - self.kf[tail[0]].time
            if len(tail) >= 2 and dt_tail > 0:
                v = (P[tail[-1], :2] - P[tail[0], :2]) / dt_tail
            dt = max(0.0, self.kf[F].time - self.kf[L].time)
            pF = P[L, :2] + v * dt               # predicted position of F (frame of A)
            radius = st.link_gap_window * self.D + 0.5 * float(np.linalg.norm(v * dt))
            gk = max(1, int(st.link_gap_keyframes))
            jobs = []
            for i in A[-gk:]:
                for j in B[:gk]:
                    pj = pF + (P[j, :2] - P[F, :2])
                    d = _rot(P[i, 2]).T @ (pj - P[i, :2])
                    if np.linalg.norm(d) - radius > self.D:
                        continue                  # cannot overlap anywhere in the window
                    jobs.append((i, j, RelPose(float(d[0]), float(d[1]), 0.0)))

            def work(job):
                i, j, pred = job
                return reg.link_local(self.kf[i].prep, self.kf[j].prep, pred,
                                      radius / self.D)

            with ThreadPoolExecutor(self.threads) as ex:
                results = list(ex.map(work, jobs))
            cands = []
            for (i, j, pred), (cm, rel) in zip(jobs, results):
                if rel is not None:
                    cands.append(self._candidate(i, j, rel, cm, P, "gap"))
            ca = np.flatnonzero(lab == lab[A[-1]])
            cb = np.flatnonzero(lab == lab[B[0]])
            ok, reason = self._decide(cands, ca, cb, P, len(cb), cancel,
                                      what=f"gap recovery: segment {sb}")
            if ok:
                self.log.append(reason)
            else:
                self.log.append(f"gap recovery between segments {sa} and {sb} failed "
                                f"({len(jobs)} pairs tried): {reason}")

    def _try_link(self, members: np.ndarray, main: np.ndarray, P: np.ndarray,
                  cancel=None) -> Tuple[bool, str]:
        st, reg = self.st, self.reg
        if len(members) < 2:
            return False, "single keyframe"

        # sampled keyframes of the segment (specimen data cached for them)
        seg = [int(i) for i in members
               if reg.link_data(self.kf[i].prep, cache=False).frac >= st.link_min_content]
        if not seg:
            return False, "too little specimen (rest is background/edge) in the frames"
        samples = [seg[k] for k in np.unique(np.linspace(0, len(seg) - 1,
                                                          min(st.link_samples, len(seg))).astype(int))]
        sdata = {s: reg.link_data(self.kf[s].prep) for s in samples}
        stride = max(1, int(math.ceil(len(main) / 250)))
        ref = [int(m) for m in main[::stride]]

        # 1. coarse specimen-only correlation of every main keyframe against
        #    every sample (main keyframes are processed one at a time, so their
        #    spectra are never all held in memory)
        def screen(m):
            if cancel and cancel():
                return []
            ld = reg.link_data(self.kf[m].prep, cache=False)
            if ld.frac < st.link_min_content:
                return []
            out = []
            for s in samples:
                cm = reg.link_coarse(ld, sdata[s])
                if cm is not None:
                    out.append((cm.peak, m, s, cm))
            return out

        with ThreadPoolExecutor(self.threads) as ex:
            found = [c for lst in ex.map(screen, ref) for c in lst]
        for s in samples:
            self.kf[s].prep._link = None
        if cancel and cancel():
            raise InterruptedError
        coarse = []
        for s in samples:
            best = sorted((c for c in found if c[2] == s), key=lambda c: -c[0])
            coarse += best[:st.link_candidates_per_sample]

        # 2. fine specimen-only ECC
        def fine(c):
            peak, m, s, cm = c
            return reg.link_refine(self.kf[m].prep, self.kf[s].prep, cm)

        with ThreadPoolExecutor(self.threads) as ex:
            rels = list(ex.map(fine, coarse))
        cands = [self._candidate(m, s, rel, cm, P, "global")
                 for (peak, m, s, cm), rel in zip(coarse, rels) if rel is not None]
        if not cands:
            return False, (f"no reliable specimen match found "
                           f"({len(coarse)} coarse candidates, none passed fine alignment)")
        return self._decide(cands, main, members, P, len(members), cancel)

    def loop_closure(self, radius_frac: float, progress=None, cancel=None) -> int:
        st = self.st
        idx = np.flatnonzero(self.kept)
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
            if (i, j) in self.tested or self.comp[i] != self.comp[j]:
                continue          # poses of different components are unrelated
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
        """Solve every connected component (each in its own frame), remove
        outlier edges, re-solve.  The largest component is the main mosaic;
        other components with at least `min_component_keyframes` keyframes
        are kept as separate (unlinked) mosaics, smaller ones are dropped."""
        thr = self.st.outlier_threshold * self.D
        removed = 0
        for _ in range(30):
            lab = components(self.n, self.edges)
            newp = np.full((self.n, 3), np.nan)
            for c in np.unique(lab):
                nodes = np.flatnonzero(lab == c)
                P0 = self.poses.copy()
                known = nodes[~np.isnan(P0[nodes]).any(axis=1)]
                if len(known) and self._prev_lab is not None:
                    # nodes solved earlier in different components have
                    # different gauges: keep only the largest group
                    groups = self._prev_lab[known]
                    vals, cnt = np.unique(groups, return_counts=True)
                    keep_g = vals[int(np.argmax(cnt))]
                    P0[known[groups != keep_g]] = np.nan
                    known = known[groups == keep_g]
                root = int(known[0]) if len(known) else int(nodes[0])
                init = integrate(self.n, self.edges, root, P0 if len(known) else None)
                fresh = integrate(self.n, self.edges, root)
                bad = np.isnan(init).any(axis=1)
                init[bad] = fresh[bad]
                if len(nodes) >= 2:
                    sol = solve(init, self.edges, nodes, self.use_rot, self.D)
                    newp[nodes] = sol[nodes]
                else:
                    newp[nodes] = 0.0
            self.poses = newp
            self._prev_lab = lab
            res = np.array([edge_residual(self.poses, e) for e in self.edges])
            if len(res) == 0 or res.max() <= thr:
                break
            cut = max(thr, 0.5 * res.max())
            keep = res <= cut
            removed += int((~keep).sum())
            self.edges = [e for e, k in zip(self.edges, keep) if k]
        lab = components(self.n, self.edges)
        self._prev_lab = lab
        vals, cnt = np.unique(lab, return_counts=True)
        order = vals[np.argsort(-cnt, kind="stable")]
        self.comp = np.full(self.n, -1, int)
        k = 0
        for rank, c in enumerate(order):
            nodes = lab == c
            if rank == 0:
                self.comp[nodes] = 0
            elif self.st.save_unlinked and nodes.sum() >= self.st.min_component_keyframes:
                k += 1
                self.comp[nodes] = k
        self.active = self.comp == 0
        self.kept = self.comp >= 0
        self.poses[~self.kept] = np.nan
        extra = int((self.comp > 0).sum())
        self.log.append(f"optimisation: removed {removed} inconsistent edges, "
                        f"{int(self.active.sum())}/{self.n} keyframes in the main mosaic"
                        + (f", {extra} in {k} unlinked component(s)" if k else "")
                        + (f", {int((~self.kept).sum())} dropped" if (~self.kept).any() else ""))
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
