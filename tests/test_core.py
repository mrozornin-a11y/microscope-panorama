import math
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from micropano.config import Settings
from micropano.fov import FieldOfView, detect_fov
from micropano.graph import solve, edge_residual, PoseGraph
from micropano.keyframes import Edge
from micropano.registration import Registrar, RelPose
from tests.synthetic import make_specimen


@pytest.fixture(scope="module")
def specimen():
    return make_specimen(1600, 1200, seed=1)


def _frame(spec, fov, sx, sy, theta=0.0):
    c, s = math.cos(theta), math.sin(theta)
    M = np.array([[c, -s, sx - c * fov.cx + s * fov.cy],
                  [s, c, sy - s * fov.cx - c * fov.cy]])
    fr = cv2.warpAffine(spec, M, (fov.frame_w, fov.frame_h),
                        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    yy, xx = np.mgrid[0:fov.frame_h, 0:fov.frame_w]
    fr[(xx - fov.cx) ** 2 + (yy - fov.cy) ** 2 > fov.radius ** 2] = 0
    return fr


def test_detect_fov_truncated_circle(specimen):
    fov = FieldOfView(330.4, 236.2, 260, 640, 480)
    fr = _frame(specimen, fov, 800, 600)
    det = detect_fov([fr])
    assert abs(det.cx - fov.cx) < 2.5
    assert abs(det.cy - fov.cy) < 2.5
    assert abs(det.radius - fov.radius) < 3


@pytest.mark.parametrize("rot", [0.0, 1.5])
def test_pairwise_registration(specimen, rot):
    fov = FieldOfView(320, 240, 220, 640, 480)
    st = Settings(max_rotation_deg=3.0 if rot else 0.0)
    reg = Registrar(fov, st)
    a = _frame(specimen, fov, 700, 600)
    b = _frame(specimen, fov, 700 + 150.3, 600 + 60.7, math.radians(rot))
    pa, pb = reg.prepare(fov.crop(a)), reg.prepare(fov.crop(b))
    rel = reg.register(pa, pb)
    assert rel is not None
    # frame b local x maps to specimen R x + (sx_b, sy_b); in frame a local
    # coordinates that is x_a = R x + (150.3, 60.7)
    assert abs(rel.dx - 150.3) < 0.5
    assert abs(rel.dy - 60.7) < 0.5
    assert abs(math.degrees(rel.phi) - rot) < 0.1
    assert rel.quality > 0.8


def test_registration_rejects_unrelated(specimen):
    fov = FieldOfView(320, 240, 200, 640, 480)
    reg = Registrar(fov, Settings())
    other = make_specimen(1600, 1200, seed=7)
    a = _frame(specimen, fov, 700, 600)
    b = _frame(other, fov, 700, 600)
    assert reg.register(reg.prepare(fov.crop(a)), reg.prepare(fov.crop(b))) is None


def test_graph_outlier_removal():
    rng = np.random.default_rng(0)
    n = 30
    gt = np.zeros((n, 3))
    gt[:, 0] = (np.arange(n) % 10) * 100.0
    gt[:, 1] = (np.arange(n) // 10) * 300.0
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(gt[j, :2] - gt[i, :2]) < 400:
                d = gt[j, :2] - gt[i, :2] + rng.normal(0, 0.3, 2)
                edges.append(Edge(i, j, RelPose(d[0], d[1], 0.0, 0.9)))
    # one gross outlier and drifted initial guess
    edges.append(Edge(3, 14, RelPose(500.0, -80.0, 0.0, 0.9), "loop"))

    class _Reg:
        D = 700.0

    class _KF:
        def __init__(self, i):
            self.id = i

    g = PoseGraph([_KF(i) for i in range(n)], edges, _Reg(), Settings())
    g.poses = gt + np.c_[rng.normal(0, 20, (n, 2)), np.zeros(n)]
    g.optimise()
    assert g.active.all()
    assert not any(e.i == 3 and e.j == 14 and e.kind == "loop" for e in g.edges)
    est = g.poses[:, :2] - g.poses[0, :2]
    err = np.linalg.norm(est - (gt[:, :2] - gt[0, :2]), axis=1)
    assert err.max() < 1.0


def test_end_to_end(tmp_path):
    from micropano.pipeline import Pipeline
    from tests.evaluate import evaluate
    from tests.synthetic import make_video
    import tifffile

    video = str(tmp_path / "v.mp4")
    gt = str(tmp_path / "gt.csv")
    make_video(video, width=640, height=360, radius=190, n_pass=3, speed=9.0,
               gt_path=gt, specimen_size=(1500, 900), seed=2)
    out = str(tmp_path / "out")
    res = Pipeline(video, out, Settings()).run()
    assert os.path.exists(res.tiff_path) and os.path.exists(res.preview_path)
    n, rms, mx = evaluate(res.csv_path, gt)
    assert n == res.n_keyframes
    assert rms < 0.5 and mx < 1.5
    img = tifffile.imread(res.tiff_path)
    assert img.shape == (res.height, res.width, 3)


# ---------------------------------------------------------------- linking
class _FakePrep:
    def __init__(self, i):
        self.id = i
        self.frac = 1.0
        self._link = None


class _FakeKF:
    def __init__(self, i, segment=0):
        self.id = i
        self.frame_index = 10 * i
        self.time = i / 3.0
        self.segment = segment
        self.prep = _FakePrep(i)


class _FakeLinkReg:
    """Returns predefined cross-segment matches {(main, seg): RelPose}."""
    D = 1000.0

    def __init__(self, table, local=None):
        self.table = table
        self.local = local or {}      # {(a, b): true RelPose} for local search
        self.local_calls = []

    def link_local(self, a, b, pred, radius_frac):
        from micropano.registration import CoarseMatch
        self.local_calls.append((a.id, b.id))
        rel = self.local.get((a.id, b.id))
        if rel is None or math.hypot(rel.dx - pred.dx, rel.dy - pred.dy) > radius_frac * self.D:
            return None, None
        return CoarseMatch(0.0, 0.0, 0.7, 0.1, 0.4), rel

    def link_data(self, prep, cache=True):
        return prep

    def link_coarse(self, la, lb):
        from micropano.registration import CoarseMatch
        if (la.id, lb.id) in self.table:
            return CoarseMatch(0.0, 0.0, 0.8, 0.1, 0.4)
        return None

    def link_refine(self, a, b, cm):
        return self.table.get((a.id, b.id))


def _two_segments(n_main=10, n_seg=10, step=200.0):
    """Main: kfs 0..n_main-1 along x at y=0; segment: next n_seg along x at
    y=600 (true positions), connected only internally."""
    gt = {}
    edges = []
    for i in range(n_main):
        gt[i] = np.array([i * step, 0.0])
    for k in range(n_seg):
        gt[n_main + k] = np.array([k * step, 600.0])
    for i in range(n_main + n_seg - 1):
        if i == n_main - 1:
            continue
        d = gt[i + 1] - gt[i]
        edges.append(Edge(i, i + 1, RelPose(d[0], d[1], 0.0, 0.95)))
    return gt, edges


def _true(gt, m, s):
    d = gt[s] - gt[m]
    return RelPose(d[0], d[1], 0.0, 0.9)


def _graph(table, n_main=10, n_seg=10, local=None, **kw):
    gt, edges = _two_segments(n_main, n_seg)
    st = Settings(link_samples=n_seg, **kw)
    kfs = [_FakeKF(i, 0 if i < n_main else 1) for i in range(n_main + n_seg)]
    g = PoseGraph(kfs, edges, _FakeLinkReg(table, local), st)
    return g, gt


def _all_true_local(gt, max_dist=800.0):
    """Local registration succeeds (with the true pose) for every pair of
    frames that really overlap."""
    return {(a, b): _true(gt, a, b) for a in gt for b in gt
            if a != b and np.linalg.norm(gt[b] - gt[a]) < max_dist}


def test_link_rejects_single_strong_match():
    gt, _ = _two_segments()
    # one excellent but wrong match (e.g. top edge matched to bottom edge)
    table = {(2, 13): RelPose(0.0, -1500.0, 0.0, 0.99)}
    g, gt = _graph(table)
    g.link_segments()
    assert not any(e.kind == "link" for e in g.edges)
    assert g.warnings and "NOT linked" in g.warnings[0]
    assert all(r["status"] == "rejected" for r in g.link_records)


def test_link_accepts_three_consistent_matches():
    gt, _ = _two_segments()
    pairs = [(1, 11), (4, 14), (7, 17)]
    table = {p: _true(gt, *p) for p in pairs}
    table[(2, 13)] = RelPose(0.0, -1500.0, 0.0, 0.99)       # wrong one
    g, gt = _graph(table)
    g.link_segments()
    links = [(e.i, e.j) for e in g.edges if e.kind == "link"]
    assert sorted(links) == sorted(pairs)
    assert not g.warnings
    st = {(r["main_kf"], r["seg_kf"]): r["status"] for r in g.link_records}
    assert st[(2, 13)] == "outlier"
    g.optimise()
    assert g.active.all()
    est = g.poses[:, :2] - g.poses[0, :2]
    for i, p in gt.items():
        assert np.linalg.norm(est[i] - (p - gt[0])) < 1e-3


def test_link_rejects_too_few_or_dependent_matches():
    gt, _ = _two_segments()
    # three matches, but all on adjacent keyframes -> not independent
    pairs = [(4, 13), (4, 14), (5, 14)]
    g, gt = _graph({p: _true(gt, *p) for p in pairs})
    g.link_segments()
    assert not any(e.kind == "link" for e in g.edges)
    assert "independent match" in g.warnings[0]


def test_link_rejects_conflicting_groups():
    gt, _ = _two_segments()
    good = [(1, 11), (4, 14), (7, 17)]
    table = {p: _true(gt, *p) for p in good}
    # a second, equally supported but shifted hypothesis (repetitive structure)
    for m, s in [(2, 12), (5, 15), (8, 18)]:
        d = gt[s] - gt[m] + np.array([0.0, -400.0])
        table[(m, s)] = RelPose(d[0], d[1], 0.0, 0.9)
    g, gt = _graph(table)
    g.link_segments()
    assert not any(e.kind == "link" for e in g.edges)
    assert "ambiguous" in g.warnings[0]


def test_link_data_excludes_background_and_edge_band(specimen):
    fov = FieldOfView(320, 240, 220, 640, 480)
    st = Settings()
    reg = Registrar(fov, st)
    fr = _frame(specimen, fov, 700, 600)
    fr[:240] = 0                              # upper half: black background
    ld = reg.link_data(reg.prepare(fov.crop(fr)), cache=False)
    core = ld.core_w > 0
    cy = reg.c_work[1]
    band = st.link_edge_band * reg.D * reg.s_work
    ys = np.nonzero(core)[0]
    # nothing from the background nor from the band along the specimen edge
    assert ys.min() > cy + band - 1
    assert 0.25 < ld.frac < 0.5
    assert np.all(ld.work_hp[~core] == 0)


def test_link_anchor_confirmed_by_neighbouring_pairs():
    """All global matches hit the same segment keyframe (one independent
    confirmation only) - the anchor is verified by local registration of
    neighbouring pairs predicted from it."""
    gt, _ = _two_segments()
    table = {p: _true(gt, *p) for p in [(3, 14), (4, 14), (5, 14)]}
    g, gt = _graph(table, local=_all_true_local(gt))
    g.link_segments()
    links = [e for e in g.edges if e.kind == "link"]
    assert len(links) >= 3 and not g.warnings
    segs = {e.j for e in links}
    assert len(segs) >= 3              # verification used other segment keyframes
    rec = [r for r in g.link_records if r["source"] == "verify"]
    assert rec and all(r["status"] == "accepted" for r in rec)
    g.optimise()
    assert g.active.all()
    est = g.poses[:, :2] - g.poses[0, :2]
    for i, p in gt.items():
        assert np.linalg.norm(est[i] - (p - gt[0])) < 1e-3


def test_link_false_anchor_not_confirmed():
    gt, _ = _two_segments()
    # one excellent but wrong match; local registration only finds the truth
    table = {(2, 13): RelPose(0.0, -1500.0, 0.0, 0.99)}
    g, gt = _graph(table, local=_all_true_local(gt))
    g.link_segments()
    assert not any(e.kind == "link" for e in g.edges)
    assert "no anchor was confirmed" in g.warnings[0]
    st = {r["status"] for r in g.link_records}
    assert st <= {"rejected", "verify-failed"}


def _gap_layout(n_a=8, n_b=8, step=200.0, skip=1):
    """Stage moves along x; tracking is lost for `skip` keyframe steps."""
    gt, edges = {}, []
    for i in range(n_a):
        gt[i] = np.array([i * step, 0.0])
    for k in range(n_b):
        gt[n_a + k] = np.array([(n_a - 1 + skip + 1 + k) * step, 0.0])
    for i in range(n_a + n_b - 1):
        if i == n_a - 1:
            continue
        d = gt[i + 1] - gt[i]
        edges.append(Edge(i, i + 1, RelPose(d[0], d[1], 0.0, 0.95)))
    return gt, edges


def test_gap_recovery_links_adjacent_segments():
    gt, edges = _gap_layout()
    kfs = [_FakeKF(i, 0 if i < 8 else 1) for i in range(16)]
    for k in kfs:                      # constant speed, keyframes 1/3 s apart
        k.time = gt[k.id][0] / 600.0
    reg = _FakeLinkReg({}, _all_true_local(gt))
    g = PoseGraph(kfs, edges, reg, Settings())
    g.link_segments()
    assert any(e.kind == "link" for e in g.edges)
    assert any("gap recovery" in m and "linked" in m for m in g.log)
    assert any(r["source"] == "gap" and r["status"] == "accepted" for r in g.link_records)
    g.optimise()
    assert g.active.all()
    est = g.poses[:, :2] - g.poses[0, :2]
    for i, p in gt.items():
        assert np.linalg.norm(est[i] - (p - gt[0])) < 1e-3


def test_unlinked_component_kept_separately():
    gt, _ = _two_segments()
    g, gt = _graph({})
    g.link_segments()
    g.optimise()
    assert g.active[:10].all() and not g.active[10:].any()
    assert (g.comp[10:] == 1).all() and g.kept.all()
    # the unlinked component is solved in its own frame
    est = g.poses[10:, :2] - g.poses[10, :2]
    for k in range(10):
        assert np.linalg.norm(est[k] - (gt[10 + k] - gt[10])) < 1e-3


def test_gap_recovery_real_frames(tmp_path):
    """Sequential chain broken in the middle of a pass (frames still
    overlap): local gap recovery re-links it with correct poses."""
    import csv
    from micropano.keyframes import KeyframeSelector
    from micropano.video import VideoSource
    from tests.synthetic import make_video

    video = str(tmp_path / "v.mp4")
    gtp = str(tmp_path / "gt.csv")
    make_video(video, width=640, height=360, radius=190, n_pass=2, speed=9.0,
               gt_path=gtp, specimen_size=(1500, 900), seed=4, border=0.3)
    gt = {int(r["frame"]): np.array([float(r["x"]), float(r["y"])])
          for r in csv.DictReader(open(gtp))}
    st = Settings()
    v = VideoSource(video)
    fov = detect_fov(v.sample_frames(10))
    reg = Registrar(fov, st)
    kfs, edges = KeyframeSelector(v, fov, st, reg, str(tmp_path)).run()
    cut = len(kfs) // 3
    edges = [e for e in edges if not (e.i == cut - 1 and e.j == cut)]
    for k in kfs[cut:]:
        k.segment += 1
    g = PoseGraph(kfs, edges, reg, st)
    g.link_segments()
    links = [e for e in g.edges if e.kind == "link"]
    assert links and not g.warnings
    assert any("gap recovery" in m and "linked" in m for m in g.log)
    for e in links:
        true = gt[kfs[e.j].frame_index] - gt[kfs[e.i].frame_index]
        assert np.linalg.norm(true - [e.pose.dx, e.pose.dy]) < 1.0


def test_unlinked_component_saved(tmp_path):
    """Two passes that do not overlap: the second one cannot be linked and
    is written as a separate TIFF instead of being silently dropped."""
    import csv
    from micropano.pipeline import Pipeline
    from tests.synthetic import make_video, snake_path

    n = len(snake_path(210, 1290, 210, 2, 1.45 * 380, 9.0, pause_every=90))
    first = len(snake_path(210, 1290, 210, 1, 1.45 * 380, 9.0, pause_every=90))
    video = str(tmp_path / "v.mp4")
    make_video(video, width=640, height=360, radius=190, n_pass=2, speed=9.0,
               specimen_size=(1500, 1000), seed=5, border=0.3, pass_step=1.45,
               jumps=[(first / n, int(1.45 * 380 / 9.0) + 4)])
    res = Pipeline(video, str(tmp_path / "out"), Settings(blending="feather")).run()
    assert len(res.components) == 1
    c = res.components[0]
    assert os.path.exists(c.tiff_path) and os.path.exists(c.preview_path)
    assert c.n_keyframes >= 3
    assert any("saved separately" in w for w in res.warnings)
    rows = list(csv.DictReader(open(res.csv_path)))
    assert sum(r["status"] == "unlinked" for r in rows) == c.n_keyframes
    assert all(r["component"] == "1" for r in rows if r["status"] == "unlinked")


# ------------------------------------------------------- manual attachment
def test_transform_from_points():
    from micropano.merge import transform_from_points, apply_T
    T = np.array([120.0, -40.0, math.radians(2.0)])
    comp = np.array([[0.0, 0.0], [500.0, 30.0], [100.0, 400.0]])
    main = np.array([apply_T(T, [*p, 0.0])[:2] for p in comp])
    est, rms = transform_from_points(main, comp)
    assert np.allclose(est, T, atol=1e-9) and rms < 1e-9
    est1, _ = transform_from_points(main[:1], comp[:1])        # 1 pair: shift only
    assert est1[2] == 0.0 and np.allclose(est1[:2], main[0] - comp[0])


def test_manual_attach_refine_and_merge(tmp_path):
    import csv
    import hashlib
    from micropano.merge import MergeSession
    from micropano.pipeline import Pipeline
    from tests.synthetic import make_video

    video = str(tmp_path / "v.mp4")
    gtp = str(tmp_path / "gt.csv")
    make_video(video, width=640, height=360, radius=190, n_pass=3, speed=9.0, gt_path=gtp,
               specimen_size=(1500, 900), seed=6, border=0.3, jumps=[(0.5, 45)])
    out = str(tmp_path / "out")
    # automatic linking made impossible -> unlinked components that do overlap
    st = Settings(blending="feather", link_min_matches=99, link_verify_min=99)
    res = Pipeline(video, out, st).run()
    assert res.components and res.merge_dir and MergeSession.available(out)
    gt = {int(r["frame"]): np.array([float(r["x"]), float(r["y"])])
          for r in csv.DictReader(open(gtp))}
    ses = MergeSession(out)
    c = max(ses.unlinked(), key=lambda k: len(ses.comp_ids(k)))

    def true_T(c):
        a = np.mean([ses.world_pose(i)[:2] - gt[ses.frames[i]["frame"]]
                     for i in ses.main_ids()], 0)
        b = np.mean([ses.pose(i)[:2] - gt[ses.frames[i]["frame"]] for i in ses.comp_ids(c)], 0)
        return np.array([*(a - b), 0.0])

    Tt = true_T(c)
    # far away: no overlap, nothing to confirm
    far = ses.refine(c, Tt + [5 * ses.D, 0, 0])
    assert not far.confirmed and far.n_matched == 0
    # rough manual position: refined to the true one and confirmed
    r = ses.refine(c, Tt + [0.12 * ses.D, -0.08 * ses.D, math.radians(1.0)], 0.3)
    assert r.confirmed and r.n_consistent >= 2
    assert np.linalg.norm(r.T[:2] - Tt[:2]) < 1.0 and abs(r.T[2]) < math.radians(0.05)

    h = hashlib.md5(open(res.tiff_path, "rb").read()).hexdigest()
    mo = ses.merge(c, r.T, "registered", r.summary())
    assert os.path.exists(mo.tiff_path) and os.path.basename(mo.tiff_path) == "mosaic_merged.tif"
    assert hashlib.md5(open(res.tiff_path, "rb").read()).hexdigest() == h   # untouched
    rows = list(csv.DictReader(open(mo.csv_path)))
    used = [r_ for r_ in rows if r_["status"] in ("used", "merged")]
    assert sum(r_["status"] == "merged" for r_ in rows) == len(ses.comp_ids(c))
    est = np.array([[float(r_["x"]), float(r_["y"])] for r_ in used])
    ref = np.array([gt[int(r_["frame"])] for r_ in used])
    e = (est - est.mean(0)) - (ref - ref.mean(0))
    assert np.sqrt((e ** 2).sum(1).mean()) < 0.5

    # state survives re-opening
    ses2 = MergeSession(out)
    assert ses2.comp_status(c)["status"] == "merged"
    assert c not in ses2.unlinked()


def test_manual_attach_manual_position(tmp_path):
    """Merging without confirmed registration is allowed but marked."""
    import csv
    from micropano.merge import MergeSession
    from micropano.pipeline import Pipeline
    from tests.synthetic import make_video

    video = str(tmp_path / "v.mp4")
    make_video(video, width=640, height=360, radius=190, n_pass=2, speed=9.0,
               specimen_size=(1500, 900), seed=6, border=0.3, jumps=[(0.6, 45)])
    out = str(tmp_path / "out")
    st = Settings(blending="feather", link_min_matches=99, link_verify_min=99)
    res = Pipeline(video, out, st).run()
    assert res.components
    ses = MergeSession(out)
    c = ses.unlinked()[0]
    mo = ses.merge(c, [10.0, 2000.0, 0.0], "manual", "manual position")
    rows = list(csv.DictReader(open(mo.csv_path)))
    assert sum(r["status"] == "merged-manual" for r in rows) == len(ses.comp_ids(c))
    assert "MANUALLY POSITIONED" in open(os.path.join(out, "merge_log.txt")).read()
    assert MergeSession(out).comp_status(c)["method"] == "manual"
