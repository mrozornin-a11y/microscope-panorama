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
