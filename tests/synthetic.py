"""Synthetic microscope video for testing.

A random 'rock' texture (grains + fine texture + cracks) is moved under a
fixed circular field of view along a snake path; the video contains
vignetting, sensor noise, brightness flicker, motion-blurred frames and
pauses.  Ground-truth stage positions are written alongside.
"""

from __future__ import annotations

import argparse
import csv
import math

import cv2
import numpy as np


def make_specimen(w: int, h: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # grains: Voronoi cells computed at low resolution
    s = 4
    lw, lh = w // s, h // s
    n = int(w * h / 90000) + 50
    pts = rng.uniform([0, 0], [lw, lh], size=(n, 2)).astype(np.float32)
    lab = np.zeros((lh, lw), np.int32)
    best = np.full((lh, lw), np.inf, np.float32)
    yy, xx = np.mgrid[0:lh, 0:lw].astype(np.float32)
    for k, (px, py) in enumerate(pts):
        d = (xx - px) ** 2 + (yy - py) ** 2
        upd = d < best
        best[upd] = d[upd]
        lab[upd] = k
    lab = cv2.resize(lab.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    colors = rng.uniform(60, 220, size=(n, 3)).astype(np.float32)
    img = colors[lab]
    # grain boundaries
    edges = cv2.Laplacian(lab.astype(np.float32), cv2.CV_32F) != 0
    edges = cv2.dilate(edges.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    img[edges] *= 0.35
    # multi-scale texture
    for sigma, amp in ((1.0, 18), (3.0, 14), (12.0, 10)):
        noise = rng.normal(0, 1, (h, w)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), sigma)
        noise /= noise.std() + 1e-6
        img += amp * noise[..., None] * rng.uniform(0.7, 1.0, 3).astype(np.float32)
    # cracks
    for _ in range(int(w * h / 400000) + 3):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        ang = rng.uniform(0, 2 * math.pi)
        pts_l = []
        for _ in range(40):
            pts_l.append((int(x), int(y)))
            ang += rng.normal(0, 0.3)
            x += 25 * math.cos(ang)
            y += 25 * math.sin(ang)
        cv2.polylines(img, [np.array(pts_l)], False, (30, 30, 40), 2, cv2.LINE_AA)
    return np.clip(img, 0, 255).astype(np.uint8)


def snake_path(x0, x1, y0, n_pass, pass_step, speed, pause_every=0):
    """Positions (stage offsets) per frame for a snake path."""
    pos = []
    y = y0
    for p in range(n_pass):
        xa, xb = (x0, x1) if p % 2 == 0 else (x1, x0)
        n = int(abs(xb - xa) / speed)
        for k in range(n):
            pos.append((xa + (xb - xa) * k / n, y))
            if pause_every and k % pause_every == pause_every // 2:
                pos.extend([pos[-1]] * 10)  # stage stopped for a while
        if p < n_pass - 1:
            m = int(pass_step / speed)
            for k in range(m):
                pos.append((xb, y + pass_step * k / m))
            y += pass_step
    return pos


def make_video(path: str, width=1280, height=720, radius=380, seed=0,
               n_pass=4, speed=14.0, rot_deg=0.0, fps=30.0,
               blur_prob=0.08, gt_path=None, specimen_size=(3600, 2400), jump=0):
    rng = np.random.default_rng(seed + 1)
    sw, sh = specimen_size
    spec = make_specimen(sw, sh, seed)
    cx, cy = width / 2 + 7.3, height / 2 - 4.1
    D = 2 * radius
    pass_step = 0.62 * D
    margin = radius + 20
    path_xy = snake_path(margin, sw - margin, margin, n_pass, pass_step, speed, pause_every=90)
    if jump:
        # stage moved abruptly: drop `jump` frames in the middle of pass 2
        k0 = len(path_xy) * 3 // 8
        path_xy = path_xy[:k0] + path_xy[k0 + jump:]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    circle = (rr <= radius).astype(np.float32)
    circle = cv2.GaussianBlur(circle, (0, 0), 2.0)
    vign = (1.0 - 0.35 * (rr / radius) ** 2) * circle
    vign = np.clip(vign, 0, 1)[..., None]
    wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not wr.isOpened():
        raise RuntimeError("cannot open video writer")
    gt = []
    rot = math.radians(rot_deg)
    for k, (sx, sy) in enumerate(path_xy):
        jx, jy = rng.normal(0, 0.3, 2)
        sx, sy = sx + jx, sy + jy
        th = rot * math.sin(k / 50.0)
        c, s = math.cos(th), math.sin(th)
        # frame pixel p (relative to circle centre) shows specimen at R p + (sx, sy)
        M = np.array([[c, -s, sx - c * cx + s * cy],
                      [s, c, sy - s * cx - c * cy]], np.float64)
        fr = cv2.warpAffine(spec, M, (width, height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_REFLECT)
        fr = fr.astype(np.float32)
        if rng.random() < blur_prob:
            L = int(rng.integers(9, 25))
            ker = np.zeros((L, L), np.float32)
            ker[L // 2, :] = 1.0 / L
            ker = cv2.warpAffine(ker, cv2.getRotationMatrix2D((L / 2, L / 2), rng.uniform(0, 180), 1), (L, L))
            ker /= ker.sum()
            fr = cv2.filter2D(fr, -1, ker)
        fr *= vign * (1.0 + rng.normal(0, 0.02))
        fr += rng.normal(0, 3.0, fr.shape).astype(np.float32)
        wr.write(np.clip(fr, 0, 255).astype(np.uint8))
        gt.append((k, sx, sy, math.degrees(th)))
    wr.release()
    if gt_path:
        with open(gt_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "x", "y", "rot_deg"])
            w.writerows(gt)
    return gt, (cx, cy, radius), spec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--gt", default=None)
    ap.add_argument("--rot", type=float, default=0.0)
    ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jump", type=int, default=0)
    a = ap.parse_args()
    gt, circ, _ = make_video(a.out, gt_path=a.gt, rot_deg=a.rot, n_pass=a.passes, seed=a.seed,
                            jump=a.jump)
    print(f"{len(gt)} frames, circle {circ}")
