"""Diagnostic output for cross-segment links (re-attaching tracking segments)."""

from __future__ import annotations

import csv
import math
import os
from typing import List

import cv2
import numpy as np

from .keyframes import KeyFrame
from .registration import Registrar, RelPose

_COLS = ["attempt", "segment_size", "status", "main_kf", "seg_kf", "main_frame",
         "seg_frame", "ncc", "second_peak", "quality", "overlap", "tx", "ty",
         "rot_deg", "residual_px"]


def link_preview(reg: Registrar, a: KeyFrame, b: KeyFrame, rel: RelPose,
                 title: str) -> np.ndarray:
    """False-colour overlap of two keyframes at working scale: frame A
    (main mosaic) in magenta, frame B (segment) in green, so correctly
    aligned structure looks grey and misalignment shows as colour fringes.
    Thin outlines show the specimen-only regions used for matching."""
    s = reg.s_work
    cw = reg.c_work
    R = np.array([[math.cos(rel.phi), -math.sin(rel.phi)],
                  [math.sin(rel.phi), math.cos(rel.phi)]])
    t = cw - R @ cw + np.array([rel.dx, rel.dy]) * s       # p_A = R p_B + t
    h, w = a.prep.work.shape
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], float) @ R.T + t
    x0 = int(math.floor(min(0, corners[:, 0].min())))
    y0 = int(math.floor(min(0, corners[:, 1].min())))
    x1 = int(math.ceil(max(w, corners[:, 0].max())))
    y1 = int(math.ceil(max(h, corners[:, 1].max())))
    W, H = x1 - x0, y1 - y0
    Ma = np.array([[1, 0, -x0], [0, 1, -y0]], np.float64)
    Mb = np.hstack([R, (t - [x0, y0])[:, None]])
    fov = reg.mask_w
    ga = cv2.warpAffine(a.prep.work, Ma, (W, H))
    gb = cv2.warpAffine(b.prep.work, Mb, (W, H))
    fa = cv2.warpAffine(fov, Ma, (W, H), flags=cv2.INTER_NEAREST) > 0
    fb = cv2.warpAffine(fov, Mb, (W, H), flags=cv2.INTER_NEAREST) > 0
    img = np.zeros((H, W, 3), np.uint8)
    img[..., 0] = np.where(fa, ga, 0)      # B channel  (magenta = R+B)
    img[..., 2] = np.where(fa, ga, 0)      # R channel
    img[..., 1] = np.where(fb, gb, 0)      # G channel
    # specimen-only regions used for matching
    for frame, M, col in ((a, Ma, (255, 0, 255)), (b, Mb, (0, 255, 0))):
        core = reg.link_data(frame.prep, cache=False).core_w
        cm = cv2.warpAffine(core, M, (W, H), flags=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, col, 1, cv2.LINE_AA)
    bar = np.full((44, W, 3), 32, np.uint8)
    y = 17
    for line in title.split("\n")[:2]:
        cv2.putText(bar, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1,
                    cv2.LINE_AA)
        y += 19
    return np.vstack([bar, img])


def write_link_diagnostics(out_dir: str, records: List[dict], kfs: List[KeyFrame],
                           reg: Registrar, max_images: int = 60) -> str:
    """segment_links.csv + one overlap preview per cross-segment match."""
    d = os.path.join(out_dir, "diagnostics")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "segment_links.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(_COLS + ["preview"])
        # accepted first, then rejected, then outliers; best scores first
        order = {"accepted": 0, "rejected": 1, "outlier": 2}
        recs = sorted(records, key=lambda r: (r["attempt"], order.get(r["status"], 3),
                                              -r["quality"]))
        for k, r in enumerate(recs):
            name = ""
            if k < max_images:
                name = (f"link_a{r['attempt']:02d}_{r['status']}_kf{r['main_kf']}"
                        f"-kf{r['seg_kf']}.jpg")
                title = (f"attempt {r['attempt']}  {r['status'].upper()}  main kf {r['main_kf']} "
                         f"(frame {r['main_frame']})  <-  segment kf {r['seg_kf']} "
                         f"(frame {r['seg_frame']})\n"
                         f"ncc {r['ncc']:.3f} (2nd {r['second_peak']:.3f})  "
                         f"ECC {r['quality']:.3f}  overlap {r['overlap']:.2f}  "
                         f"residual {r['residual_px']:.1f} px")
                img = link_preview(reg, kfs[r["main_kf"]], kfs[r["seg_kf"]], r["rel"], title)
                cv2.imwrite(os.path.join(d, name), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            row = []
            for c in _COLS:
                v = r[c]
                row.append(f"{v:.4f}" if isinstance(v, float) else v)
            wr.writerow(row + [name])
    return path
