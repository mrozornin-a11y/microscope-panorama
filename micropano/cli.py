"""Command-line interface:  python -m micropano video.mov -o out_dir"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import fields

from .config import Settings
from .fov import FieldOfView, detect_fov
from .pipeline import Pipeline
from .video import VideoSource


def _add_settings_args(p: argparse.ArgumentParser):
    g = p.add_argument_group("advanced settings")
    for f in fields(Settings):
        name = "--" + f.name.replace("_", "-")
        default = f.default
        if isinstance(default, bool):
            g.add_argument(name, type=lambda s: s.lower() in ("1", "true", "yes", "on"),
                           default=None, metavar="BOOL", help=f"(default: {default})")
        elif default is None:
            g.add_argument(name, type=str, default=None, help="(default: none)")
        else:
            g.add_argument(name, type=type(default), default=None,
                           help=f"(default: {default})")


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="micropano",
        description="Build a panorama of a thin/polished section from a microscope video.")
    p.add_argument("video", help="input .mov / .mp4")
    p.add_argument("-o", "--out", default=None,
                   help="output directory (default: <video name>_mosaic)")
    p.add_argument("--circle", type=float, nargs=3, metavar=("CX", "CY", "R"),
                   help="field-of-view circle in pixels (skip auto detection)")
    p.add_argument("--config", help="JSON file with settings")
    p.add_argument("--detect-only", action="store_true",
                   help="only detect the circle and save fov.jpg")
    _add_settings_args(p)
    a = p.parse_args(argv)

    st = Settings()
    if a.config:
        with open(a.config, encoding="utf-8") as fh:
            st = Settings.from_json(fh.read())
    for f in fields(Settings):
        v = getattr(a, f.name)
        if v is not None:
            setattr(st, f.name, v)

    out = a.out or os.path.splitext(a.video)[0] + "_mosaic"
    os.makedirs(out, exist_ok=True)
    video = VideoSource(a.video)
    if a.circle:
        fov = FieldOfView(a.circle[0], a.circle[1], a.circle[2],
                          video.width, video.height, st.mask_margin)
    else:
        fov = detect_fov(video.sample_frames(15), st.mask_margin)
    import cv2
    from .fov import draw_fov
    cv2.imwrite(os.path.join(out, "fov.jpg"), draw_fov(video.first_frame(), fov))
    print(f"circle: cx={fov.cx:.1f} cy={fov.cy:.1f} r={fov.radius:.1f}")
    if a.detect_only:
        return 0

    last = [-1]

    def progress(frac, msg):
        pct = int(frac * 100)
        line = f"\r[{'#' * (pct // 4):<25}] {pct:3d}%  {msg[:60]:<60}"
        sys.stdout.write(line)
        sys.stdout.flush()
        last[0] = pct

    res = Pipeline(a.video, out, st, fov, progress).run()
    print()
    for line in res.log:
        print("  " + line)
    print(f"mosaic: {res.tiff_path} ({res.width} x {res.height})")
    print(f"preview: {res.preview_path}")
    print(f"positions: {res.csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
