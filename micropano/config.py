"""Processing parameters.

All distances that depend on the optics are expressed as fractions of the
field-of-view diameter, so the same defaults work for any magnification and
video resolution.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from typing import Optional


@dataclass
class Settings:
    # --- field of view -----------------------------------------------------
    # Fraction of the radius trimmed from the edge of the detected circle
    # (the rim of the microscope field is usually dark / blurred / coloured).
    mask_margin: float = 0.04

    # --- frame analysis ----------------------------------------------------
    # How many frames per second of video are analysed (the rest are skipped).
    analysis_fps: float = 10.0
    # A frame is considered blurred if its sharpness is below
    # min_sharpness * (median sharpness of the recent frames).
    min_sharpness: float = 0.6
    # Optional absolute lower bound for the sharpness (variance of Laplacian
    # on the working-scale image); 0 disables it.
    min_sharpness_abs: float = 0.0
    # New keyframe is taken when the stage moved by at least this fraction of
    # the field-of-view diameter relative to the previous keyframe.
    keyframe_min_shift: float = 0.2
    # Expected overlap between neighbouring passes of the snake path
    # (fraction of the diameter). Used to search for cross-pass matches.
    expected_overlap: float = 0.3
    # Maximal rotation between frames, degrees. 0 -> pure translation model.
    max_rotation_deg: float = 0.0
    # Number of consecutive unregistrable frames after which tracking starts
    # a new segment (it is later re-attached through cross-pass matches).
    lost_patience: int = 3

    # --- registration ------------------------------------------------------
    # Diameter (px) of the field of view at the working (fine) scale.
    work_diameter: int = 640
    # Diameter (px) at the coarse scale used for global correlation search.
    coarse_diameter: int = 256
    # Minimal peak of the masked normalised cross-correlation.
    min_ncc: float = 0.2
    # Maximal ratio second-best-peak / best-peak (ambiguity test against
    # repetitive texture).
    max_peak_ratio: float = 0.9
    # Minimal correlation coefficient after fine (ECC) alignment.
    min_quality: float = 0.5
    # Minimal overlap area of two frames (fraction of the mask area).
    min_overlap_area: float = 0.1

    # --- global optimisation -----------------------------------------------
    loop_closure: bool = True
    # Maximal number of cross-pass match candidates tested per keyframe.
    max_neighbours: int = 8
    # Edge whose residual after global optimisation exceeds this fraction of
    # the diameter is considered an outlier and removed.
    outlier_threshold: float = 0.01

    # --- rendering ---------------------------------------------------------
    # 'multiband' (best, slower), 'feather' (fast) or 'none' (hard seams).
    blending: str = "multiband"
    # Number of pyramid levels for multiband blending.
    blend_levels: int = 5
    # Exponent applied to the feather weights (higher -> narrower transition).
    feather_power: float = 1.5
    # Divide frames by the estimated illumination profile (vignetting).
    flat_field: bool = True
    # Interpolation: 'linear', 'cubic' or 'lanczos'.
    interpolation: str = "cubic"
    tile_size: int = 2048
    tiff_compression: str = "zlib"
    preview_max_size: int = 2400
    jpeg_quality: int = 92

    # --- misc --------------------------------------------------------------
    # Where full-resolution keyframes are cached (None -> temporary dir).
    cache_dir: Optional[str] = None
    keep_cache: bool = False
    threads: int = 0  # 0 -> number of CPUs

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    @classmethod
    def from_json(cls, text: str) -> "Settings":
        return cls.from_dict(json.loads(text))
