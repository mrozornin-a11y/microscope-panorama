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

    # --- linking of disconnected tracking segments ------------------------
    # A segment is attached to the main mosaic only if at least this many
    # independent cross-segment matches agree on one relative transform.
    link_min_matches: int = 3
    # Keyframes of a segment sampled for cross-segment matching.
    link_samples: int = 12
    # Best coarse candidates per sampled keyframe that are refined.
    link_candidates_per_sample: int = 3
    # Matches agree if the segment positions they imply differ by less than
    # this fraction of the diameter.
    link_tolerance: float = 0.03
    # Band along the outer specimen boundary excluded from cross-segment
    # matching (fraction of the diameter).
    link_edge_band: float = 0.05
    # Minimal share of the field of view covered by specimen (after removing
    # background and the edge band) for a keyframe to be used for linking.
    link_min_content: float = 0.15
    # Grey level below which pixels count as background around the specimen.
    link_dark_level: float = 25.0
    # Anchor verification: a single good cross-segment match is only a
    # hypothesis; this many neighbouring frame pairs (predicted from it) are
    # registered locally, and at least link_verify_min of them must confirm
    # the same transform within link_tolerance.
    link_verify_pairs: int = 4
    link_verify_min: int = 2
    # Local gap recovery between temporally adjacent tracking segments:
    # search radius around the position extrapolated from the stage motion
    # before the loss (fraction of the diameter), and how many keyframes on
    # each side of the gap are tried.
    link_gap_window: float = 0.35
    link_gap_keyframes: int = 3
    # Save CSV and overlap previews of cross-segment matches.
    link_diagnostics: bool = True
    # Components that could not be linked are rendered as separate TIFFs if
    # they have at least this many keyframes (smaller ones are dropped).
    save_unlinked: bool = True
    min_component_keyframes: int = 3

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
